import warnings
warnings.filterwarnings("ignore")
import sys
import os

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(current_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
from datetime import datetime
from torch.cuda import amp
from contextlib import nullcontext
import joblib

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

try:
    from train_avg.config import ConfigV2
    from train_avg.utils import CsvLogger, get_edge_index, calc_layer_lengths, batch_time_edge_index
    from train_avg.dataset import EVChargerDatasetV2, create_sliding_windows
    from train_avg.metrics import print_metrics
    from train_avg.evaluation import periodic_evaluate_mae, evaluate_model
except ImportError:
    from config import ConfigV2
    from utils import CsvLogger, get_edge_index, calc_layer_lengths, batch_time_edge_index
    from dataset import EVChargerDatasetV2, create_sliding_windows
    from metrics import print_metrics
    from evaluation import periodic_evaluate_mae, evaluate_model

from model_avg import SpatioTemporalDiffusionModelV2
from scheduler import MultiStageOneCycleLR

def train():
    dist.init_process_group("nccl")
    rank = int(os.environ["RANK"])
    device_id = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(device_id)

    cfg = ConfigV2()

    if rank == 0:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg.RUN_ID = run_id
        print(f"Starting DDP training with Ensemble-K={cfg.ENSEMBLE_K}. Run ID: {cfg.RUN_ID}")
        os.makedirs(os.path.dirname(cfg.MODEL_SAVE_PATH_TEMPLATE), exist_ok=True)
        
        model_save_path_best = cfg.MODEL_SAVE_PATH_TEMPLATE.format(run_id=cfg.RUN_ID, rank="best")
        model_save_path_second_best = cfg.MODEL_SAVE_PATH_TEMPLATE.format(run_id=cfg.RUN_ID, rank="second_best")
        model_save_path_mae_best = cfg.MODEL_SAVE_PATH_TEMPLATE.format(run_id=cfg.RUN_ID, rank="mae_best")
        model_save_path_mae_second_best = cfg.MODEL_SAVE_PATH_TEMPLATE.format(run_id=cfg.RUN_ID, rank="mae_second_best")

        scaler_y_save_path = cfg.SCALER_Y_SAVE_PATH_TEMPLATE.format(run_id=cfg.RUN_ID)
        scaler_mm_save_path = cfg.SCALER_MM_SAVE_PATH_TEMPLATE.format(run_id=cfg.RUN_ID)
        scaler_z_save_path = cfg.SCALER_Z_SAVE_PATH_TEMPLATE.format(run_id=cfg.RUN_ID)

        best_val_loss = float('inf')
        second_best_val_loss = float('inf')
        best_val_mae = float('inf')
        second_best_val_mae = float('inf')
        best_model_path_for_eval = None
        second_best_model_path_for_eval = None
        best_model_path_for_val = None
        second_best_model_path_for_val = None

    run_id_list = [cfg.RUN_ID]
    dist.broadcast_object_list(run_id_list, src=0)
    if rank != 0: 
        cfg.RUN_ID = run_id_list[0]
        scaler_y_save_path = cfg.SCALER_Y_SAVE_PATH_TEMPLATE.format(run_id=cfg.RUN_ID)
        scaler_mm_save_path = cfg.SCALER_MM_SAVE_PATH_TEMPLATE.format(run_id=cfg.RUN_ID)
        scaler_z_save_path = cfg.SCALER_Z_SAVE_PATH_TEMPLATE.format(run_id=cfg.RUN_ID)

    log_headers = ['epoch', 'avg_train_loss', 'avg_val_loss', 'lr', 'avg_val_mae']
    csv_logger = CsvLogger(
        log_dir='./results', 
        run_id=cfg.RUN_ID,  
        rank=rank,          
        headers=log_headers
    )

    adj_matrix = np.load(cfg.ADJ_MATRIX_PATH)
    distances = adj_matrix[adj_matrix > 0]
    sigma = np.std(distances)
    adj_matrix = np.exp(-np.square(adj_matrix) / (sigma**2))
    adj_matrix[adj_matrix < 0.1] = 0

    original_edge_index, original_edge_weights = get_edge_index(adj_matrix)

    train_features = np.load(cfg.TRAIN_FEATURES_PATH)
    val_features = np.load(cfg.VAL_FEATURES_PATH)
    
    train_dataset = EVChargerDatasetV2(train_features, cfg.HISTORY_LEN, cfg.PRED_LEN, cfg)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, sampler=train_sampler, drop_last=True, pin_memory=True, num_workers=4)
    print(len(train_dataset), len(train_dataloader))

    if rank == 0 and cfg.NORMALIZATION_TYPE != "none":
        scaler_y, scaler_mm, scaler_z = train_dataset.get_scaler()
        joblib.dump(scaler_y, scaler_y_save_path)
        joblib.dump(scaler_mm, scaler_mm_save_path)
        joblib.dump(scaler_z, scaler_z_save_path)
    dist.barrier()

    train_y_scaler = joblib.load(scaler_y_save_path) if os.path.exists(scaler_y_save_path) else None
    train_mm_scaler = joblib.load(scaler_mm_save_path) if os.path.exists(scaler_mm_save_path) else None
    train_z_scaler = joblib.load(scaler_z_save_path) if os.path.exists(scaler_z_save_path) else None

    val_dataset = EVChargerDatasetV2(val_features, cfg.HISTORY_LEN, cfg.PRED_LEN, cfg, scaler_y=train_y_scaler, scaler_mm=train_mm_scaler, scaler_z=train_z_scaler)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, sampler=val_sampler, drop_last=True, pin_memory=True, num_workers=4)
    
    val_eval_indices = list(range(len(val_dataset)))
    val_eval_indices_subset = val_eval_indices[:cfg.EVAL_ON_VAL_BATCHES * cfg.BATCH_SIZE]
    val_eval_dataset = Subset(val_dataset, val_eval_indices_subset)
    val_eval_sampler = DistributedSampler(val_eval_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    val_eval_loader = DataLoader(
        val_eval_dataset,
        batch_size=cfg.BATCH_SIZE,
        num_workers=4,
        pin_memory=True,
        sampler=val_eval_sampler
    )

    model = SpatioTemporalDiffusionModelV2(
        in_features=cfg.TARGET_FEAT_DIM, out_features=cfg.TARGET_FEAT_DIM,
        history_features=cfg.HISTORY_FEATURES, static_features=cfg.STATIC_FEATURES, future_known_features=cfg.FUTURE_KNOWN_FEAT_DIM,
        model_dim=cfg.MODEL_DIM, num_heads=cfg.NUM_HEADS, T=cfg.T, depth=cfg.DEPTH
    ).to(device_id)
    
    ddp_model = DDP(model, device_ids=[device_id], find_unused_parameters=False)

    optimizer = optim.AdamW(ddp_model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=1e-4)
    scaler = amp.GradScaler()

    scheduler = MultiStageOneCycleLR(optimizer=optimizer,
                                     total_steps=cfg.EPOCHS,
                                     warmup_ratio=cfg.WARMUP_EPOCHS / cfg.EPOCHS,
                                     cooldown_ratio=cfg.COOLDOWN_EPOCHS / cfg.EPOCHS)

    len_list = calc_layer_lengths(cfg.PRED_LEN, cfg.DEPTH)
    L_max = len_list[0]
    E_orig = original_edge_index.shape[1]

    # Micro-batch edge data
    micro_batch_size_eff = cfg.ENSEMBLE_K
    (micro_full_edge_index, micro_full_edge_weights) = batch_time_edge_index(
        original_edge_index, original_edge_weights, cfg.NUM_NODES, micro_batch_size_eff, L_max, device_id
    )
    micro_edge_data = []
    for L_d in len_list:
        num_edges_needed = E_orig * micro_batch_size_eff * L_d
        slice_idx = micro_full_edge_index[:, :num_edges_needed]
        slice_w = micro_full_edge_weights[:num_edges_needed]
        micro_edge_data.append((slice_idx, slice_w))

    # Validation edge data
    (val_full_edge_index, val_full_edge_weights) = batch_time_edge_index(
        original_edge_index, original_edge_weights, cfg.NUM_NODES, cfg.BATCH_SIZE, L_max, device_id
    )
    val_edge_data = []
    for L_d in len_list:
        num_edges_needed = E_orig * cfg.BATCH_SIZE * L_d
        slice_idx = val_full_edge_index[:, :num_edges_needed]
        slice_w = val_full_edge_weights[:num_edges_needed]
        val_edge_data.append((slice_idx, slice_w))

    original_val_samples = create_sliding_windows(val_features, cfg.HISTORY_LEN, cfg.PRED_LEN)
    y_true_original = np.array([s[1][:, :, :cfg.TARGET_FEAT_DIM] for s in original_val_samples]).squeeze(-1)[:cfg.EVAL_ON_VAL_BATCHES * cfg.BATCH_SIZE]

    for epoch in range(cfg.EPOCHS):
        train_sampler.set_epoch(epoch)
        ddp_model.train()
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS} [Train]", disable=(rank != 0))
        total_train_loss = 0.0
        optimizer.zero_grad()

        for i, (history_c, static_c, future_x0, future_known_c, idx) in enumerate(progress_bar):
            current_batch_size = future_x0.shape[0]
            is_grad_update_step = ((i + 1) % cfg.ACCUMULATION_STEPS == 0) or ((i + 1) == len(train_dataloader))
            batch_loss_tracker = 0.0

            for micro_idx in range(current_batch_size):
                is_last_micro = (micro_idx == current_batch_size - 1)
                if is_grad_update_step and is_last_micro:
                    context = nullcontext()
                else:
                    context = ddp_model.no_sync()
                
                with context:
                    with amp.autocast():
                        hist_micro = history_c[micro_idx:micro_idx+1].to(device_id)
                        stat_micro = static_c[micro_idx:micro_idx+1].to(device_id)
                        future_micro = future_x0[micro_idx:micro_idx+1].to(device_id)
                        known_micro = future_known_c[micro_idx:micro_idx+1].to(device_id)
                        
                        K = cfg.ENSEMBLE_K
                        hist_exp = hist_micro.repeat_interleave(K, dim=0).permute(0, 2, 1, 3)
                        stat_exp = stat_micro.repeat_interleave(K, dim=0)
                        future_exp = future_micro.repeat_interleave(K, dim=0).permute(0, 2, 1, 3)
                        known_exp = known_micro.repeat_interleave(K, dim=0).permute(0, 2, 1, 3)
                        
                        noise = torch.randn_like(future_exp)
                        t_scalar = torch.randint(0, cfg.T, (1,), device=device_id).long()
                        t_exp = t_scalar.repeat(K)
                        
                        sqrt_alpha_bar = ddp_model.module.sqrt_alphas_cumprod[t_exp].view(K, 1, 1, 1)
                        sqrt_one_minus_alpha_bar = ddp_model.module.sqrt_one_minus_alphas_cumprod[t_exp].view(K, 1, 1, 1)
                        x_t = sqrt_alpha_bar * future_exp + sqrt_one_minus_alpha_bar * noise

                        # Gradient checkpointing requires input grad
                        x_t.requires_grad_(True)

                        predicted_noise = ddp_model(
                            x_t, t_exp, hist_exp, stat_exp, known_exp, micro_edge_data, micro_edge_data
                        )
                        
                        # --- Loss Calculation (Energy Score + Mean MSE) ---
                        pred_x0_exp = (x_t - sqrt_one_minus_alpha_bar * predicted_noise) / (sqrt_alpha_bar + 1e-8)
                        
                        # Re-arrange to [1, K, C, N, L]
                        pred_x0_grouped = pred_x0_exp.view(1, K, cfg.NUM_NODES, cfg.PRED_LEN, cfg.TARGET_FEAT_DIM)
                        pred_x0_grouped = pred_x0_grouped.permute(0, 1, 4, 2, 3) 
                        
                        # Target: [1, C, N, L]
                        target_x0 = future_micro.permute(0, 3, 2, 1) 
                        target_x0_expanded = target_x0.unsqueeze(1) 
                        
                        weights = ddp_model.module.alphas_cumprod[t_scalar].view(1, 1, 1, 1, 1)
                        
                        # 1. Mean MSE (Accurate Mean)
                        ensemble_mean = pred_x0_grouped.mean(dim=1) 
                        mse_weights = weights.squeeze(1)
                        loss_mean_mse = ((ensemble_mean - target_x0)**2 * mse_weights).mean()
                        
                        # 2. Individual L1 (Attraction)
                        l1_dist = (pred_x0_grouped - target_x0_expanded).abs()
                        loss_individual_l1 = (l1_dist * weights).mean()
                        
                        # 3. Repulsion (Pairwise Distance)
                        # Expand for pairwise diff: [1, K, 1, C, N, L] - [1, 1, K, C, N, L]
                        x_i = pred_x0_grouped.unsqueeze(2)
                        x_j = pred_x0_grouped.unsqueeze(1)
                        pairwise_dist = (x_i - x_j).abs() 
                        # Apply weights (broadcasts correctly)
                        loss_repulsion = (pairwise_dist * weights.unsqueeze(1)).mean()
                        
                        loss = cfg.MEAN_MSE_LAMBDA * loss_mean_mse + \
                               cfg.INDIVIDUAL_L1_LAMBDA * loss_individual_l1 - \
                               cfg.REPULSION_LAMBDA * loss_repulsion
                        
                        loss = loss / (current_batch_size * cfg.ACCUMULATION_STEPS)

                    scaler.scale(loss).backward()
                    batch_loss_tracker += loss.item() * (current_batch_size * cfg.ACCUMULATION_STEPS)

            if is_grad_update_step:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                avg_loss_step = batch_loss_tracker / current_batch_size
                total_train_loss += avg_loss_step 
                if rank == 0: progress_bar.set_postfix(loss=avg_loss_step)
            else:
                avg_loss_step = batch_loss_tracker / current_batch_size
                total_train_loss += avg_loss_step

        ddp_model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for tensors in val_dataloader:
                history_c, static_c, future_x0, future_known_c = [d.to(device_id) for d in tensors[:4]]
                with amp.autocast():
                    b = future_x0.shape[0]
                    history_c_p = history_c.permute(0, 2, 1, 3)
                    future_x0_p = future_x0.permute(0, 2, 1, 3)
                    future_known_c_p = future_known_c.permute(0, 2, 1, 3)

                    noise = torch.randn_like(future_x0_p)
                    k = torch.randint(0, cfg.T, (b,), device=device_id).long()
                    
                    sqrt_alpha_bar_k = ddp_model.module.sqrt_alphas_cumprod[k].view(b, 1, 1, 1)
                    sqrt_one_minus_alpha_bar_k = ddp_model.module.sqrt_one_minus_alphas_cumprod[k].view(b, 1, 1, 1)
                    x_k = sqrt_alpha_bar_k * future_x0_p + sqrt_one_minus_alpha_bar_k * noise

                    curr_val_edge_data = []
                    for (e_idx, e_w) in val_edge_data:
                         num_edges = E_orig * b * (e_idx.shape[1] // (E_orig * cfg.BATCH_SIZE))
                         curr_val_edge_data.append((e_idx[:, :num_edges], e_w[:num_edges]))

                    pred_noise = ddp_model(
                        x_k, k, history_c_p, static_c, future_known_c_p, curr_val_edge_data, curr_val_edge_data
                    )
                    val_loss = nn.MSELoss()(pred_noise, noise)
                
                total_val_loss += val_loss.item()

        avg_train_loss_local = total_train_loss / len(train_dataloader)
        avg_train_loss_tensor = torch.tensor(avg_train_loss_local).to(device_id)
        dist.all_reduce(avg_train_loss_tensor, op=dist.ReduceOp.SUM)
        avg_train_loss = avg_train_loss_tensor.item() / world_size

        avg_val_loss_local = total_val_loss / len(val_dataloader)
        avg_val_loss_tensor = torch.tensor(avg_val_loss_local).to(device_id)
        dist.all_reduce(avg_val_loss_tensor, op=dist.ReduceOp.SUM)
        avg_val_loss = avg_val_loss_tensor.item() / world_size

        if rank == 0:
            epoch_log_data = {
                'epoch': epoch + 1,
                'avg_train_loss': avg_train_loss,
                'avg_val_loss': avg_val_loss,
                'lr': optimizer.param_groups[0]['lr'],
                'avg_val_mae': '' 
            }
            print(f"Epoch {epoch+1} Summary: Avg Train Loss: {avg_train_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            if avg_val_loss < best_val_loss:
                second_best_val_loss = best_val_loss
                best_val_loss = avg_val_loss
                if best_model_path_for_val is not None and os.path.exists(best_model_path_for_val):
                    os.rename(best_model_path_for_val, model_save_path_second_best)
                    second_best_model_path_for_val = model_save_path_second_best
                torch.save(ddp_model.module.state_dict(), model_save_path_best)
                best_model_path_for_val = model_save_path_best
            elif avg_val_loss < second_best_val_loss:
                second_best_val_loss = avg_val_loss
                torch.save(ddp_model.module.state_dict(), model_save_path_second_best)
                second_best_model_path_for_val = model_save_path_second_best

        run_mae_eval = cfg.EVAL_ON_VAL and (epoch + 1) % cfg.EVAL_ON_VAL_EPOCH == 0
        if run_mae_eval:
            current_val_mae = float('inf') 
            if rank == 0: print(f"Epoch {epoch+1}, running periodic MAE evaluation...")
            current_val_seq_local = periodic_evaluate_mae(
                ddp_model.module, 
                val_eval_loader,
                train_y_scaler,
                original_edge_index,
                original_edge_weights,
                cfg,
                device_id
            )
        
            gathered_results = [None] * world_size
            dist.gather_object(
                current_val_seq_local,
                gathered_results if rank == 0 else None,
                dst=0
            )
            if rank == 0:
                current_val_seq = np.concatenate(gathered_results, axis=0)
                current_val_seq = current_val_seq[:len(y_true_original)]
                current_val_mae = np.mean(np.abs(current_val_seq - y_true_original))
                epoch_log_data['avg_val_mae'] = current_val_mae
                print(f"Avg Val MAE: {current_val_mae:.4f}")

                if current_val_mae < best_val_mae:
                    second_best_val_mae = best_val_mae
                    best_val_mae = current_val_mae
                    if best_model_path_for_eval is not None and os.path.exists(best_model_path_for_eval):
                        if os.path.exists(model_save_path_mae_second_best):
                            os.remove(model_save_path_mae_second_best)
                        os.rename(best_model_path_for_eval, model_save_path_mae_second_best)
                        second_best_model_path_for_val = model_save_path_mae_second_best
                    torch.save(ddp_model.module.state_dict(), model_save_path_mae_best)
                    best_model_path_for_eval = model_save_path_mae_best
                elif current_val_mae < second_best_val_mae:
                    second_best_val_mae = current_val_mae
                    torch.save(ddp_model.module.state_dict(), model_save_path_mae_second_best)
                    second_best_model_path_for_val = model_save_path_mae_second_best

        if rank == 0:
            csv_logger.log_epoch(epoch_log_data)

        dist.barrier()
        scheduler.step()

    dist.barrier()
    
    if rank == 0:
        print("\n" + "="*50 + "\nTraining finished. Starting evaluation on the test set...\n" + "="*50 + "\n")
    
    path_list_to_eval = [best_model_path_for_eval, second_best_model_path_for_eval,best_model_path_for_val,second_best_model_path_for_val] if rank == 0 else [None, None, None, None]
    dist.broadcast_object_list(path_list_to_eval, src=0)

    best_model_path_synced, second_best_model_path_synced, best_model_path_for_val_synced, second_best_model_path_for_val_synced = path_list_to_eval

    if best_model_path_for_val_synced and os.path.exists(best_model_path_for_val_synced):
        if rank == 0: print(f"\n[ALL GPUS] Evaluating BEST VAL model: {os.path.basename(best_model_path_for_val_synced)}")
        metrics_best_val = evaluate_model(cfg, best_model_path_for_val_synced, scaler_y_save_path, scaler_mm_save_path, scaler_z_save_path, device=f"cuda:{device_id}", rank=rank, world_size=world_size,key='best_val')
        if rank == 0: print_metrics(metrics_best_val)

    if second_best_model_path_for_val_synced and os.path.exists(second_best_model_path_for_val_synced):
        if rank == 0: print(f"\n[ALL GPUS] Evaluating 2ND BEST VAL model: {os.path.basename(second_best_model_path_for_val_synced)}")
        metrics_second_best_val = evaluate_model(cfg, second_best_model_path_for_val_synced, scaler_y_save_path, scaler_mm_save_path, scaler_z_save_path, device=f"cuda:{device_id}", rank=rank, world_size=world_size,key='second_best_val')
        if rank == 0: print_metrics(metrics_second_best_val)

    if best_model_path_synced and os.path.exists(best_model_path_synced):
        if rank == 0: print(f"\n[ALL GPUS] Evaluating BEST model: {os.path.basename(best_model_path_synced)}")
        metrics_best = evaluate_model(cfg, best_model_path_synced, scaler_y_save_path, scaler_mm_save_path, scaler_z_save_path, device=f"cuda:{device_id}", rank=rank, world_size=world_size,key='best')
        if rank == 0: print_metrics(metrics_best)

    if second_best_model_path_synced and os.path.exists(second_best_model_path_synced):
        if rank == 0: print(f"\n[ALL GPUS] Evaluating 2ND BEST model: {os.path.basename(second_best_model_path_synced)}")
        metrics_second_best = evaluate_model(cfg, second_best_model_path_synced, scaler_y_save_path, scaler_mm_save_path, scaler_z_save_path, device=f"cuda:{device_id}", rank=rank, world_size=world_size,key='second_best')
        if rank == 0: print_metrics(metrics_second_best)

    dist.destroy_process_group()

if __name__ == "__main__":
    train()