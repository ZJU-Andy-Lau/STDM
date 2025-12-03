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
    from train_lite.config import ConfigV2
    from train_lite.utils import CsvLogger, calc_layer_lengths
    from train_lite.dataset import EVChargerDatasetV2, create_sliding_windows
    from train_lite.metrics import print_metrics
    from train_lite.evaluation import periodic_evaluate_mae, evaluate_model
except ImportError:
    from config import ConfigV2
    from utils import CsvLogger, calc_layer_lengths
    from dataset import EVChargerDatasetV2, create_sliding_windows
    from metrics import print_metrics
    from evaluation import periodic_evaluate_mae, evaluate_model

from model_lite import SpatioTemporalDiffusionModelV2
from scheduler import MultiStageOneCycleLR

def train():
    # --- DDP 初始化 ---
    # 检查是否在 DDP 环境中
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        device_id = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(device_id)
    else:
        # 单机 fallback (用于调试)
        print("Warning: Not running in DDP mode. Fallback to single GPU.")
        rank = 0
        device_id = 0
        world_size = 1
        torch.cuda.set_device(0)

    cfg = ConfigV2()

    if rank == 0:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg.RUN_ID = run_id
        print(f"Starting Training (Lite Mode). Run ID: {cfg.RUN_ID}")
        print(f"Batch Size: {cfg.BATCH_SIZE}, Ensemble K: {cfg.ENSEMBLE_K}")
        
        os.makedirs(os.path.dirname(cfg.MODEL_SAVE_PATH_TEMPLATE), exist_ok=True)
        
        # 路径模板
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

    # 同步 RUN_ID
    if dist.is_initialized():
        run_id_list = [cfg.RUN_ID]
        dist.broadcast_object_list(run_id_list, src=0)
        if rank != 0: 
            cfg.RUN_ID = run_id_list[0]
            scaler_y_save_path = cfg.SCALER_Y_SAVE_PATH_TEMPLATE.format(run_id=cfg.RUN_ID)
            scaler_mm_save_path = cfg.SCALER_MM_SAVE_PATH_TEMPLATE.format(run_id=cfg.RUN_ID)
            scaler_z_save_path = cfg.SCALER_Z_SAVE_PATH_TEMPLATE.format(run_id=cfg.RUN_ID)

    # 日志
    log_headers = ['epoch', 'avg_train_loss', 'avg_val_loss', 'lr', 'avg_val_mae']
    csv_logger = CsvLogger(log_dir='./results', run_id=cfg.RUN_ID, rank=rank, headers=log_headers)

    # --- 加载密集图 ---
    adj_matrix = np.load(cfg.ADJ_MATRIX_PATH)
    distances = adj_matrix[adj_matrix > 0]
    sigma = np.std(distances)
    adj_matrix = np.exp(-np.square(adj_matrix) / (sigma**2))
    adj_matrix[adj_matrix < 0.1] = 0
    # 转换为 Tensor，不生成 edge_index
    adj_tensor = torch.from_numpy(adj_matrix).float().to(device_id)

    # --- 数据加载 ---
    train_features = np.load(cfg.TRAIN_FEATURES_PATH)
    val_features = np.load(cfg.VAL_FEATURES_PATH)
    
    train_dataset = EVChargerDatasetV2(train_features, cfg.HISTORY_LEN, cfg.PRED_LEN, cfg)
    
    if dist.is_initialized():
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        train_sampler = None

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=cfg.BATCH_SIZE, 
        sampler=train_sampler, 
        shuffle=(train_sampler is None),
        drop_last=True, 
        pin_memory=True, 
        num_workers=4
    )
    
    if rank == 0:
        print(f"Train Dataset Size: {len(train_dataset)}, Steps per Epoch: {len(train_dataloader)}")
        if cfg.NORMALIZATION_TYPE != "none":
            scaler_y, scaler_mm, scaler_z = train_dataset.get_scaler()
            joblib.dump(scaler_y, scaler_y_save_path)
            joblib.dump(scaler_mm, scaler_mm_save_path)
            joblib.dump(scaler_z, scaler_z_save_path)
    if dist.is_initialized():
        dist.barrier()

    train_y_scaler = joblib.load(scaler_y_save_path) if os.path.exists(scaler_y_save_path) else None
    train_mm_scaler = joblib.load(scaler_mm_save_path) if os.path.exists(scaler_mm_save_path) else None
    train_z_scaler = joblib.load(scaler_z_save_path) if os.path.exists(scaler_z_save_path) else None

    val_dataset = EVChargerDatasetV2(val_features, cfg.HISTORY_LEN, cfg.PRED_LEN, cfg, scaler_y=train_y_scaler, scaler_mm=train_mm_scaler, scaler_z=train_z_scaler)
    
    if dist.is_initialized():
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        val_sampler = None

    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=cfg.BATCH_SIZE, 
        sampler=val_sampler, 
        drop_last=True, 
        pin_memory=True, 
        num_workers=4
    )
    
    val_eval_indices = list(range(len(val_dataset)))
    val_eval_indices_subset = val_eval_indices[:cfg.EVAL_ON_VAL_BATCHES * cfg.BATCH_SIZE]
    val_eval_dataset = Subset(val_dataset, val_eval_indices_subset)
    
    if dist.is_initialized():
        val_eval_sampler = DistributedSampler(val_eval_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        val_eval_sampler = None

    val_eval_loader = DataLoader(
        val_eval_dataset,
        batch_size=cfg.BATCH_SIZE,
        num_workers=4,
        pin_memory=True,
        sampler=val_eval_sampler
    )

    # --- 模型初始化 ---
    model = SpatioTemporalDiffusionModelV2(
        in_features=cfg.TARGET_FEAT_DIM, out_features=cfg.TARGET_FEAT_DIM,
        history_features=cfg.HISTORY_FEATURES, static_features=cfg.STATIC_FEATURES, future_known_features=cfg.FUTURE_KNOWN_FEAT_DIM,
        model_dim=cfg.MODEL_DIM, num_heads=cfg.NUM_HEADS, T=cfg.T, depth=cfg.DEPTH,
        max_channels=cfg.MAX_CHANNELS 
    ).to(device_id)
    
    if dist.is_initialized():
        ddp_model = DDP(model, device_ids=[device_id], find_unused_parameters=False)
    else:
        ddp_model = model

    optimizer = optim.AdamW(ddp_model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=1e-4)
    scaler = amp.GradScaler()

    scheduler = MultiStageOneCycleLR(optimizer=optimizer,
                                     total_steps=cfg.EPOCHS,
                                     warmup_ratio=cfg.WARMUP_EPOCHS / cfg.EPOCHS,
                                     cooldown_ratio=cfg.COOLDOWN_EPOCHS / cfg.EPOCHS)

    original_val_samples = create_sliding_windows(val_features, cfg.HISTORY_LEN, cfg.PRED_LEN)
    y_true_original = np.array([s[1][:, :, :cfg.TARGET_FEAT_DIM] for s in original_val_samples]).squeeze(-1)[:cfg.EVAL_ON_VAL_BATCHES * cfg.BATCH_SIZE]

    # --- 训练循环 ---
    for epoch in range(cfg.EPOCHS):
        if dist.is_initialized():
            train_sampler.set_epoch(epoch)
            
        ddp_model.train()
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS} [Train]", disable=(rank != 0))
        total_train_loss = 0.0
        batch_loss_tracker = 0.0
        optimizer.zero_grad()

        for i, (history_c, static_c, future_x0, future_known_c, idx) in enumerate(progress_bar):
            # 判断是否是梯度更新步
            is_grad_update_step = ((i + 1) % cfg.ACCUMULATION_STEPS == 0) or ((i + 1) == len(train_dataloader))
            
            # DDP 上下文管理 (Accumulation 时不同步梯度)
            if is_grad_update_step:
                context = nullcontext()
            else:
                if dist.is_initialized():
                    context = ddp_model.no_sync()
                else:
                    context = nullcontext()
            
            with context:
                with amp.autocast():
                    # 1. 数据上 GPU 
                    # DataLoader 输出形状: (Batch, Length, Nodes, Channels)
                    hist_batch = history_c.to(device_id)
                    stat_batch = static_c.to(device_id)
                    future_batch = future_x0.to(device_id)
                    known_batch = future_known_c.to(device_id)
                    
                    B = hist_batch.shape[0]
                    K = cfg.ENSEMBLE_K

                    # 2. 扩展 Batch 以适配 Ensemble，并修正维度顺序
                    # 输入: (B, L, N, C) -> 目标: (B*K, N, L, C)
                    # repeat_interleave 会保留原维度，所以需要 permute
                    
                    # (B, L, N, C) -> (B*K, L, N, C) -> (B*K, N, L, C)
                    hist_exp = hist_batch.repeat_interleave(K, dim=0).permute(0, 2, 1, 3) 
                    
                    # Static: (B, N, F) -> (B*K, N, F) (无 Time 维，无需 permute)
                    stat_exp = stat_batch.repeat_interleave(K, dim=0)
                    
                    # Future & Known: (B, L, N, C) -> (B*K, N, L, C)
                    future_exp = future_batch.repeat_interleave(K, dim=0).permute(0, 2, 1, 3)
                    known_exp = known_batch.repeat_interleave(K, dim=0).permute(0, 2, 1, 3)
                    
                    # 3. 扩散过程
                    noise = torch.randn_like(future_exp) # Shape: (B*K, N, L, C)
                    
                    # 生成时间步 t
                    # 为每个 Batch 生成一个 t，然后重复 K 次
                    t_per_sample = torch.randint(0, cfg.T, (B,), device=device_id).long()
                    t_exp = t_per_sample.repeat_interleave(K) # [t1, t1, t2, t2...]

                    # 获取 alpha_bar
                    if dist.is_initialized():
                        sqrt_alpha_bar = ddp_model.module.sqrt_alphas_cumprod[t_exp].view(B*K, 1, 1, 1)
                        sqrt_one_minus_alpha_bar = ddp_model.module.sqrt_one_minus_alphas_cumprod[t_exp].view(B*K, 1, 1, 1)
                    else:
                        sqrt_alpha_bar = ddp_model.sqrt_alphas_cumprod[t_exp].view(B*K, 1, 1, 1)
                        sqrt_one_minus_alpha_bar = ddp_model.sqrt_one_minus_alphas_cumprod[t_exp].view(B*K, 1, 1, 1)
                    
                    x_t = sqrt_alpha_bar * future_exp + sqrt_one_minus_alpha_bar * noise
                    x_t.requires_grad_(True)

                    # 4. 模型预测 (传入密集图 adj_tensor)
                    # 输入形状现在已确认为 (B*K, N, L, C)，与模型要求一致
                    predicted_noise = ddp_model(
                        x_t, t_exp, hist_exp, stat_exp, known_exp, adj_tensor
                    )
                    
                    # 5. Loss 计算 (Batch Ensemble Loss)
                    pred_x0_exp = (x_t - sqrt_one_minus_alpha_bar * predicted_noise) / (sqrt_alpha_bar + 1e-8)
                    
                    # Reshape 以分组: (B, K, N, L, C)
                    pred_x0_grouped = pred_x0_exp.view(B, K, cfg.NUM_NODES, cfg.PRED_LEN, cfg.TARGET_FEAT_DIM)
                    
                    # Target 准备: 
                    # future_batch 是 (B, L, N, C)，必须转为 (B, N, L, C) 才能与 pred_x0_grouped (N, L) 对齐
                    target_x0 = future_batch.permute(0, 2, 1, 3) # (B, N, L, C)
                    target_x0_expanded = target_x0.unsqueeze(1) # (B, 1, N, L, C)
                    
                    # Weights: (B*K) -> (B, K, 1, 1, 1) -> (B, 1, 1, 1, 1)
                    if dist.is_initialized():
                        alphas_cumprod = ddp_model.module.alphas_cumprod
                    else:
                        alphas_cumprod = ddp_model.alphas_cumprod
                        
                    weights = alphas_cumprod[t_exp].view(B, K, 1, 1, 1)
                    weights = weights[:, 0:1, :, :, :] 
                    
                    # (1) Mean MSE
                    ensemble_mean = pred_x0_grouped.mean(dim=1, keepdim=True) 
                    loss_mean_mse = ((ensemble_mean - target_x0_expanded)**2 * weights).mean()
                    
                    # (2) Individual L1
                    l1_dist = (pred_x0_grouped - target_x0_expanded).abs()
                    loss_individual_l1 = (l1_dist * weights).mean()
                    
                    # (3) Repulsion
                    x_i = pred_x0_grouped.unsqueeze(2)
                    x_j = pred_x0_grouped.unsqueeze(1)
                    pairwise_dist = (x_i - x_j).abs() 
                    loss_repulsion = (pairwise_dist * weights.unsqueeze(1)).mean()

                    bias = pred_x0_grouped - target_x0_expanded
                    sum_bias = bias.sum(dim=1, keepdim=True) # (B, 1, N, L, C)
                    loss_bias_sum = (sum_bias.abs() * weights).mean()

                    ensemble_var = pred_x0_grouped.var(dim=1, unbiased=True) # [1, C, N, L]
                    true_squared_error = (ensemble_mean.detach() - target_x0)**2 
                    
                    # 使用 Huber Loss 或 L1 Loss 来对齐方差和误差
                    loss_var_align = (torch.abs(torch.sqrt(ensemble_var) - torch.sqrt(true_squared_error)) * weights).mean()
                    
                    loss = cfg.MEAN_MSE_LAMBDA * loss_mean_mse + \
                           cfg.INDIVIDUAL_L1_LAMBDA * loss_individual_l1 - \
                           cfg.REPULSION_LAMBDA * loss_repulsion + \
                           cfg.BIAS_SUM_LAMBDA * loss_bias_sum + \
                           cfg.VAR_LAMBDA * loss_var_align
                    
                    loss = loss / cfg.ACCUMULATION_STEPS

                scaler.scale(loss).backward()
                batch_loss_tracker += loss.item() * cfg.ACCUMULATION_STEPS

            if is_grad_update_step:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                avg_loss_step = batch_loss_tracker
                total_train_loss += avg_loss_step 
                batch_loss_tracker = 0.0 
                if rank == 0: progress_bar.set_postfix(loss=avg_loss_step)
            else:
                pass

        # --- 验证循环 ---
        ddp_model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for tensors in val_dataloader:
                history_c, static_c, future_x0, future_known_c = [d.to(device_id) for d in tensors[:4]]
                with amp.autocast():
                    # future_x0: (B, L, N, C) -> 需要调整为 (B, N, L, C)
                    future_x0 = future_x0.permute(0, 2, 1, 3)
                    # history_c: (B, L, N, C) -> (B, N, L, C)
                    history_c = history_c.permute(0, 2, 1, 3)
                    # future_known_c: (B, L, N, C) -> (B, N, L, C)
                    future_known_c = future_known_c.permute(0, 2, 1, 3)
                    
                    b = future_x0.shape[0]
                    noise = torch.randn_like(future_x0) # (B, N, L, C)
                    
                    k = torch.randint(0, cfg.T, (b,), device=device_id).long()
                    
                    if dist.is_initialized():
                        sqrt_alpha = ddp_model.module.sqrt_alphas_cumprod[k].view(b, 1, 1, 1)
                        sqrt_one_minus = ddp_model.module.sqrt_one_minus_alphas_cumprod[k].view(b, 1, 1, 1)
                    else:
                        sqrt_alpha = ddp_model.sqrt_alphas_cumprod[k].view(b, 1, 1, 1)
                        sqrt_one_minus = ddp_model.sqrt_one_minus_alphas_cumprod[k].view(b, 1, 1, 1)

                    x_k = sqrt_alpha * future_x0 + sqrt_one_minus * noise
                    
                    pred_noise = ddp_model(x_k, k, history_c, static_c, future_known_c, adj_tensor)
                    val_loss = nn.MSELoss()(pred_noise, noise)
                    total_val_loss += val_loss.item()

        # --- Loss 聚合与日志 ---
        num_updates = len(train_dataloader) / cfg.ACCUMULATION_STEPS
        avg_train_loss_local = total_train_loss / max(num_updates, 1)
        
        avg_train_loss_tensor = torch.tensor(avg_train_loss_local).to(device_id)
        if dist.is_initialized():
            dist.all_reduce(avg_train_loss_tensor, op=dist.ReduceOp.SUM)
            avg_train_loss = avg_train_loss_tensor.item() / world_size
        else:
            avg_train_loss = avg_train_loss_local

        avg_val_loss_local = total_val_loss / len(val_dataloader)
        avg_val_loss_tensor = torch.tensor(avg_val_loss_local).to(device_id)
        if dist.is_initialized():
            dist.all_reduce(avg_val_loss_tensor, op=dist.ReduceOp.SUM)
            avg_val_loss = avg_val_loss_tensor.item() / world_size
        else:
            avg_val_loss = avg_val_loss_local

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
                
                state_dict = ddp_model.module.state_dict() if dist.is_initialized() else ddp_model.state_dict()
                torch.save(state_dict, model_save_path_best)
                best_model_path_for_val = model_save_path_best
            elif avg_val_loss < second_best_val_loss:
                second_best_val_loss = avg_val_loss
                state_dict = ddp_model.module.state_dict() if dist.is_initialized() else ddp_model.state_dict()
                torch.save(state_dict, model_save_path_second_best)
                second_best_model_path_for_val = model_save_path_second_best

        # --- 周期性 MAE 评估 ---
        run_mae_eval = cfg.EVAL_ON_VAL and (epoch + 1) % cfg.EVAL_ON_VAL_EPOCH == 0
        if run_mae_eval:
            current_val_mae = float('inf') 
            if rank == 0: print(f"Epoch {epoch+1}, running periodic MAE evaluation...")
            
            model_to_eval = ddp_model.module if dist.is_initialized() else ddp_model
            # Note: periodic_evaluate_mae already handles permute (L, N) -> (N, L) internally if it assumes (B, L, N, C) from Loader.
            # But let's check evaluation.py again.
            # It calls ddim_sample with .permute(0, 2, 1, 3).
            # Our Loader gives (B, L, N, C).
            # So .permute(0, 2, 1, 3) -> (B, N, L, C).
            # This is CORRECT for our new model.
            current_val_seq_local = periodic_evaluate_mae(
                model_to_eval, 
                val_eval_loader,
                train_y_scaler,
                adj_tensor,
                cfg,
                device_id
            )
        
            if dist.is_initialized():
                gathered_results = [None] * world_size
                dist.gather_object(
                    current_val_seq_local,
                    gathered_results if rank == 0 else None,
                    dst=0
                )
            else:
                gathered_results = [current_val_seq_local]

            if rank == 0:
                current_val_seq = np.concatenate(gathered_results, axis=0)
                current_val_seq = current_val_seq[:len(y_true_original)]
                current_val_mae = np.mean(np.abs(current_val_seq - y_true_original))
                epoch_log_data['avg_val_mae'] = current_val_mae
                print(f"Avg Val MAE: {current_val_mae:.4f}")

                state_dict = ddp_model.module.state_dict() if dist.is_initialized() else ddp_model.state_dict()
                
                if current_val_mae < best_val_mae:
                    second_best_val_mae = best_val_mae
                    best_val_mae = current_val_mae
                    if best_model_path_for_eval is not None and os.path.exists(best_model_path_for_eval):
                        if os.path.exists(model_save_path_mae_second_best):
                            os.remove(model_save_path_mae_second_best)
                        os.rename(best_model_path_for_eval, model_save_path_mae_second_best)
                        second_best_model_path_for_val = model_save_path_mae_second_best
                    torch.save(state_dict, model_save_path_mae_best)
                    best_model_path_for_eval = model_save_path_mae_best
                elif current_val_mae < second_best_val_mae:
                    second_best_val_mae = current_val_mae
                    torch.save(state_dict, model_save_path_mae_second_best)
                    second_best_model_path_for_val = model_save_path_mae_second_best

        if rank == 0:
            csv_logger.log_epoch(epoch_log_data)

        if dist.is_initialized():
            dist.barrier()
        scheduler.step()

    if dist.is_initialized():
        dist.barrier()
    
    if rank == 0:
        print("\n" + "="*50 + "\nTraining finished. Starting evaluation on the test set...\n" + "="*50 + "\n")
    
    path_list_to_eval = [best_model_path_for_eval, second_best_model_path_for_eval, best_model_path_for_val, second_best_model_path_for_val] if rank == 0 else [None, None, None, None]
    
    if dist.is_initialized():
        dist.broadcast_object_list(path_list_to_eval, src=0)
    
    best_model_path_synced, second_best_model_path_synced, best_model_path_for_val_synced, second_best_model_path_for_val_synced = path_list_to_eval

    def run_eval(path, key_name, desc):
        if path and os.path.exists(path):
            if rank == 0: print(f"\n[ALL GPUS] Evaluating {desc}: {os.path.basename(path)}")
            metrics = evaluate_model(cfg, path, scaler_y_save_path, scaler_mm_save_path, scaler_z_save_path, device=f"cuda:{device_id}", rank=rank, world_size=world_size, key=key_name)
            if rank == 0: print_metrics(metrics)

    run_eval(best_model_path_for_val_synced, 'best_val', 'BEST VAL model')
    run_eval(second_best_model_path_for_val_synced, 'second_best_val', '2ND BEST VAL model')
    run_eval(best_model_path_synced, 'best', 'BEST MAE model')
    run_eval(second_best_model_path_synced, 'second_best', '2ND BEST MAE model')

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    train()