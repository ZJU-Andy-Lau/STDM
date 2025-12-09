import warnings
warnings.filterwarnings("ignore")
import sys
import os
import copy
import math

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(current_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

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
    from train_lite.utils import CsvLogger, NanDebugger
    from train_lite.dataset import EVChargerDatasetV2, create_sliding_windows
    from train_lite.metrics import print_metrics
    from train_lite.evaluation import periodic_evaluate_mae, evaluate_model
except ImportError:
    from config import ConfigV2
    from utils import CsvLogger, NanDebugger
    from dataset import EVChargerDatasetV2, create_sliding_windows
    from metrics import print_metrics
    from evaluation import periodic_evaluate_mae, evaluate_model

from model_lite import SpatioTemporalDiffusionModelV2
from scheduler import MultiStageOneCycleLR

# --- [DEBUG TOOL 1] 层级 NaN 监控器 ---
class LayerNanMonitor:
    def __init__(self, model):
        self.hooks = []
        # 递归注册 Hook 到每一个计算子模块
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv1d, nn.Linear, nn.MultiheadAttention, nn.LayerNorm, nn.GroupNorm, nn.Embedding)):
                self.hooks.append(
                    module.register_forward_hook(self._get_hook(name))
                )
    
    def _get_hook(self, name):
        def hook(module, input, output):
            # 1. 检查权重 (Weights)
            if hasattr(module, 'weight') and module.weight is not None:
                if torch.isnan(module.weight).any():
                    print(f"\n[DEBUG-FATAL] Layer '{name}' WEIGHT contains NaN!")
                    raise RuntimeError(f"NaN weights detected at layer: {name}")
                if torch.isinf(module.weight).any():
                    print(f"\n[DEBUG-FATAL] Layer '{name}' WEIGHT contains Inf!")
                    raise RuntimeError(f"Inf weights detected at layer: {name}")

            # 2. 检查输入 (Input)
            # input 是一个 tuple
            for i, inp in enumerate(input):
                if inp is not None and isinstance(inp, torch.Tensor):
                    if torch.isnan(inp).any():
                        print(f"\n[DEBUG-FATAL] Layer '{name}' INPUT[{i}] contains NaN!")
                        raise RuntimeError(f"NaN input detected at layer: {name}")
            
            # 3. 检查输出 (Output)
            if isinstance(output, tuple):
                out_tensor = output[0]
            else:
                out_tensor = output
            
            if torch.isnan(out_tensor).any():
                print(f"\n[DEBUG-FATAL] Layer '{name}' OUTPUT produced NaN!")
                # 打印更多调试信息
                if hasattr(module, 'weight'):
                    print(f" - Weight stats: min={module.weight.min()}, max={module.weight.max()}")
                print(f" - Input stats: min={input[0].min() if len(input)>0 else 'N/A'}, max={input[0].max() if len(input)>0 else 'N/A'}")
                raise RuntimeError(f"NaN output detected at layer: {name}")
            
            if torch.isinf(out_tensor).any():
                print(f"\n[DEBUG-FATAL] Layer '{name}' OUTPUT produced Inf!")
                raise RuntimeError(f"Inf output detected at layer: {name}")
        return hook

    def remove(self):
        for h in self.hooks:
            h.remove()

def train():
    # --- DDP 初始化 ---
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        device_id = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(device_id)
    else:
        print("Warning: Not running in DDP mode. Fallback to single GPU.")
        rank = 0
        device_id = 0
        world_size = 1
        torch.cuda.set_device(0)

    cfg = ConfigV2()

    if rank == 0:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg.RUN_ID = run_id
        print(f"Starting Training (Debug Mode). Run ID: {cfg.RUN_ID}")
        print(f"Batch Size: {cfg.BATCH_SIZE}, Ensemble K: {cfg.ENSEMBLE_K}")
        print(f"Strategies: Energy Loss + Resilient Rollback + Float32 Loss + Gradient Supervision")
        
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

    if dist.is_initialized():
        run_id_list = [cfg.RUN_ID]
        dist.broadcast_object_list(run_id_list, src=0)
        if rank != 0: 
            cfg.RUN_ID = run_id_list[0]
            scaler_y_save_path = cfg.SCALER_Y_SAVE_PATH_TEMPLATE.format(run_id=cfg.RUN_ID)
            scaler_mm_save_path = cfg.SCALER_MM_SAVE_PATH_TEMPLATE.format(run_id=cfg.RUN_ID)
            scaler_z_save_path = cfg.SCALER_Z_SAVE_PATH_TEMPLATE.format(run_id=cfg.RUN_ID)

    log_headers = ['epoch', 'avg_train_loss', 'avg_val_loss', 'lr', 'avg_val_mae']
    csv_logger = CsvLogger(log_dir='./results', run_id=cfg.RUN_ID, rank=rank, headers=log_headers)

    adj_matrix = np.load(cfg.ADJ_MATRIX_PATH)
    distances = adj_matrix[adj_matrix > 0]
    sigma = np.std(distances)
    adj_matrix = np.exp(-np.square(adj_matrix) / (sigma**2))
    adj_matrix[adj_matrix < 0.1] = 0
    adj_tensor = torch.from_numpy(adj_matrix).float().to(device_id)

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

    # --- [DEBUG] 激活层级 NaN 监控器 ---
    # 这会在每次 Forward 时检查，性能略有损耗但对调试是必须的
    if rank == 0:
        print("Initializing LayerNanMonitor...")
    nan_monitor = LayerNanMonitor(ddp_model)

    optimizer = optim.AdamW(ddp_model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=1e-4)
    scaler = amp.GradScaler()

    scheduler = MultiStageOneCycleLR(optimizer=optimizer,
                                     total_steps=cfg.EPOCHS,
                                     warmup_ratio=cfg.WARMUP_EPOCHS / cfg.EPOCHS,
                                     cooldown_ratio=cfg.COOLDOWN_EPOCHS / cfg.EPOCHS)

    original_val_samples = create_sliding_windows(val_features, cfg.HISTORY_LEN, cfg.PRED_LEN)
    y_true_original = np.array([s[1][:, :, :cfg.TARGET_FEAT_DIM] for s in original_val_samples]).squeeze(-1)[:cfg.EVAL_ON_VAL_BATCHES * cfg.BATCH_SIZE]

    max_retries = 10
    torch.autograd.set_detect_anomaly(True) # 开启 Anomaly Detection

    for epoch in range(cfg.EPOCHS):
        if rank == 0:
            print(f"Creating snapshot for Epoch {epoch+1}...")
        
        snapshot = {
            'model': copy.deepcopy(ddp_model.state_dict()),
            'optimizer': copy.deepcopy(optimizer.state_dict()),
            'scheduler': copy.deepcopy(scheduler.state_dict()),
            'scaler': copy.deepcopy(scaler.state_dict())
        }

        retry_count = 0
        epoch_success = False
        
        while not epoch_success:
            if retry_count >= max_retries:
                if rank == 0:
                    print(f"[CRITICAL] Max retries ({max_retries}) reached. Stopping.")
                return 

            if retry_count > 0:
                if rank == 0:
                    print(f"--- [Retry] Restarting Epoch {epoch+1} (Attempt {retry_count + 1}) ---")
            
            if dist.is_initialized():
                train_sampler.set_epoch(epoch + retry_count * 100)
                
            ddp_model.train()
            optimizer.zero_grad()
            
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1} [Train]", disable=(rank != 0))
            total_train_loss = 0.0
            batch_loss_tracker = 0.0
            nan_detected_in_epoch = False 

            for i, (history_c, static_c, future_x0, future_known_c, idx) in enumerate(progress_bar):
                is_grad_update_step = ((i + 1) % cfg.ACCUMULATION_STEPS == 0) or ((i + 1) == len(train_dataloader))
                
                # 在 Forward 之前检查一次参数，确认是否已经被上一步污染
                for name, param in ddp_model.named_parameters():
                    if torch.isnan(param).any():
                        print(f"[FATAL-LOOP-START] Parameter {name} contains NaN at start of batch {i}! Rollback failed or previous step leaked.")
                        nan_detected_in_epoch = True
                        break
                if nan_detected_in_epoch: break

                context = nullcontext() if is_grad_update_step else (ddp_model.no_sync() if dist.is_initialized() else nullcontext())
                
                try:
                    with context:
                        with amp.autocast():
                            hist_batch = history_c.to(device_id)
                            stat_batch = static_c.to(device_id)
                            future_batch = future_x0.to(device_id)
                            known_batch = future_known_c.to(device_id)
                            
                            B = hist_batch.shape[0]
                            K = cfg.ENSEMBLE_K

                            hist_exp = hist_batch.repeat_interleave(K, dim=0).permute(0, 2, 1, 3) 
                            stat_exp = stat_batch.repeat_interleave(K, dim=0)
                            future_exp = future_batch.repeat_interleave(K, dim=0).permute(0, 2, 1, 3)
                            known_exp = known_batch.repeat_interleave(K, dim=0).permute(0, 2, 1, 3)
                            
                            noise = torch.randn_like(future_exp)
                            t_per_sample = torch.randint(0, cfg.T, (B,), device=device_id).long()
                            t_exp = t_per_sample.repeat_interleave(K) 

                            if dist.is_initialized():
                                sqrt_alpha_bar = ddp_model.module.sqrt_alphas_cumprod[t_exp].view(B*K, 1, 1, 1)
                                sqrt_one_minus_alpha_bar = ddp_model.module.sqrt_one_minus_alphas_cumprod[t_exp].view(B*K, 1, 1, 1)
                            else:
                                sqrt_alpha_bar = ddp_model.sqrt_alphas_cumprod[t_exp].view(B*K, 1, 1, 1)
                                sqrt_one_minus_alpha_bar = ddp_model.sqrt_one_minus_alphas_cumprod[t_exp].view(B*K, 1, 1, 1)
                            
                            x_t = sqrt_alpha_bar * future_exp + sqrt_one_minus_alpha_bar * noise
                            x_t.requires_grad_(True)

                            # Forward (NanMonitor 会在这里工作)
                            predicted_noise = ddp_model(
                                x_t, t_exp, hist_exp, stat_exp, known_exp, adj_tensor
                            )

                        # Loss 计算 (Float32)
                        with amp.autocast(enabled=False):
                            predicted_noise_fp32 = predicted_noise.float()
                            x_t_fp32 = x_t.float()
                            
                            if dist.is_initialized():
                                sqrt_alpha_bar_fp32 = ddp_model.module.sqrt_alphas_cumprod[t_exp].view(B*K, 1, 1, 1).float()
                                sqrt_one_minus_alpha_bar_fp32 = ddp_model.module.sqrt_one_minus_alphas_cumprod[t_exp].view(B*K, 1, 1, 1).float()
                                alphas_cumprod_fp32 = ddp_model.module.alphas_cumprod.float()
                            else:
                                sqrt_alpha_bar_fp32 = ddp_model.sqrt_alphas_cumprod[t_exp].view(B*K, 1, 1, 1).float()
                                sqrt_one_minus_alpha_bar_fp32 = ddp_model.sqrt_one_minus_alphas_cumprod[t_exp].view(B*K, 1, 1, 1).float()
                                alphas_cumprod_fp32 = ddp_model.alphas_cumprod.float()

                            pred_x0_exp = (x_t_fp32 - sqrt_one_minus_alpha_bar_fp32 * predicted_noise_fp32) / (sqrt_alpha_bar_fp32 + 1e-8)
                            pred_x0_grouped = pred_x0_exp.view(B, K, cfg.NUM_NODES, cfg.PRED_LEN, cfg.TARGET_FEAT_DIM)
                            target_x0 = future_batch.permute(0, 2, 1, 3).float()
                            target_x0_expanded = target_x0.unsqueeze(1) 
                            
                            weights = alphas_cumprod_fp32[t_exp].view(B, K, 1, 1, 1)
                            weights = weights[:, 0:1, :, :, :] 

                            ensemble_mean = pred_x0_grouped.mean(dim=1, keepdim=True)
                            ensemble_var = pred_x0_grouped.var(dim=1, unbiased=True, keepdim=True)
                            ensemble_var = torch.clamp(ensemble_var, min=1e-4)

                            loss_mean_mse = ((ensemble_mean - target_x0_expanded)**2 * weights).mean()

                            # --- Energy Score Calculation with Probe ---
                            eps_safe = 1e-8
                            diff = pred_x0_grouped - target_x0_expanded
                            sum_sq = diff.pow(2).sum(dim=-1) 
                            es_accuracy = torch.sqrt(sum_sq + eps_safe).mean(dim=1)
                            
                            diff_div = pred_x0_grouped.unsqueeze(2) - pred_x0_grouped.unsqueeze(1)
                            sum_sq_div = diff_div.pow(2).sum(dim=-1)
                            
                            # --- [DEBUG TOOL 2] 多样性数值探针 ---
                            min_div_val = sum_sq_div.min().item()
                            if min_div_val < 1e-9:
                                print(f"\n[DEBUG-ENERGY] Batch {i}: Diversity term (sum_sq) is extremely small: {min_div_val:.4e}")
                                print("This will likely cause Infinite Gradients in backward pass!")

                            es_diversity = torch.sqrt(sum_sq_div + eps_safe).mean(dim=(1, 2))
                            
                            es_combined = es_accuracy - 0.5 * es_diversity
                            loss_energy = (es_combined.unsqueeze(1).unsqueeze(-1) * weights).mean()

                            sq_error = (target_x0_expanded - ensemble_mean)**2
                            nll_term1 = 0.5 * torch.log(ensemble_var)
                            nll_term2 = 0.5 * sq_error / (ensemble_var + 1e-8)
                            nll_per_element = nll_term1 + nll_term2
                            nll_per_element_clamped = torch.clamp(nll_per_element, max=100.0) 
                            loss_nll = (nll_per_element_clamped * weights).mean()

                            loss = cfg.MEAN_MSE_LAMBDA * loss_mean_mse + \
                                   cfg.ENERGY_LAMBDA * loss_energy + \
                                   cfg.NLL_LAMBDA * loss_nll
                            
                            loss = loss / cfg.ACCUMULATION_STEPS

                            if torch.isnan(loss) or torch.isinf(loss):
                                print(f"[Rank {rank}] Loss became NaN at Batch {i} (FP32 calculation).")
                                nan_detected_in_epoch = True

                        if not nan_detected_in_epoch:
                            scaler.scale(loss).backward()
                            batch_loss_tracker += loss.item() * cfg.ACCUMULATION_STEPS
                            
                            # --- [DEBUG TOOL 3] 梯度监理 (Gradient Supervisor) ---
                            # 在 unscale 之前或之后检查皆可，这里我们为了安全，先检查是否有 Nan
                            # 注意：Scaler 可能会在 step 时处理 NaN，但我们要看到底是谁产生的
                            if is_grad_update_step:
                                scaler.unscale_(optimizer) # 先 unscale 以查看真实梯度值
                                has_nan_grad = False
                                for name, param in ddp_model.named_parameters():
                                    if param.grad is not None:
                                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                            print(f"\n[DEBUG-GRAD] Parameter '{name}' has NaN/Inf gradient!")
                                            has_nan_grad = True
                                        
                                        # 可选：打印梯度 Norm 过大的情况
                                        grad_norm = param.grad.norm().item()
                                        if grad_norm > 100.0:
                                            print(f"[DEBUG-GRAD] Parameter '{name}' has large grad norm: {grad_norm:.4f}")
                                
                                if has_nan_grad:
                                    print(f"[DEBUG-GRAD] Batch {i}: Gradient explosion detected. Stopping batch.")
                                    nan_detected_in_epoch = True
                                    # 不要 step，直接触发回滚
                                    optimizer.zero_grad() 

                    if is_grad_update_step and not nan_detected_in_epoch:
                        torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        avg_loss_step = batch_loss_tracker
                        total_train_loss += avg_loss_step 
                        batch_loss_tracker = 0.0 
                        if rank == 0: progress_bar.set_postfix(loss=avg_loss_step)
                
                except RuntimeError as e:
                    if "NaN" in str(e) or "Inf" in str(e):
                        print(f"[EXCEPTION CAUGHT] {e}")
                        nan_detected_in_epoch = True
                    else:
                        raise e

                # DDP Sync Logic for Rollback
                is_nan_local = torch.tensor(1.0 if nan_detected_in_epoch else 0.0, device=device_id)
                if dist.is_initialized():
                    dist.all_reduce(is_nan_local, op=dist.ReduceOp.MAX)
                
                if is_nan_local.item() > 0.5:
                    nan_detected_in_epoch = True
                    break 

            if nan_detected_in_epoch:
                retry_count += 1
                if rank == 0: print("Restoring state...")
                ddp_model.load_state_dict(snapshot['model'])
                optimizer.load_state_dict(snapshot['optimizer'])
                scheduler.load_state_dict(snapshot['scheduler'])
                scaler.load_state_dict(snapshot['scaler'])
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                continue
            else:
                epoch_success = True

        # Validation Logic (Unchanged)
        ddp_model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for tensors in val_dataloader:
                history_c, static_c, future_x0, future_known_c = [d.to(device_id) for d in tensors[:4]]
                with amp.autocast():
                    future_x0 = future_x0.permute(0, 2, 1, 3)
                    history_c = history_c.permute(0, 2, 1, 3)
                    future_known_c = future_known_c.permute(0, 2, 1, 3)
                    b = future_x0.shape[0]
                    noise = torch.randn_like(future_x0) 
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

        run_mae_eval = cfg.EVAL_ON_VAL and (epoch + 1) % cfg.EVAL_ON_VAL_EPOCH == 0
        if run_mae_eval:
            if rank == 0: print(f"Epoch {epoch+1}, running periodic MAE evaluation...")
            model_to_eval = ddp_model.module if dist.is_initialized() else ddp_model
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
        dist.destroy_process_group()

if __name__ == "__main__":
    train()