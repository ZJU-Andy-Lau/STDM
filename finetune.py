import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
from tqdm import tqdm
import json
import os
from datetime import datetime
from torch.cuda import amp
from scipy.stats import t
import pandas as pd
import properscoring as ps
from contextlib import nullcontext
import math
import random
import csv
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib

# 导入分布式训练所需的库    
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import LambdaLR, SequentialLR, CosineAnnealingWarmRestarts

# 依赖文件 (需在同一目录下)
from model_sigma import SpatioTemporalDiffusionModelV2
from scheduler import MultiStageOneCycleLR

# ==========================================
# 用户配置区域
# ==========================================
# 【重要】请在这里填入你要续训的 Run ID
RESUME_RUN_ID = "20251123_220353"  # <--- 修改这里
# ==========================================

class CsvLogger:
    def __init__(self, log_dir, run_id, rank, headers):
        self.log_dir = log_dir
        self.run_id = run_id
        self.rank = rank
        self.headers = headers
        self.log_file_path = None
        if self.rank == 0:
            try:
                os.makedirs(self.log_dir, exist_ok=True)
                self.log_file_path = os.path.join(self.log_dir, f"{self.run_id}.csv")
                # 如果文件不存在，写入表头；如果是续训，通常我们会追加，但为了简单起见，这里作为新文件
                mode = 'w' 
                with open(self.log_file_path, mode, newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(self.headers)
            except Exception as e:
                print(f"[Rank 0 Logger Error] Failed to initialize logger: {e}")
                self.log_file_path = None 

    def log_epoch(self, epoch_data):
        if self.rank != 0 or self.log_file_path is None: return
        try:
            row = [epoch_data.get(h, '') for h in self.headers]
            with open(self.log_file_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(row)
        except Exception: pass

class ConfigV2:
    NORMALIZATION_TYPE = "minmax" 
    RUN_ID = None 

    # 数据参数
    NUM_NODES = 275
    HISTORY_LEN = 12
    PRED_LEN = 12
    
    # 特征维度
    TARGET_FEAT_DIM = 1
    DYNAMIC_FEAT_DIM = 12
    STATIC_FEAT_DIM = 7
    FUTURE_KNOWN_FEAT_DIM = 13
    HISTORY_FEATURES = TARGET_FEAT_DIM + DYNAMIC_FEAT_DIM
    STATIC_FEATURES = STATIC_FEAT_DIM
    
    # 模型参数
    MODEL_DIM = 64
    NUM_HEADS = 4
    DEPTH = 4
    T = 1000
    
    # 续训参数
    EPOCHS = 2             # <--- 仅训练 2 个 Epoch
    BATCH_SIZE = 4 
    LEARNING_RATE = 5e-5   # <--- 续训时可以使用较小的学习率
    ACCUMULATION_STEPS = 4

    WARMUP_EPOCHS = 0      # 续训通常不需要长的 warmup
    COOLDOWN_EPOCHS = 0    
    
    # 路径模板 (读取旧的Scaler，保存新的模型)
    MODEL_SAVE_PATH_TEMPLATE = "./weights/st_diffusion_model_v2_{run_id}_{rank}.pth"
    # 注意：Scaler 路径使用 RESUME_RUN_ID 读取
    SCALER_Y_READ_PATH = f"./weights/scaler_y_v2_{RESUME_RUN_ID}.pkl"
    SCALER_MM_READ_PATH = f"./weights/scaler_mm_v2_{RESUME_RUN_ID}.pkl"
    SCALER_Z_READ_PATH = f"./weights/scaler_z_v2_{RESUME_RUN_ID}.pkl"

    # 数据文件路径
    TRAIN_FEATURES_PATH = './urbanev/features_train_wea_poi.npy'
    VAL_FEATURES_PATH = './urbanev/features_valid_wea_poi.npy'
    TEST_FEATURES_PATH = './urbanev/features_test_wea_poi.npy'
    ADJ_MATRIX_PATH = './urbanev/dis.npy'
    
    # 评估配置
    EVAL_SEED = 42
    SAMPLING_ETA = 0.0

# --- 辅助函数 ---
def get_edge_index(adj_matrix):
    edge_indices = []
    edge_weights = []
    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            if adj_matrix[i, j] != 0:
                edge_indices.append((i, j))
                edge_weights.append(adj_matrix[i, j])
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_weights = torch.tensor(edge_weights, dtype=torch.float)
    return edge_index, edge_weights

def create_sliding_windows(data, history_len, pred_len):
    samples = []
    total_len = len(data)
    for i in range(total_len - history_len - pred_len + 1):
        history = data[i : i + history_len]
        future = data[i + history_len : i + history_len + pred_len]
        samples.append((history, future))
    return samples

def calc_layer_lengths(L_in, depth, kernel_size=3, stride=2, padding=1, dilation=1):
    lengths = [L_in]
    for i in range(depth):
        L_prev = lengths[-1]
        L_out = math.floor((L_prev + 2*padding - dilation*(kernel_size - 1) - 1) / stride + 1)
        lengths.append(L_out)
    return lengths

def batch_time_edge_index(edge_index, edge_weights, num_nodes, batch_size, time_steps, device):
    if time_steps <= 0: return torch.empty(2, 0, dtype=torch.long, device=device)
    num_total_graphs = batch_size * time_steps
    edge_index_list = [edge_index + i * num_nodes for i in range(num_total_graphs)]
    batched_edge_index = torch.cat(edge_index_list, dim=1).to(device)
    batched_edge_weights = edge_weights.repeat(num_total_graphs).to(device)
    return batched_edge_index, batched_edge_weights

class EVChargerDatasetV2(Dataset):
    def __init__(self, features, history_len, pred_len, cfg, scaler_y=None,scaler_mm=None,scaler_z=None):
        self.cfg = cfg
        minmax_features = features[:, :, cfg.HISTORY_FEATURES-4:cfg.HISTORY_FEATURES+5].copy()
        zscore_features = features[:, :, cfg.HISTORY_FEATURES+5:].copy()
        dynamic_features = features[:, :, :cfg.HISTORY_FEATURES].copy()
        static_features = features[0, :, cfg.HISTORY_FEATURES:].copy()
        target_col_original = dynamic_features[:, :, 0]

        self.scaler_y = scaler_y
        self.scaler_mm = scaler_mm
        self.scaler_z = scaler_z

        # 必须确保传入了 Scaler，因为这是续训
        if self.scaler_y is None or self.scaler_mm is None or self.scaler_z is None:
            raise ValueError("For resuming training, scalers must be provided (loaded from disk).")

        if self.cfg.NORMALIZATION_TYPE != "none":
            target_col_reshaped = target_col_original.reshape(-1, 1)
            normalized_target = self.scaler_y.transform(target_col_reshaped)
            dynamic_features[:, :, 0] = normalized_target.reshape(target_col_original.shape)

        mm_norm = self.scaler_mm.transform(minmax_features.reshape(-1, minmax_features.shape[-1])).reshape(minmax_features.shape)
        z_norm = self.scaler_z.transform(zscore_features.reshape(-1, zscore_features.shape[-1])).reshape(zscore_features.shape)

        dynamic_features[:, :, cfg.HISTORY_FEATURES-4:] = mm_norm[:, :, :4]
        static_features[:, :5] = mm_norm[0, :, -5:]
        static_features[:, 5:] = z_norm[0, :, :]
        self.static_features = torch.tensor(static_features, dtype=torch.float)
        self.samples = create_sliding_windows(dynamic_features, history_len, pred_len)

    def get_scaler(self):
        return self.scaler_y, self.scaler_mm, self.scaler_z
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        history, future = self.samples[idx]
        history_c = torch.tensor(history, dtype=torch.float)
        future_x0 = torch.tensor(future[:, :, :self.cfg.TARGET_FEAT_DIM], dtype=torch.float)
        known_start_idx = self.cfg.TARGET_FEAT_DIM + (self.cfg.DYNAMIC_FEAT_DIM - self.cfg.FUTURE_KNOWN_FEAT_DIM)
        future_known_c = torch.tensor(future[:, :, known_start_idx : self.cfg.HISTORY_FEATURES], dtype=torch.float)
        return history_c, self.static_features, future_x0, future_known_c, idx

# --- 评估相关代码 ---
class EvalConfig(ConfigV2):
    BATCH_SIZE = 8
    NUM_SAMPLES = 10
    SAMPLING_STEPS = 10
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def calculate_metrics(y_true, y_pred_median, y_pred_samples, device):
    metrics = {}
    metrics['mae'] = float(np.mean(np.abs(y_pred_median - y_true)))
    metrics['rmse'] = float(np.sqrt(np.mean((y_pred_median - y_true)**2)))
    metrics['crps'] = float(ps.crps_ensemble(y_true, y_pred_samples.transpose(1, 0, 2, 3), axis=0).mean())
    return metrics

def print_metrics(metrics):
    print("\n--- Final Evaluation Metrics ---")
    print(f"MAE:  {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"CRPS: {metrics['crps']:.4f}")
    print("--------------------------------\n")

def evaluate_model(train_cfg, model_path, scaler_y, scaler_mm, scaler_z, device, rank, world_size):
    # 这里的参数略有不同，直接传入对象而不是路径，因为内存中已经加载了
    cfg = EvalConfig()
    cfg.RUN_ID = train_cfg.RUN_ID
    cfg.DEVICE = device

    model = SpatioTemporalDiffusionModelV2(
        in_features=cfg.TARGET_FEAT_DIM, out_features=cfg.TARGET_FEAT_DIM,
        history_features=cfg.HISTORY_FEATURES, static_features=cfg.STATIC_FEATURES,
        future_known_features=cfg.FUTURE_KNOWN_FEAT_DIM, model_dim=cfg.MODEL_DIM,
        num_heads=cfg.NUM_HEADS, T=cfg.T, depth=cfg.DEPTH
    ).to(cfg.DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=cfg.DEVICE))
    model.eval()
    
    adj_matrix = np.load(cfg.ADJ_MATRIX_PATH)
    distances = adj_matrix[adj_matrix > 0]
    sigma = np.std(distances)
    adj_matrix = np.exp(-np.square(adj_matrix) / (sigma**2))
    adj_matrix[adj_matrix < 0.1] = 0
    original_edge_index, original_edge_weights = get_edge_index(adj_matrix)

    test_features = np.load(cfg.TEST_FEATURES_PATH)
    test_dataset = EVChargerDatasetV2(test_features, cfg.HISTORY_LEN, cfg.PRED_LEN, cfg, scaler_y=scaler_y, scaler_mm=scaler_mm, scaler_z=scaler_z)
    
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, sampler=test_sampler)

    all_predictions_list, all_samples_list, all_true_list = [], [], []
    disable_tqdm = (dist.is_initialized() and dist.get_rank() != 0)

    with torch.no_grad(), amp.autocast():
        for tensors in tqdm(test_dataloader, desc="Evaluating", disable=disable_tqdm):
            history_c, static_c, future_x0_true, future_known_c, idx = tensors
            history_c = history_c.to(cfg.DEVICE)
            static_c = static_c.to(cfg.DEVICE)
            future_x0_true = future_x0_true.to(cfg.DEVICE)
            future_known_c = future_known_c.to(cfg.DEVICE)

            all_true_list.append(future_x0_true.permute(0, 2, 1, 3).cpu().numpy())

            b = history_c.shape[0]
            len_list = calc_layer_lengths(cfg.PRED_LEN, cfg.DEPTH)
            edge_data = [batch_time_edge_index(original_edge_index.to(device), original_edge_weights.to(device), cfg.NUM_NODES, b, len_list[d], cfg.DEVICE) for d in range(cfg.DEPTH + 1)]

            generated_samples = []
            for _ in range(cfg.NUM_SAMPLES):
                sample = model.ddim_sample(
                    history_c=history_c.permute(0, 2, 1, 3), static_c=static_c,
                    future_known_c=future_known_c.permute(0, 2, 1, 3),
                    history_edge_data=edge_data,
                    future_edge_data=edge_data,
                    shape=future_x0_true.permute(0, 2, 1, 3).shape, sampling_steps=cfg.SAMPLING_STEPS, eta=cfg.SAMPLING_ETA
                )
                generated_samples.append(sample)
            
            stacked_samples = torch.stack(generated_samples, dim=0)
            median_prediction = torch.median(stacked_samples, dim=0).values
            all_predictions_list.append(median_prediction.cpu().numpy())
            all_samples_list.append(stacked_samples.cpu().numpy())

    local_predictions = np.concatenate(all_predictions_list, axis=0) if all_predictions_list else np.empty((0))
    local_samples = np.concatenate(all_samples_list, axis=1) if all_samples_list else np.empty((0))
    local_true = np.concatenate(all_true_list, axis=0) if all_true_list else np.empty((0))

    gathered_preds = [None] * world_size
    gathered_samples = [None] * world_size
    gathered_true = [None] * world_size

    dist.gather_object(local_predictions, gathered_preds if rank == 0 else None, dst=0)
    dist.gather_object(local_samples, gathered_samples if rank == 0 else None, dst=0)
    dist.gather_object(local_true, gathered_true if rank == 0 else None, dst=0)
    
    if rank == 0:
        all_predictions_norm = np.concatenate(gathered_preds, axis=0)
        all_samples_norm = np.concatenate(gathered_samples, axis=1)
        all_true_norm = np.concatenate(gathered_true, axis=0)

        # 反归一化
        all_predictions = scaler_y.inverse_transform(all_predictions_norm.reshape(-1, 1)).reshape(all_predictions_norm.shape)
        y_true_original = scaler_y.inverse_transform(all_true_norm.reshape(-1, 1)).reshape(all_true_norm.shape)
        all_samples = scaler_y.inverse_transform(all_samples_norm.reshape(-1, 1)).reshape(all_samples_norm.shape)

        all_predictions = np.clip(all_predictions, 0.0, 1.0)
        all_samples = np.clip(all_samples, 0.0, 1.0)

        # 调整维度
        all_predictions = all_predictions.squeeze(-1).transpose(0, 2, 1)
        y_true_original = y_true_original.squeeze(-1).transpose(0, 2, 1)
        all_samples = all_samples.squeeze(-1).transpose(1, 0, 3, 2)

        np.save(f'./results/pred_{RESUME_RUN_ID}_finetune.npy', all_predictions)
        np.save(f'./results/samples_{RESUME_RUN_ID}_finetune.npy', all_samples)

        return calculate_metrics(y_true_original, all_predictions, all_samples, cfg.DEVICE)
    return None


def resume_training():
    dist.init_process_group("nccl")
    rank = int(os.environ["RANK"])
    device_id = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(device_id)

    cfg = ConfigV2()
    # 为本次续训生成一个新的 ID，避免覆盖旧文件
    if rank == 0:
        cfg.RUN_ID = f"{RESUME_RUN_ID}_finetune"
        print(f"[Info] Resuming from {RESUME_RUN_ID}. New Run ID: {cfg.RUN_ID}")
        print(f"[Info] Training for {cfg.EPOCHS} epochs WITHOUT validation.")
    else:
        cfg.RUN_ID = "DUMMY" # 占位

    # 广播 Run ID
    run_id_list = [cfg.RUN_ID]
    dist.broadcast_object_list(run_id_list, src=0)
    cfg.RUN_ID = run_id_list[0]

    # 日志
    log_headers = ['epoch', 'avg_train_loss', 'lr']
    csv_logger = CsvLogger('./results', cfg.RUN_ID, rank, log_headers)

    # 加载邻接矩阵
    adj_matrix = np.load(cfg.ADJ_MATRIX_PATH)
    distances = adj_matrix[adj_matrix > 0]
    sigma = np.std(distances)
    adj_matrix = np.exp(-np.square(adj_matrix) / (sigma**2))
    adj_matrix[adj_matrix < 0.1] = 0
    original_edge_index, original_edge_weights = get_edge_index(adj_matrix)

    # 1. 加载之前的 Scaler (必须存在)
    if rank == 0:
        print(f"[Info] Loading scalers from run {RESUME_RUN_ID}...")
        if not os.path.exists(cfg.SCALER_Y_READ_PATH):
            raise FileNotFoundError(f"Scaler Y not found at {cfg.SCALER_Y_READ_PATH}")
        scaler_y = joblib.load(cfg.SCALER_Y_READ_PATH)
        scaler_mm = joblib.load(cfg.SCALER_MM_READ_PATH)
        scaler_z = joblib.load(cfg.SCALER_Z_READ_PATH)
    else:
        scaler_y, scaler_mm, scaler_z = None, None, None
    
    # 广播 Scaler 对象到所有进程 (简单起见，或者每个进程都load)
    # joblib 加载通常很快，这里让每个进程尝试加载（注意：非 rank0 可能需要完整路径访问权限）
    # 更安全的做法是只在 rank0 加载然后 broadcast，但对于 sklearn scaler 对象，
    # 只要所有节点都能访问文件系统，直接加载最简单。
    try:
        scaler_y = joblib.load(cfg.SCALER_Y_READ_PATH)
        scaler_mm = joblib.load(cfg.SCALER_MM_READ_PATH)
        scaler_z = joblib.load(cfg.SCALER_Z_READ_PATH)
    except:
        pass # 如果非rank0加载失败，可能需要其他处理，这里假设共享文件系统

    # 2. 准备数据
    train_features = np.load(cfg.TRAIN_FEATURES_PATH)
    train_dataset = EVChargerDatasetV2(train_features, cfg.HISTORY_LEN, cfg.PRED_LEN, cfg, scaler_y, scaler_mm, scaler_z)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, sampler=train_sampler, drop_last=True, pin_memory=True, num_workers=4)

    # 3. 初始化模型并加载权重
    model = SpatioTemporalDiffusionModelV2(
        in_features=cfg.TARGET_FEAT_DIM, out_features=cfg.TARGET_FEAT_DIM,
        history_features=cfg.HISTORY_FEATURES, static_features=cfg.STATIC_FEATURES, future_known_features=cfg.FUTURE_KNOWN_FEAT_DIM,
        model_dim=cfg.MODEL_DIM, num_heads=cfg.NUM_HEADS, T=cfg.T, depth=cfg.DEPTH
    ).to(device_id)

    # 寻找最佳权重文件
    # 优先尝试加载 mae_best，其次 best，最后 second_best
    possible_weights = [
        # f"./weights/st_diffusion_model_v2_{RESUME_RUN_ID}_mae_best.pth",
        f"./weights/st_diffusion_model_v2_{RESUME_RUN_ID}_best.pth",
        # f"./weights/st_diffusion_model_v2_{RESUME_RUN_ID}_mae_second_best.pth"
    ]
    loaded = False
    for w_path in possible_weights:
        if os.path.exists(w_path):
            map_location = {'cuda:%d' % 0: 'cuda:%d' % device_id}
            state_dict = torch.load(w_path, map_location=f"cuda:{device_id}")
            model.load_state_dict(state_dict)
            if rank == 0: print(f"[Info] Successfully loaded weights from: {w_path}")
            loaded = True
            break
    
    if not loaded:
        raise FileNotFoundError(f"Could not find any weight file for Run ID {RESUME_RUN_ID} in ./weights/")

    ddp_model = DDP(model, device_ids=[device_id], find_unused_parameters=False)
    optimizer = optim.AdamW(ddp_model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=1e-4)
    scaler = amp.GradScaler()

    # 简单的 StepLR 或者保持 Constant，因为只有几轮
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    len_list = calc_layer_lengths(cfg.PRED_LEN, cfg.DEPTH)
    L_max = len_list[0]
    E_orig = original_edge_index.shape[1]
    (full_edge_index, full_edge_weights) = batch_time_edge_index(
        original_edge_index, original_edge_weights, cfg.NUM_NODES, cfg.BATCH_SIZE, L_max, device_id
    )
    edge_data = []
    for L_d in len_list:
        num_edges_needed = E_orig * cfg.BATCH_SIZE * L_d
        slice_idx = full_edge_index[:, :num_edges_needed]
        slice_w = full_edge_weights[:num_edges_needed]
        edge_data.append((slice_idx, slice_w))

    # 4. 训练循环 (无验证)
    finetuned_model_path = cfg.MODEL_SAVE_PATH_TEMPLATE.format(run_id=cfg.RUN_ID, rank="")

    for epoch in range(cfg.EPOCHS):
        train_sampler.set_epoch(epoch)
        ddp_model.train()
        progress_bar = tqdm(train_dataloader, desc=f"Resume Epoch {epoch+1}/{cfg.EPOCHS}", disable=(rank != 0))
        total_train_loss = 0.0
        optimizer.zero_grad()

        for i, (history_c, static_c, future_x0, future_known_c, idx) in enumerate(progress_bar):
            tensors = [d.to(device_id) for d in (history_c, static_c, future_x0, future_known_c)]
            history_c, static_c, future_x0, future_known_c = tensors
            
            with amp.autocast():
                b = future_x0.shape[0]
                history_c_p = history_c.permute(0, 2, 1, 3)
                future_x0_p = future_x0.permute(0, 2, 1, 3)
                future_known_c_p = future_known_c.permute(0, 2, 1, 3)

                noise = torch.randn_like(future_x0_p)
                offset_scale = 0.1
                offset = offset_scale * torch.randn(b, 1, 1, 1, device=device_id)
                noise = noise + offset

                k = torch.randint(0, cfg.T, (b,), device=device_id).long()
                sqrt_alpha_bar_k = ddp_model.module.sqrt_alphas_cumprod[k].view(b, 1, 1, 1)
                sqrt_one_minus_alpha_bar_k = ddp_model.module.sqrt_one_minus_alphas_cumprod[k].view(b, 1, 1, 1)
                x_k = sqrt_alpha_bar_k * future_x0_p + sqrt_one_minus_alpha_bar_k * noise

                predicted_noise, predicted_logvar = ddp_model(x_k, k, history_c_p, static_c, future_known_c_p, edge_data, edge_data)
                pred_logvar = torch.clamp(predicted_logvar, -5.0, 3.0)
                nll = 0.5 * torch.exp(-pred_logvar) * (noise - predicted_noise) ** 2 + 0.5 * pred_logvar
                logvar_reg = 1e-3 * (pred_logvar ** 2)
                loss = (nll + logvar_reg).mean() / cfg.ACCUMULATION_STEPS

            if (i + 1) % cfg.ACCUMULATION_STEPS == 0 or (i + 1) == len(train_dataloader):
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                with ddp_model.no_sync():
                    scaler.scale(loss).backward()

            total_train_loss += loss.item() * cfg.ACCUMULATION_STEPS
            if rank == 0: progress_bar.set_postfix(loss=loss.item())

        avg_train_loss_local = total_train_loss / len(train_dataloader)
        avg_train_loss_tensor = torch.tensor(avg_train_loss_local).to(device_id)
        dist.all_reduce(avg_train_loss_tensor, op=dist.ReduceOp.SUM)
        avg_train_loss = avg_train_loss_tensor.item() / world_size

        if rank == 0:
            print(f"Epoch {epoch+1} Done. Avg Train Loss: {avg_train_loss:.4f}")
            epoch_log = {'epoch': epoch + 1, 'avg_train_loss': avg_train_loss, 'lr': optimizer.param_groups[0]['lr']}
            csv_logger.log_epoch(epoch_log)

        scheduler.step()
    
    # 5. 保存模型
    if rank == 0:
        print(f"Saving finetuned model to {finetuned_model_path}...")
        torch.save(ddp_model.module.state_dict(), finetuned_model_path)
    
    dist.barrier() # 等待保存完成

    # 6. 运行评估
    if rank == 0:
        print("\n" + "="*50 + "\nStarting Evaluation on Finetuned Model...\n" + "="*50)
    
    metrics = evaluate_model(
        cfg, finetuned_model_path, scaler_y, scaler_mm, scaler_z,
        device=f"cuda:{device_id}", rank=rank, world_size=world_size
    )

    if rank == 0:
        print_metrics(metrics)
        # 可选：保存结果到文件
        # np.save(f'./results/metrics_{cfg.RUN_ID}.npy', metrics)

    dist.destroy_process_group()

if __name__ == "__main__":
    resume_training()