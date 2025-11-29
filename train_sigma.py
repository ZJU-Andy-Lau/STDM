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

# 从您工作区中的 model_sigma.py 导入 V2 模型
from model_sigma import SpatioTemporalDiffusionModelV2

from scheduler import MultiStageOneCycleLR

# --- [新增] 模块化日志记录器 ---
class CsvLogger:
    """
    一个用于 DDP 训练的模块化 CSV 日志记录器。
    它只会在 rank 0 进程上创建和写入文件。
    """
    def __init__(self, log_dir, run_id, rank, headers):
        """
        初始化记录器。

        参数:
            log_dir (str): 日志文件存放的目录 (例如: './results')
            run_id (str): 当前运行的 ID，用作文件名 (例如: '20251028_100000')
            rank (int): 当前 DDP 进程的 rank。
            headers (list[str]): CSV 文件的表头 (例如: ['epoch', 'train_loss', 'val_loss'])
        """
        self.log_dir = log_dir
        self.run_id = run_id
        self.rank = rank
        self.headers = headers
        self.log_file_path = None

        # 只有 rank 0 进程执行文件操作
        if self.rank == 0:
            try:
                # 确保日志目录存在
                os.makedirs(self.log_dir, exist_ok=True)
                
                # 定义日志文件路径
                self.log_file_path = os.path.join(self.log_dir, f"{self.run_id}.csv")
                
                # 创建文件并写入表头
                with open(self.log_file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(self.headers)
                print(f"[Rank 0] CsvLogger initialized. Logging to {self.log_file_path}")
                    
            except Exception as e:
                print(f"[Rank 0 Logger Error] Failed to initialize logger: {e}")
                self.log_file_path = None # 初始化失败，禁用日志

    def log_epoch(self, epoch_data):
        """
        记录一个 epoch 的数据。

        参数:
            epoch_data (dict): 包含要记录数据的字典。
                               键 (key) 必须与初始化时的 headers 对应。
                               例如: {'epoch': 1, 'train_loss': 0.5, ...}
        """
        # 只有 rank 0 且日志文件已成功初始化时才写入
        if self.rank != 0 or self.log_file_path is None:
            return

        try:
            # 按表头顺序准备要写入的数据行
            # 使用 .get(h, '') 来优雅地处理缺失值：
            # 如果字典中没有某个键 (例如 'avg_val_mae')，
            # 它会写入一个空字符串，而不是-
            row = [epoch_data.get(h, '') for h in self.headers]
            
            # 以追加模式 ('a') 打开文件并写入数据
            with open(self.log_file_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(row)
                
        except Exception as e:
            print(f"[Rank 0 Logger Error] Failed to log epoch data: {e}")

# --- V2 版本配置参数 ---
class ConfigV2:
    NORMALIZATION_TYPE = "minmax" 
    RUN_ID = None 

    # 数据参数
    NUM_NODES = 275
    HISTORY_LEN = 12
    PRED_LEN = 12
    
    # 特征维度定义
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
    
    # --- [核心修改] 集成训练参数 ---
    ENSEMBLE_K = 4          # 对于每个样本，生成 K 个不同的预测
    ENSEMBLE_LAMBDA = 1.0   # 集成损失的权重
    LOGVAR_LAMBDA = 0.1     # 原始 LogVar 辅助损失的权重 (防止 LogVar 分支坍缩)
    
    # 训练参数
    EPOCHS = 100
    BATCH_SIZE = 2 # 注意：这是【单张卡】的batch size。实际输入到模型的 batch 会变为 BATCH_SIZE * ENSEMBLE_K
    LEARNING_RATE = 1e-4
    ACCUMULATION_STEPS = 4

    WARMUP_EPOCHS = 5      # 预热阶段的 Epoch 数量
    COOLDOWN_EPOCHS = 50    # 退火阶段的 Epoch 数量
    CYCLE_EPOCHS = 10    # 每个余弦退火周期的 Epoch 数量 (T_0)
    
    # 文件路径模板
    MODEL_SAVE_PATH_TEMPLATE = "./weights/st_diffusion_model_v2_{run_id}_{rank}.pth"
    SCALER_Y_SAVE_PATH_TEMPLATE = "./weights/scaler_y_v2_{run_id}.pkl"
    SCALER_MM_SAVE_PATH_TEMPLATE = "./weights/scaler_mm_v2_{run_id}.pkl"
    SCALER_Z_SAVE_PATH_TEMPLATE = "./weights/scaler_z_v2_{run_id}.pkl"

    # --- 周期性 MAE 评估的配置 ---
    EVAL_ON_VAL = True               # 是否开启周期性 MAE 评估
    EVAL_ON_VAL_EPOCH = 5            # 每 5 个 epoch 运行一次
    EVAL_ON_VAL_BATCHES = 48         # 使用 48 个 batch 进行评估
    EVAL_ON_VAL_SAMPLES = 5          # 评估时生成 5 个样本
    EVAL_ON_VAL_STEPS = 20           # 评估时使用 20 步采样 (为了速度)
    SAMPLING_ETA = 0.0               # 评估时使用 DDIM (eta=0.0)
    EVAL_SEED = 42 

    # 数据文件路径
    TRAIN_FEATURES_PATH = './urbanev/features_train_wea_poi.npy'
    VAL_FEATURES_PATH = './urbanev/features_valid_wea_poi.npy'
    TEST_FEATURES_PATH = './urbanev/features_test_wea_poi.npy'
    ADJ_MATRIX_PATH = './urbanev/dis.npy'

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

# --- 数据集类 (保持不变) ---
class EVChargerDatasetV2(Dataset):
    def __init__(self, features, history_len, pred_len, cfg, scaler_y=None,scaler_mm=None,scaler_z=None):
        self.cfg = cfg

        minmax_features = features[:, :, cfg.HISTORY_FEATURES-4:cfg.HISTORY_FEATURES+5].copy()
        zscore_features = features[:, :, cfg.HISTORY_FEATURES+5:].copy()
        

        dynamic_features = features[:, :, :cfg.HISTORY_FEATURES].copy()
        static_features = features[0, :, cfg.HISTORY_FEATURES:].copy()

        target_col_original = dynamic_features[:, :, 0]

        if scaler_y is None:
            self.scaler_y = self._initialize_scaler_y(target_col_original)
        else:
            self.scaler_y = scaler_y

        if scaler_mm is None:
            self.scaler_mm = self._initialize_scaler_mm(minmax_features)
        else:
            self.scaler_mm = scaler_mm
        
        if scaler_z is None:
            self.scaler_z = self._initialize_scaler_z(zscore_features)
        else:
            self.scaler_z = scaler_z

        if self.cfg.NORMALIZATION_TYPE != "none":
            target_col_reshaped = target_col_original.reshape(-1, 1)
            normalized_target = self.scaler_y.transform(target_col_reshaped)
            dynamic_features[:, :, 0] = normalized_target.reshape(target_col_original.shape)

        mm_norm = self.scaler_mm.transform(
            minmax_features.reshape(-1, minmax_features.shape[-1])
        ).reshape(minmax_features.shape)

        z_norm = self.scaler_z.transform(
            zscore_features.reshape(-1, zscore_features.shape[-1])
        ).reshape(zscore_features.shape)


        dynamic_features[:, :, cfg.HISTORY_FEATURES-4:] = mm_norm[:, :, :4]

        static_features[:, :5] = mm_norm[0, :, -5:]
        static_features[:, 5:] = z_norm[0, :, :]
        self.static_features = torch.tensor(static_features, dtype=torch.float)
        self.samples = create_sliding_windows(dynamic_features, history_len, pred_len)

    def _initialize_scaler_y(self, data):
        if self.cfg.NORMALIZATION_TYPE == "minmax":
            scaler = MinMaxScaler(feature_range=(-1, 1))
        elif self.cfg.NORMALIZATION_TYPE == "zscore":
            scaler = StandardScaler()
        else: return None
        scaler.fit(data.reshape(-1, 1))
        return scaler
    
    def _initialize_scaler_mm(self, data):
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(data.reshape(-1, data.shape[-1]))
        return scaler

    def _initialize_scaler_z(self, data):
        scaler = StandardScaler()
        scaler.fit(data.reshape(-1, data.shape[-1]))
        return scaler

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        history, future = self.samples[idx]
        history_c = torch.tensor(history, dtype=torch.float)
        future_x0 = torch.tensor(future[:, :, :self.cfg.TARGET_FEAT_DIM], dtype=torch.float)
        known_start_idx = self.cfg.TARGET_FEAT_DIM + (self.cfg.DYNAMIC_FEAT_DIM - self.cfg.FUTURE_KNOWN_FEAT_DIM)
        future_known_c = torch.tensor(future[:, :, known_start_idx : self.cfg.HISTORY_FEATURES], dtype=torch.float)
        return history_c, self.static_features, future_x0, future_known_c, idx


    def get_scaler(self):
        return self.scaler_y, self.scaler_mm, self.scaler_z

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

def set_seed(seed):
    """设置随机种子以确保可复现性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

# --- 周期性 MAE 评估函数 ---
@torch.no_grad()
def periodic_evaluate_mae(model, loader, scaler, edge_index, edge_weights, cfg, device):
    """
    在验证集子集上运行 MAE 评估。
    此函数只应在 rank 0 上调用。
    """
    model.eval() 
    
    rank = dist.get_rank()
    
    progress_bar = tqdm(
        loader, 
        desc=f"Periodic Val MAE (Rank {rank})", 
        disable=(rank != 0), 
        ncols=100
    )

    all_predictions_list = []
    for (history_c, static_c, future_x0_true, future_known_c, idx) in progress_bar:
        tensors = [d.to(device) for d in (history_c, static_c, future_x0_true, future_known_c)]
        history_c, static_c, future_x0_true, future_known_c = tensors
        
        b = history_c.shape[0]
        len_list = calc_layer_lengths(cfg.PRED_LEN, cfg.DEPTH)
        edge_data = [batch_time_edge_index(edge_index, edge_weights, cfg.NUM_NODES, b, len_list[d], device) for d in range(cfg.DEPTH + 1)]

        generated_samples = []
        for _ in range(cfg.EVAL_ON_VAL_SAMPLES):
            sample = model.ddim_sample(
                history_c=history_c.permute(0, 2, 1, 3), 
                static_c=static_c,
                future_known_c=future_known_c.permute(0, 2, 1, 3),
                history_edge_data=edge_data,
                future_edge_data=edge_data,
                shape=future_x0_true.permute(0, 2, 1, 3).shape, 
                sampling_steps=cfg.EVAL_ON_VAL_STEPS, 
                eta=cfg.SAMPLING_ETA 
            )
            generated_samples.append(sample)
        
        stacked_samples = torch.stack(generated_samples, dim=0)
        median_prediction = torch.median(stacked_samples, dim=0).values

        denorm_pred = median_prediction.cpu().numpy()
        denorm_true = future_x0_true.permute(0, 2, 1, 3).cpu().numpy()
        if scaler:
            denorm_pred = scaler.inverse_transform(denorm_pred.reshape(-1, 1)).reshape(denorm_pred.shape)
            denorm_true = scaler.inverse_transform(denorm_true.reshape(-1, 1)).reshape(denorm_true.shape)

        denorm_pred = np.clip(denorm_pred, 0.0, 1.0) 

        all_predictions_list.append(denorm_pred)


    model.train() 
    return np.concatenate(all_predictions_list, axis=0).squeeze(-1).transpose(0, 2, 1)



# --- 主训练函数 ---
def train():
    dist.init_process_group("nccl")
    rank = int(os.environ["RANK"])
    device_id = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(device_id)

    cfg = ConfigV2()

    # --- 初始化变量 ---
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

    # --- 日志 ---
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

    # --- 核心修改：为 Ensemble Batch 预生成 Edge Index ---
    # 实际 Batch Size 变为 B * K
    effective_batch_size = cfg.BATCH_SIZE * cfg.ENSEMBLE_K
    
    (full_edge_index, full_edge_weights) = batch_time_edge_index(
        original_edge_index, 
        original_edge_weights, 
        cfg.NUM_NODES, 
        effective_batch_size, # 这里使用扩展后的 batch size
        L_max, 
        device_id
    )
    edge_data = []
    for L_d in len_list:
        num_edges_needed = E_orig * effective_batch_size * L_d
        slice_idx = full_edge_index[:, :num_edges_needed]
        slice_w = full_edge_weights[:num_edges_needed]
        edge_data.append((slice_idx, slice_w))

    # --- 验证时使用标准的 Batch Size (不需要 K 倍) ---
    # 所以需要准备一套原始大小的 edge_data 给 validation loop
    (val_full_edge_index, val_full_edge_weights) = batch_time_edge_index(
        original_edge_index, 
        original_edge_weights, 
        cfg.NUM_NODES, 
        cfg.BATCH_SIZE, 
        L_max, 
        device_id
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
            tensors = [d.to(device_id) for d in (history_c, static_c, future_x0, future_known_c)]
            history_c, static_c, future_x0, future_known_c = tensors
            
            with amp.autocast():
                # --- 1. 数据扩充 (Batch Expansion) ---
                K = cfg.ENSEMBLE_K
                batch_size_curr = future_x0.shape[0]
                
                # [B, ...] -> [B*K, ...]
                history_c_exp = history_c.repeat_interleave(K, dim=0).permute(0, 2, 1, 3)
                static_c_exp = static_c.repeat_interleave(K, dim=0)
                future_x0_exp = future_x0.repeat_interleave(K, dim=0).permute(0, 2, 1, 3)
                future_known_c_exp = future_known_c.repeat_interleave(K, dim=0).permute(0, 2, 1, 3)
                
                expanded_bs = batch_size_curr * K
                
                # --- 2. 噪声生成 ---
                # 为每个扩充后的样本生成独立的噪声，这样对于同一个原始样本，模型会看到 K 个不同的噪声版本
                noise = torch.randn_like(future_x0_exp)
                
                # 时间步采样: 对每个原始样本采样一个 t，然后重复 K 次
                # 这样同一个样本的 K 个副本在同一个时间步 t 进行训练 (便于计算分布)
                t = torch.randint(0, cfg.T, (batch_size_curr,), device=device_id).long()
                t_exp = t.repeat_interleave(K, dim=0)
                
                # 加噪
                sqrt_alpha_bar = ddp_model.module.sqrt_alphas_cumprod[t_exp].view(expanded_bs, 1, 1, 1)
                sqrt_one_minus_alpha_bar = ddp_model.module.sqrt_one_minus_alphas_cumprod[t_exp].view(expanded_bs, 1, 1, 1)
                x_t = sqrt_alpha_bar * future_x0_exp + sqrt_one_minus_alpha_bar * noise

                # --- 3. 模型前向传播 ---
                # 注意：edge_data 已经是为扩展后的 Batch Size 准备的
                # 处理最后一个不满的 batch 的 edge_data 切片
                curr_edge_data = []
                for (e_idx, e_w) in edge_data:
                    num_edges = E_orig * expanded_bs * (e_idx.shape[1] // (E_orig * effective_batch_size))
                    curr_edge_data.append((e_idx[:, :num_edges], e_w[:num_edges]))

                predicted_noise, predicted_logvar = ddp_model(
                    x_t, t_exp, history_c_exp, static_c_exp, future_known_c_exp, curr_edge_data, curr_edge_data
                )
                
                # --- 4. 集成分布损失计算 (Ensemble Distribution Loss) ---
                
                # (A) 重构 x0 (Estimate x0 from predicted noise)
                # formula: x0 = (xt - sqrt(1-alpha_bar) * eps) / sqrt(alpha_bar)
                # 添加 eps 防止除零 (虽然 alpha_bar 通常 > 0)
                pred_x0_exp = (x_t - sqrt_one_minus_alpha_bar * predicted_noise) / (sqrt_alpha_bar + 1e-8)
                
                # (B) 重塑维度以分组: [B*K, C, N, L] -> [B, K, C, N, L]
                pred_x0_grouped = pred_x0_exp.view(batch_size_curr, K, cfg.TARGET_FEAT_DIM, cfg.NUM_NODES, cfg.PRED_LEN)
                target_x0 = future_x0.permute(0, 3, 1, 2) # [B, C, N, L]
                
                # (C) 计算集成统计量
                ensemble_mean = pred_x0_grouped.mean(dim=1) # [B, C, N, L]
                ensemble_var = pred_x0_grouped.var(dim=1, unbiased=False) + 1e-6 # [B, C, N, L], 加上 epsilon 防止 log(0)
                
                # (D) 集成 NLL 损失 (Ensemble NLL Loss)
                # 迫使 K 个预测的均值接近真值，且分布符合高斯
                # Loss = 0.5 * log(var) + 0.5 * (target - mean)^2 / var
                ensemble_nll = 0.5 * torch.log(ensemble_var) + 0.5 * (target_x0 - ensemble_mean)**2 / ensemble_var
                
                # (E) 信噪比加权 (SNR Weighting)
                # 由于在高噪声区域(t很大)，x0 的重构误差会非常大，我们需要降低其权重
                # 使用 alpha_bar 作为权重 (类似于 SNR 权重)
                weights = ddp_model.module.alphas_cumprod[t].view(batch_size_curr, 1, 1, 1)
                weighted_ensemble_loss = (ensemble_nll * weights).mean()
                
                # (F) 辅助损失: LogVar Regularization
                # 依然需要训练 LogVar 分支，以保证推理时 ddim_sample 正常工作
                # 对每个样本单独计算标准的 NLL
                min_logvar, max_logvar = -5.0, 3.0
                pred_logvar_clamped = torch.clamp(predicted_logvar, min_logvar, max_logvar)
                aux_nll = 0.5 * torch.exp(-pred_logvar_clamped) * (noise - predicted_noise) ** 2 + 0.5 * pred_logvar_clamped
                aux_loss = aux_nll.mean()

                # (G) 总损失
                loss = cfg.ENSEMBLE_LAMBDA * weighted_ensemble_loss + cfg.LOGVAR_LAMBDA * aux_loss
                loss = loss / cfg.ACCUMULATION_STEPS

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

        ddp_model.eval()
        total_val_loss = 0
        
        # 验证集使用标准计算方式 (不进行 Ensemble 扩展，或者仅计算单次 NLL)
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

                    # 处理 val edge data
                    curr_val_edge_data = []
                    for (e_idx, e_w) in val_edge_data:
                         num_edges = E_orig * b * (e_idx.shape[1] // (E_orig * cfg.BATCH_SIZE))
                         curr_val_edge_data.append((e_idx[:, :num_edges], e_w[:num_edges]))

                    pred_noise, pred_logvar = ddp_model(
                        x_k, k, history_c_p, static_c, future_known_c_p, curr_val_edge_data, curr_val_edge_data
                    )

                    min_logvar, max_logvar = -5.0, 3.0
                    pred_logvar = torch.clamp(pred_logvar, min_logvar, max_logvar)

                    nll = 0.5 * torch.exp(-pred_logvar) * (noise - pred_noise) ** 2 + 0.5 * pred_logvar
                    val_loss = nll.mean()
                
                total_val_loss += val_loss.item()

        avg_train_loss_local = total_train_loss / len(train_dataloader)
        avg_train_loss_tensor = torch.tensor(avg_train_loss_local).to(device_id)
        dist.all_reduce(avg_train_loss_tensor, op=dist.ReduceOp.SUM)
        avg_train_loss = avg_train_loss_tensor.item() / world_size

        avg_val_loss_local = total_val_loss / len(val_dataloader)
        avg_val_loss_tensor = torch.tensor(avg_val_loss_local).to(device_id)
        dist.all_reduce(avg_val_loss_tensor, op=dist.ReduceOp.SUM)
        avg_val_loss = avg_val_loss_tensor.item() / world_size

        # --- 日志数据准备 ---
        if rank == 0:
            epoch_log_data = {
                'epoch': epoch + 1,
                'avg_train_loss': avg_train_loss,
                'avg_val_loss': avg_val_loss,
                'lr': optimizer.param_groups[0]['lr'],
                'avg_val_mae': '' 
            }

        # --- 模型保存逻辑 ---
        if rank == 0:
            print(f"Epoch {epoch+1} Summary: Avg Train Loss: {avg_train_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}")
            if avg_val_loss < best_val_loss:
                second_best_val_loss = best_val_loss
                best_val_loss = avg_val_loss
                if best_model_path_for_val is not None and os.path.exists(best_model_path_for_val):
                    os.rename(best_model_path_for_val, model_save_path_second_best)
                    print(f"Model {os.path.basename(best_model_path_for_val)} promoted to 2nd best.")
                    second_best_model_path_for_val = model_save_path_second_best
                torch.save(ddp_model.module.state_dict(), model_save_path_best)
                best_model_path_for_val = model_save_path_best
                print(f"🎉 New best model saved to {model_save_path_best} with validation loss: {best_val_loss:.4f}")
            
            elif avg_val_loss < second_best_val_loss:
                second_best_val_loss = avg_val_loss
                torch.save(ddp_model.module.state_dict(), model_save_path_second_best)
                second_best_model_path_for_val = model_save_path_second_best
                print(f"🥈 New 2nd best model saved to {model_save_path_second_best} with validation loss: {second_best_val_loss:.4f}")

        run_mae_eval = cfg.EVAL_ON_VAL and (epoch + 1) % cfg.EVAL_ON_VAL_EPOCH == 0
        if run_mae_eval:
            current_val_mae = float('inf') 
            
            print(f"Epoch {epoch+1}, running periodic MAE evaluation...")
            # 注意：periodic_evaluate_mae 使用 val_edge_data (标准 batch size)
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

                mae_log = f", Avg Val MAE: {current_val_mae:.4f}" if current_val_mae != float('inf') else ", Avg Val MAE: (skipped)"
                print(f"Epoch {epoch+1} Summary: Avg Train Loss: {avg_train_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}{mae_log}, LR: {optimizer.param_groups[0]['lr']:.2e}")            
            
                if current_val_mae < best_val_mae:
                    second_best_val_mae = best_val_mae
                    if best_model_path_for_eval is not None and os.path.exists(best_model_path_for_eval):
                        if os.path.exists(model_save_path_mae_second_best):
                            try:
                                os.remove(model_save_path_mae_second_best)
                            except OSError as e:
                                print(f"Warning: Could not remove old second_best model: {e}")
                        os.rename(best_model_path_for_eval, model_save_path_mae_second_best)
                        print(f"Model {os.path.basename(best_model_path_for_eval)} promoted to 2nd best (MAE).")
                        second_best_model_path_for_eval = model_save_path_mae_second_best

                    best_val_mae = current_val_mae
                    torch.save(ddp_model.module.state_dict(), model_save_path_mae_best)
                    best_model_path_for_eval = model_save_path_mae_best
                    print(f"🎉 New best model saved to {model_save_path_mae_best} with validation MAE: {best_val_mae:.4f}")

                elif current_val_mae < second_best_val_mae:
                    second_best_val_mae = current_val_mae
                    torch.save(ddp_model.module.state_dict(), model_save_path_mae_second_best)
                    second_best_model_path_for_eval = model_save_path_mae_second_best
                    print(f"🥈 New 2nd best model saved to {model_save_path_mae_second_best} with validation MAE: {second_best_val_mae:.4f}")

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

    metrics_best = None
    metrics_second_best = None
    metrics_best_val = None
    metrics_second_best_val = None

    if best_model_path_for_val_synced and os.path.exists(best_model_path_for_val_synced):
        if rank == 0:
            print(f"\n[ALL GPUS] Evaluating BEST VAL model (in parallel): {os.path.basename(best_model_path_for_val_synced)}")
        metrics_best_val = evaluate_model(
            cfg, best_model_path_for_val_synced, scaler_y_save_path, scaler_mm_save_path, scaler_z_save_path,
            device=f"cuda:{device_id}", rank=rank, world_size=world_size,key='best_val'
        )
        if rank == 0:
            print(f"[ALL GPUS] Finished evaluating BEST VAL model.")
            print(f"===== [FINAL RESULT 1/4] BEST VAL Model ({os.path.basename(best_model_path_for_val_synced)}) =====")
            print_metrics(metrics_best_val)
    else:
        if rank == 0:
            print("No best val model was saved. Skipping evaluation.")
            
    if second_best_model_path_for_val_synced and os.path.exists(second_best_model_path_for_val_synced):
        if rank == 0:
             print(f"\n[ALL GPUS] Evaluating 2ND BEST VAL model (in parallel): {os.path.basename(second_best_model_path_for_val_synced)}")
        metrics_second_best_val = evaluate_model(
            cfg, second_best_model_path_for_val_synced, scaler_y_save_path, scaler_mm_save_path, scaler_z_save_path,
            device=f"cuda:{device_id}", rank=rank, world_size=world_size,key='second_best_val'
        )
        if rank == 0:
            print(f"[ALL GPUS] Finished evaluating 2ND BEST VAL model.")
            print(f"\n===== [FINAL RESULT 2/4] 2ND BEST VAL Model ({os.path.basename(second_best_model_path_for_val_synced)}) =====")
            print_metrics(metrics_second_best_val)
    else:
        if rank == 0:
            print(f"[ALL GPUS] No second best val model was saved. Skipping evaluation.")

    if best_model_path_synced and os.path.exists(best_model_path_synced):
        if rank == 0:
            print(f"\n[ALL GPUS] Evaluating BEST model (in parallel): {os.path.basename(best_model_path_synced)}")
        metrics_best = evaluate_model(
            cfg, best_model_path_synced, scaler_y_save_path, scaler_mm_save_path, scaler_z_save_path,
            device=f"cuda:{device_id}", rank=rank, world_size=world_size,key='best'
        )
        if rank == 0:
            print(f"[ALL GPUS] Finished evaluating BEST model.")
            print(f"===== [FINAL RESULT 3/4] BEST Model ({os.path.basename(best_model_path_synced)}) =====")
            print_metrics(metrics_best)
    else:
        if rank == 0:
            print("No best model was saved. Skipping evaluation.")
            
    if second_best_model_path_synced and os.path.exists(second_best_model_path_synced):
        if rank == 0:
             print(f"\n[ALL GPUS] Evaluating 2ND BEST model (in parallel): {os.path.basename(second_best_model_path_synced)}")
        metrics_second_best = evaluate_model(
            cfg, second_best_model_path_synced, scaler_y_save_path, scaler_mm_save_path, scaler_z_save_path,
            device=f"cuda:{device_id}", rank=rank, world_size=world_size,key='second_best'
        )
        if rank == 0:
            print(f"[ALL GPUS] Finished evaluating 2ND BEST model.")
            print(f"\n===== [FINAL RESULT 4/4] 2ND BEST Model ({os.path.basename(second_best_model_path_synced)}) =====")
            print_metrics(metrics_second_best)
    else:
        if rank == 0:
            print(f"[ALL GPUS] No second best model was saved. Skipping evaluation.")


    if rank == 0:
        print("\n" + "="*50 + "\nSequential Evaluation Results\n" + "="*50 + "\n")
        
        if metrics_best:
            print(f"===== [FINAL RESULT 1/4] BEST Model ({os.path.basename(best_model_path_synced)}) =====")
            print_metrics(metrics_best)
        else:
            print("===== [FINAL RESULT 1/4] BEST Model: SKIPPED (not saved or error) =====")
            
        if metrics_second_best:
            print(f"\n===== [FINAL RESULT 2/4] 2ND BEST Model ({os.path.basename(second_best_model_path_synced)}) =====")
            print_metrics(metrics_second_best)
        else:
            print("\n===== [FINAL RESULT 2/4] 2ND BEST Model: SKIPPED (not saved or error) =====")
        
        if metrics_best_val:
            print(f"===== [FINAL RESULT 3/4] BEST VAL Model ({os.path.basename(best_model_path_for_val_synced)}) =====")
            print_metrics(metrics_best_val)
        else:
            print("===== [FINAL RESULT 3/4] BEST VAL Model: SKIPPED (not saved or error) =====")
            
        if metrics_second_best_val:
            print(f"\n===== [FINAL RESULT 4/4] 2ND BEST VAL Model ({os.path.basename(second_best_model_path_for_val_synced)}) =====")
            print_metrics(metrics_second_best_val)
        else:
            print("\n===== [FINAL RESULT 4/4] 2ND BEST VAL Model: SKIPPED (not saved or error) =====")

    dist.destroy_process_group()

class EvalConfig(ConfigV2):
    BATCH_SIZE = 8
    NUM_SAMPLES = 20
    SAMPLING_STEPS = 50
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def calculate_metrics(y_true, y_pred_median, y_pred_samples, device):
    metrics = {}
    metrics['mae'] = float(np.mean(np.abs(y_pred_median - y_true)))
    metrics['rmse'] = float(np.sqrt(np.mean((y_pred_median - y_true)**2)))
    rae_numerator = np.sum(np.abs(y_pred_median - y_true))
    rae_denominator = np.sum(np.abs(np.mean(y_true) - y_true))
    metrics['rae'] = float(rae_numerator / (rae_denominator + 1e-9))
    non_zero_mask = y_true != 0
    metrics['mape'] = float(np.mean(np.abs((y_pred_median[non_zero_mask] - y_true[non_zero_mask]) / y_true[non_zero_mask])) * 100) if np.any(non_zero_mask) else 0.0
    metrics['crps'] = float(ps.crps_ensemble(y_true, y_pred_samples.transpose(1, 0, 2, 3), axis=0).mean())

    horizon_metrics = []
    for i in range(y_true.shape[1]): # 遍历预测长度
        y_true_h, y_pred_median_h, y_pred_samples_h = y_true[:, i, :], y_pred_median[:, i, :], y_pred_samples[:, :, i, :]
        mae_h = float(np.mean(np.abs(y_pred_median_h - y_true_h)))
        rmse_h = float(np.sqrt(np.mean((y_pred_median_h - y_true_h)**2)))
        rae_num_h = np.sum(np.abs(y_pred_median_h - y_true_h))
        rae_den_h = np.sum(np.abs(np.mean(y_true_h) - y_true_h))
        rae_h = float(rae_num_h / (rae_den_h + 1e-9))
        non_zero_mask_h = y_true_h != 0
        if np.any(non_zero_mask_h):
            mape_h = float(np.mean(np.abs((y_pred_median_h[non_zero_mask_h] - y_true_h[non_zero_mask_h]) / y_true_h[non_zero_mask_h])) * 100)
        else:
            mape_h = 0.0
        crps_h = float(ps.crps_ensemble(y_true_h, y_pred_samples_h.transpose(1, 0, 2), axis=0).mean())
        horizon_metrics.append([f't+{i+1}', mae_h, rmse_h, rae_h, mape_h, crps_h])

    df = pd.DataFrame(horizon_metrics, columns=['Horizon', 'MAE', 'RMSE', 'RAE', 'MAPE', 'CRPS'])
    metrics['horizon_metrics'] = df.to_dict('records')
    return metrics

def print_metrics(metrics):
    print("\n--- Overall Metrics ---")
    print(f"MAE:  {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"RAE:  {metrics['rae']:.4f}")
    print(f"MAPE: {metrics['mape']:.2f}%")
    print(f"CRPS: {metrics['crps']:.4f}")
    print("-----------------------\n")
    
    if 'horizon_metrics' in metrics:
        horizon_df = pd.DataFrame(metrics['horizon_metrics'])
        print("--- Horizon-wise Metrics ---")
        print(horizon_df)
        print("----------------------------\n")

    if 'dm_stat' in metrics:
        print("--- Significance Test (Diebold-Mariano) ---")
        print(f"Comparing Your Model against Naive Baseline:")
        print(f"DM Statistic: {metrics['dm_stat']:.4f}, P-value: {metrics['p_value']:.7f}")
        if metrics['p_value'] < 0.05:
            print("Conclusion: Your model is STATISTICALLY SIGNIFICANTLY BETTER than the Naive baseline (p < 0.05).")
        else:
            print("Conclusion: No statistical evidence that your model is better than the Naive baseline (p >= 0.05).")
        print("--------------------------------------------\n")

def dm_test(e1, e2, h=12, crit="MAD"):
    e1, e2 = np.array(e1), np.array(e2)
    d = np.abs(e1) - np.abs(e2)
    d_mean = np.mean(d)
    n = len(d)
    d_centered = d - d_mean
    acov = np.correlate(d_centered, d_centered, mode="full")[n-1:n+h] / n
    var_d = acov[0] + 2 * np.sum([(1 - lag/(h+1)) * acov[lag] for lag in range(1, h)])
    var_d = max(var_d, 1e-12)
    dm_stat = d_mean / np.sqrt(var_d / n)
    p_value = 1 - t.cdf(dm_stat, df=n-1)
    return dm_stat, p_value

def evaluate_model(train_cfg, model_path, scaler_y_path, scaler_mm_path, scaler_z_path, device, rank, world_size,key):
    cfg = EvalConfig()
    cfg.RUN_ID = train_cfg.RUN_ID
    cfg.NORMALIZATION_TYPE = train_cfg.NORMALIZATION_TYPE
    cfg.DEVICE = device

    # set_seed(cfg.EVAL_SEED)
    
    model = SpatioTemporalDiffusionModelV2(
        in_features=cfg.TARGET_FEAT_DIM, out_features=cfg.TARGET_FEAT_DIM,
        history_features=cfg.HISTORY_FEATURES, static_features=cfg.STATIC_FEATURES,
        future_known_features=cfg.FUTURE_KNOWN_FEAT_DIM, model_dim=cfg.MODEL_DIM,
        num_heads=cfg.NUM_HEADS, T=cfg.T, depth=cfg.DEPTH
    ).to(cfg.DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=cfg.DEVICE))
    model.eval()
    print("Model loaded successfully.")

    adj_matrix = np.load(cfg.ADJ_MATRIX_PATH)
    test_features = np.load(cfg.TEST_FEATURES_PATH)
    scaler_y = joblib.load(scaler_y_path) if os.path.exists(scaler_y_path) else None  
    scaler_mm = joblib.load(scaler_mm_path) if os.path.exists(scaler_mm_path) else None
    scaler_z = joblib.load(scaler_z_path) if os.path.exists(scaler_z_path) else None

    adj_matrix = np.load(cfg.ADJ_MATRIX_PATH)
    distances = adj_matrix[adj_matrix > 0]
    sigma = np.std(distances)

    adj_matrix = np.exp(-np.square(adj_matrix) / (sigma**2))
    adj_matrix[adj_matrix < 0.1] = 0
    
    original_edge_index, original_edge_weights = get_edge_index(adj_matrix)
    test_dataset = EVChargerDatasetV2(test_features, cfg.HISTORY_LEN, cfg.PRED_LEN, cfg, scaler_y=scaler_y, scaler_mm=scaler_mm, scaler_z=scaler_z)
    print(f"Test dataset size: {len(test_dataset)} samples.")
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, sampler=test_sampler)
    
    all_predictions_list, all_samples_list, all_true_list, all_idx_list = [], [], [], []

    disable_tqdm = (dist.is_initialized() and dist.get_rank() != 0)
    with torch.no_grad(), amp.autocast():
        for tensors in tqdm(test_dataloader, desc="Evaluating", disable=disable_tqdm):
            history_c, static_c, future_x0_true, future_known_c, idx = tensors
            history_c = history_c.to(cfg.DEVICE)
            static_c = static_c.to(cfg.DEVICE)
            future_x0_true = future_x0_true.to(cfg.DEVICE)
            future_known_c = future_known_c.to(cfg.DEVICE)

            all_true_list.append(future_x0_true.permute(0, 2, 1, 3).cpu().numpy())
            all_idx_list.append(idx.cpu().numpy()) 

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
                    shape=future_x0_true.permute(0, 2, 1, 3).shape, sampling_steps=cfg.SAMPLING_STEPS,
                    eta=cfg.SAMPLING_ETA
                )
                generated_samples.append(sample)
            
            stacked_samples = torch.stack(generated_samples, dim=0)
            median_prediction = torch.median(stacked_samples, dim=0).values

            denorm_pred, denorm_samples = median_prediction.cpu().numpy(), stacked_samples.cpu().numpy()
            all_predictions_list.append(median_prediction.cpu().numpy())
            all_samples_list.append(stacked_samples.cpu().numpy())

    all_predictions = np.concatenate(all_predictions_list, axis=0).squeeze(-1).transpose(0, 2, 1)
    all_samples = np.concatenate(all_samples_list, axis=1).squeeze(-1).transpose(1, 0, 3, 2)

    print(f"Rank {rank} :idx list:{all_idx_list}")

    if not all_predictions_list:
        local_predictions = np.empty((0, cfg.PRED_LEN, cfg.NUM_NODES, cfg.TARGET_FEAT_DIM), dtype=np.float32)
        local_samples = np.empty((cfg.NUM_SAMPLES, 0, cfg.PRED_LEN, cfg.NUM_NODES, cfg.TARGET_FEAT_DIM), dtype=np.float32)
        local_true = np.empty((0, cfg.PRED_LEN, cfg.NUM_NODES, cfg.TARGET_FEAT_DIM), dtype=np.float32)
        local_idx = np.empty((0,), dtype=np.int64)
    else:
        local_predictions = np.concatenate(all_predictions_list, axis=0)
        local_samples = np.concatenate(all_samples_list, axis=1)
        local_true = np.concatenate(all_true_list, axis=0)
        local_idx = np.concatenate(all_idx_list, axis=0) 

    gathered_preds = [None] * world_size
    gathered_samples = [None] * world_size
    gathered_true = [None] * world_size
    gathered_idx = [None] * world_size

    dist.gather_object(local_predictions, gathered_preds if rank == 0 else None, dst=0)
    dist.gather_object(local_samples, gathered_samples if rank == 0 else None, dst=0)
    dist.gather_object(local_true, gathered_true if rank == 0 else None, dst=0)
    dist.gather_object(local_idx, gathered_idx if rank == 0 else None, dst=0)
    
    if rank == 0:
        all_predictions_norm = np.concatenate(gathered_preds, axis=0)
        all_samples_norm = np.concatenate(gathered_samples, axis=1)
        all_true_norm = np.concatenate(gathered_true, axis=0)
        all_idx = np.concatenate(gathered_idx, axis=0)
        print(f"gather index:{all_idx}")

        order = np.argsort(all_idx)
        print(f"order:{order}")
        all_predictions_norm = all_predictions_norm[order]
        all_true_norm = all_true_norm[order]
        all_samples_norm = all_samples_norm[:, order]

        if scaler_y:
            all_predictions = scaler_y.inverse_transform(all_predictions_norm.reshape(-1, 1)).reshape(all_predictions_norm.shape)
            y_true_original = scaler_y.inverse_transform(all_true_norm.reshape(-1, 1)).reshape(all_true_norm.shape)
            all_samples = scaler_y.inverse_transform(all_samples_norm.reshape(-1, 1)).reshape(all_samples_norm.shape)
        else:
            all_predictions = all_predictions_norm
            y_true_original = all_true_norm
            all_samples = all_samples_norm

        all_predictions = np.clip(all_predictions, 0.0, 1.0)
        all_samples = np.clip(all_samples, 0.0, 1.0)

        all_predictions = all_predictions.squeeze(-1).transpose(0, 2, 1)
        y_true_original = y_true_original.squeeze(-1).transpose(0, 2, 1)
        all_samples = all_samples.squeeze(-1).transpose(1, 0, 3, 2)

        print(f"y_true shape:{y_true_original.shape}")
        print(f"all_predictions shape:{all_predictions.shape}")
        print(f"all_samples shape:{all_samples.shape}")

        np.save(f'./results/truths.npy', y_true_original)
        np.save(f'./results/pred_{cfg.RUN_ID}_{key}.npy', all_predictions)
        np.save(f'./results/samples_{cfg.RUN_ID}_{key}.npy', all_samples)

        try:
            all_baseline_preds = np.load("./urbanev/TimeXer_predictions.npy")
            all_baseline_preds = np.concatenate([all_baseline_preds[:, :, -1:], all_baseline_preds], axis=-1)[:, :, :-1]
            all_baseline_preds = all_baseline_preds[:all_predictions.shape[0]]
            y_true_original = y_true_original[:all_baseline_preds.shape[0]]
            all_predictions = all_predictions[:all_baseline_preds.shape[0]]
            all_samples = all_samples[:all_baseline_preds.shape[0]]
            print("\n已加载基线模型 (TimeXer) 预测，用于 DM 显著性检验。")
            print(f"all_baseline_preds shape:{all_baseline_preds.shape}")
            perform_significance = True
        except FileNotFoundError:
            print("\n警告：基线预测文件 ./urbanev/TimeXer_predictions.npy 未找到。")
            print("跳过 DM 显著性检验。")
            perform_significance = False

        final_metrics = calculate_metrics(y_true_original, all_predictions, all_samples, cfg.DEVICE)
        if perform_significance:
            errors_model = np.abs(y_true_original.flatten() - all_predictions.flatten())
            errors_baseline = np.abs(y_true_original.flatten() - all_baseline_preds.flatten())
            dm_statistic, p_value = dm_test(errors_baseline, errors_model)
            final_metrics['dm_stat'] = dm_statistic
            final_metrics['p_value'] = p_value
            
        return final_metrics
    else:
        return None

if __name__ == "__main__":
    train()