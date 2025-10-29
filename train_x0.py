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

# 从您工作区中的 model_x0.py 导入 V2 模型
from model_x0 import SpatioTemporalDiffusionModelV2

from scheduler import MultiStageOneCycleLR

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

# --- V2 版本配置参数 (已修改以对齐V2.3) ---
class ConfigV2:
    NORMALIZATION_TYPE = "minmax" 
    RUN_ID = None 

    # 数据参数
    NUM_NODES = 275
    HISTORY_LEN = 12
    PRED_LEN = 12
    
    # 特征维度定义
    TARGET_FEAT_DIM = 1
    DYNAMIC_FEAT_DIM = 10
    STATIC_FEAT_DIM = 4
    FUTURE_KNOWN_FEAT_DIM = 8
    
    HISTORY_FEATURES = TARGET_FEAT_DIM + DYNAMIC_FEAT_DIM
    STATIC_FEATURES = STATIC_FEAT_DIM
    
    # 模型参数
    MODEL_DIM = 64
    NUM_HEADS = 4
    DEPTH = 4
    T = 1000
    
    # 训练参数
    EPOCHS = 100
    BATCH_SIZE = 4 # 注意：这是【单张卡】的batch size
    LEARNING_RATE = 5e-5
    ACCUMULATION_STEPS = 1

    WARMUP_EPOCHS = 5      # 预热阶段的 Epoch 数量
    COOLDOWN_EPOCHS = 50    # 退火阶段的 Epoch 数量
    CYCLE_EPOCHS = 10    # 每个余弦退火周期的 Epoch 数量 (T_0)
    
    # --- 核心修改1: 修改文件路径模板以保存最佳和次佳模型 ---
    MODEL_SAVE_PATH_TEMPLATE = "./weights/st_diffusion_model_v2_{run_id}_{rank}.pth"
    SCALER_SAVE_PATH_TEMPLATE = "./weights/scaler_v2_{run_id}.pkl"

    # --- 周期性 MAE 评估的配置 ---
    EVAL_ON_VAL = True               # 是否开启周期性 MAE 评估
    EVAL_ON_VAL_EPOCH = 5            # 每 5 个 epoch 运行一次
    EVAL_ON_VAL_BATCHES = 48         # 使用 48 个 batch 进行评估 (48 * BATCH_SIZE=192 个样本)
    EVAL_ON_VAL_SAMPLES = 5          # 评估时生成 5 个样本
    EVAL_ON_VAL_STEPS = 20           # 评估时使用 20 步采样 (为了速度)
    SAMPLING_ETA = 0.0               # 评估时使用 DDIM (eta=0.0)
    EVAL_SEED = 42    

    # 数据文件路径
    TRAIN_FEATURES_PATH = './urbanev/features_train_v2.npy'
    VAL_FEATURES_PATH = './urbanev/features_valid_v2.npy' # 暂时尝试将测试集作为验证集
    TEST_FEATURES_PATH = './urbanev/features_test_v2.npy'
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
    def __init__(self, features, history_len, pred_len, cfg, scaler=None):
        self.cfg = cfg
        dynamic_features = features[:, :, :cfg.HISTORY_FEATURES].copy()
        self.static_features = torch.tensor(features[0, :, cfg.HISTORY_FEATURES:], dtype=torch.float)

        target_col_original = dynamic_features[:, :, 0]

        if scaler is None:
            self.scaler = self._initialize_scaler(target_col_original)
        else:
            self.scaler = scaler

        if self.cfg.NORMALIZATION_TYPE != "none":
            target_col_reshaped = target_col_original.reshape(-1, 1)
            normalized_target = self.scaler.transform(target_col_reshaped)
            dynamic_features[:, :, 0] = normalized_target.reshape(target_col_original.shape)
        
        self.samples = create_sliding_windows(dynamic_features, history_len, pred_len)

    def _initialize_scaler(self, data):
        if self.cfg.NORMALIZATION_TYPE == "minmax":
            scaler = MinMaxScaler(feature_range=(-1, 1))
        elif self.cfg.NORMALIZATION_TYPE == "zscore":
            scaler = StandardScaler()
        else: return None
        scaler.fit(data.reshape(-1, 1))
        return scaler

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        history, future = self.samples[idx]
        history_c = torch.tensor(history, dtype=torch.float)
        future_x0 = torch.tensor(future[:, :, :self.cfg.TARGET_FEAT_DIM], dtype=torch.float)
        known_start_idx = self.cfg.TARGET_FEAT_DIM + (self.cfg.DYNAMIC_FEAT_DIM - self.cfg.FUTURE_KNOWN_FEAT_DIM)
        future_known_c = torch.tensor(future[:, :, known_start_idx : self.cfg.HISTORY_FEATURES], dtype=torch.float)
        return history_c, self.static_features, future_x0, future_known_c

    def get_scaler(self):
        return self.scaler

def calc_layer_lengths(L_in, depth, kernel_size=3, stride=2, padding=1, dilation=1):
    """
    根据Conv1d参数计算每一层的输出长度
    
    参数：
        L_in : int      # 初始序列长度
        depth : int     # 网络层数（下采样层数）
        kernel_size : int
        stride : int
        padding : int
        dilation : int
        
    返回：
        lengths : list[int]  # 每一层的输出长度（包含输入层）
    """
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

# --- 新增修改: 周期性 MAE 评估函数 ---
@torch.no_grad()
def periodic_evaluate_mae(model, loader, scaler, edge_index, edge_weights, cfg, device):
    """
    在验证集子集上运行 MAE 评估。
    此函数只应在 rank 0 上调用。
    """
    set_seed(cfg.EVAL_SEED)
    model.eval() # 确保模型处于评估模式
    
    # --- 核心修改: 获取 rank 以便只在 rank 0 上显示 tqdm ---
    rank = dist.get_rank()
    
    progress_bar = tqdm(
        loader, 
        desc=f"Periodic Val MAE (Rank {rank})", 
        disable=(rank != 0), # 只在 rank 0 上显示
        ncols=100
    )

    all_predictions_list = []
    for (history_c, static_c, future_x0_true, future_known_c) in progress_bar:
        # --- 核心修改: 所有 rank 并行执行 ---
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
                sampling_steps=cfg.EVAL_ON_VAL_STEPS, # 使用更少的步数以加速
                eta=cfg.SAMPLING_ETA # 确保使用 eta=0.0
            )
            generated_samples.append(sample)
        
        stacked_samples = torch.stack(generated_samples, dim=0)
        median_prediction = torch.median(stacked_samples, dim=0).values

        # 反归一化以计算真实 MAE
        denorm_pred = median_prediction.cpu().numpy()
        denorm_true = future_x0_true.permute(0, 2, 1, 3).cpu().numpy()
        if scaler:
            denorm_pred = scaler.inverse_transform(denorm_pred.reshape(-1, 1)).reshape(denorm_pred.shape)
            denorm_true = scaler.inverse_transform(denorm_true.reshape(-1, 1)).reshape(denorm_true.shape)
        
        all_predictions_list.append(denorm_pred)


    model.train() # 评估后将模型设置回训练模式
    
    # --- 核心修改: 返回局部的(local)结果 ---
    return np.concatenate(all_predictions_list, axis=0).squeeze(-1).transpose(0, 2, 1)



# --- 主训练函数 (已修改) ---
def train():
    dist.init_process_group("nccl")
    rank = int(os.environ["RANK"])
    device_id = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(device_id)

    cfg = ConfigV2()

    # --- 核心修改2: 初始化最佳/次佳模型跟踪变量 (仅主进程) ---
    if rank == 0:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg.RUN_ID = run_id
        print(f"Starting DDP training. Run ID: {cfg.RUN_ID}")
        os.makedirs(os.path.dirname(cfg.MODEL_SAVE_PATH_TEMPLATE), exist_ok=True)
        
        model_save_path_best = cfg.MODEL_SAVE_PATH_TEMPLATE.format(run_id=cfg.RUN_ID, rank="best")
        model_save_path_second_best = cfg.MODEL_SAVE_PATH_TEMPLATE.format(run_id=cfg.RUN_ID, rank="second_best")
        
        model_save_path_mae_best = cfg.MODEL_SAVE_PATH_TEMPLATE.format(run_id=cfg.RUN_ID, rank="mae_best")
        model_save_path_mae_second_best = cfg.MODEL_SAVE_PATH_TEMPLATE.format(run_id=cfg.RUN_ID, rank="mae_second_best")

        scaler_save_path = cfg.SCALER_SAVE_PATH_TEMPLATE.format(run_id=cfg.RUN_ID)

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
        scaler_save_path = cfg.SCALER_SAVE_PATH_TEMPLATE.format(run_id=cfg.RUN_ID)

    log_headers = ['epoch', 'avg_train_loss', 'avg_val_loss', 'lr', 'avg_val_mae']
    csv_logger = CsvLogger(
        log_dir='./results',  # 指定日志目录
        run_id=cfg.RUN_ID,    # 使用全局唯一的 run_id
        rank=rank,            # 传入当前进程的 rank
        headers=log_headers
    )


    adj_matrix = np.load(cfg.ADJ_MATRIX_PATH)
    distances = adj_matrix[adj_matrix > 0]
    sigma = np.std(distances)

    # 应用高斯核函数
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
        joblib.dump(train_dataset.get_scaler(), scaler_save_path)
    dist.barrier()
    
    train_scaler = joblib.load(scaler_save_path) if os.path.exists(scaler_save_path) else None

    val_dataset = EVChargerDatasetV2(val_features, cfg.HISTORY_LEN, cfg.PRED_LEN, cfg, scaler=train_scaler)
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
    # criterion = nn.MSELoss()
    criterion = nn.HuberLoss()
    scaler = amp.GradScaler()

    # warmup_scheduler = LambdaLR(
    #     optimizer,
    #     lr_lambda=lambda epoch: (epoch + 1) / cfg.WARMUP_EPOCHS if epoch < cfg.WARMUP_EPOCHS else 1
    # )

    # main_scheduler = CosineAnnealingWarmRestarts(
    #     optimizer,
    #     T_0=cfg.CYCLE_EPOCHS,
    #     T_mult=1,
    #     eta_min=1e-6
    # )

    # scheduler = SequentialLR(
    #     optimizer,
    #     schedulers=[warmup_scheduler, main_scheduler],
    #     milestones=[cfg.WARMUP_EPOCHS]
    # )


    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS, eta_min=1e-6)

    scheduler = MultiStageOneCycleLR(optimizer=optimizer,
                                     total_steps=cfg.EPOCHS,
                                     warmup_ratio=cfg.WARMUP_EPOCHS / cfg.EPOCHS,
                                     cooldown_ratio=cfg.COOLDOWN_EPOCHS / cfg.EPOCHS)

    len_list = calc_layer_lengths(cfg.PRED_LEN, cfg.DEPTH)
    L_max = len_list[0]
    E_orig = original_edge_index.shape[1]

    (full_edge_index, full_edge_weights) = batch_time_edge_index(
        original_edge_index, 
        original_edge_weights, 
        cfg.NUM_NODES, 
        cfg.BATCH_SIZE, 
        L_max, 
        device_id
    )
    edge_data = []
    for L_d in len_list:
        num_edges_needed = E_orig * cfg.BATCH_SIZE * L_d

        # 这些是视图(views)，不是拷贝(copies)，几乎不占内存
        slice_idx = full_edge_index[:, :num_edges_needed]
        slice_w = full_edge_weights[:num_edges_needed]
        edge_data.append((slice_idx, slice_w))

    original_val_samples = create_sliding_windows(val_features, cfg.HISTORY_LEN, cfg.PRED_LEN)
    y_true_original = np.array([s[1][:, :, :cfg.TARGET_FEAT_DIM] for s in original_val_samples]).squeeze(-1)[:cfg.EVAL_ON_VAL_BATCHES * cfg.BATCH_SIZE]

    for epoch in range(cfg.EPOCHS):
        train_sampler.set_epoch(epoch)
        ddp_model.train()
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS} [Train]", disable=(rank != 0))
        total_train_loss = 0.0
        optimizer.zero_grad()

        for i, (history_c, static_c, future_x0, future_known_c) in enumerate(progress_bar):
            tensors = [d.to(device_id) for d in (history_c, static_c, future_x0, future_known_c)]
            history_c, static_c, future_x0, future_known_c = tensors
            
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

                predicted_x0 = ddp_model(x_k, k, history_c_p, static_c, future_known_c_p, edge_data, edge_data)
                loss = criterion(predicted_x0, future_x0_p)
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
        
        with torch.no_grad():
            for tensors in val_dataloader:
                history_c, static_c, future_x0, future_known_c = [d.to(device_id) for d in tensors]
                
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

                    predicted_x0 = ddp_model(x_k, k, history_c_p, static_c, future_known_c_p, edge_data, edge_data)
                    val_loss = criterion(predicted_x0, future_x0_p)

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
            # 初始化本 epoch 的日志数据字典
            epoch_log_data = {
                'epoch': epoch + 1,
                'avg_train_loss': avg_train_loss,
                'avg_val_loss': avg_val_loss,
                'lr': optimizer.param_groups[0]['lr'],
                'avg_val_mae': ''  # 默认为空字符串，如果 MAE 不运行，则记录为空
            }


        # --- 核心修改3: 更新保存最佳/次佳模型的逻辑 ---
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
            current_val_mae = float('inf') # 默认为无穷大
            
            print(f"Epoch {epoch+1}, running periodic MAE evaluation...")
            current_val_seq_local = periodic_evaluate_mae(
                ddp_model.module, # 传入原始模型
                val_eval_loader,
                train_scaler,
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
                # 确保数据量正确 (因为 DistributedSampler 可能会填充数据)
                current_val_seq = current_val_seq[:len(y_true_original)]
                current_val_mae = np.mean(np.abs(current_val_seq - y_true_original))

                epoch_log_data['avg_val_mae'] = current_val_mae

                # 打印日志
                mae_log = f", Avg Val MAE: {current_val_mae:.4f}" if current_val_mae != float('inf') else ", Avg Val MAE: (skipped)"
                print(f"Epoch {epoch+1} Summary: Avg Train Loss: {avg_train_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}{mae_log}, LR: {optimizer.param_groups[0]['lr']:.2e}")            
            
                # --- 核心修改4: 使用 MAE (如果已计算) 来保存模型 ---
                # 只有当 current_val_mae 被计算过时 (不为 inf)，才执行保存逻辑
                if current_val_mae < best_val_mae:
                    second_best_val_mae = best_val_mae
                    if best_model_path_for_eval is not None and os.path.exists(best_model_path_for_eval):
                        # 检查旧的 "second_best" 路径是否存在，如果存在先删除
                        if os.path.exists(model_save_path_mae_second_best):
                            try:
                                os.remove(model_save_path_mae_second_best)
                            except OSError as e:
                                print(f"Warning: Could not remove old second_best model: {e}")
                        # 将旧的 "best" 重命名为 "second_best"
                        os.rename(best_model_path_for_eval, model_save_path_second_best)
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

    # --- 核心修改4: 采用V2.3的分布式评估流程 ---
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

    # 1. --- 所有卡并行评估 BEST model ---
    if best_model_path_synced and os.path.exists(best_model_path_synced):
        if rank == 0:
            print(f"\n[ALL GPUS] Evaluating BEST model (in parallel): {os.path.basename(best_model_path_synced)}")
        # 所有进程都调用 evaluate_model，函数内部会处理 DDP
        metrics_best = evaluate_model(
            cfg, best_model_path_synced, scaler_save_path, 
            device=f"cuda:{device_id}", rank=rank, world_size=world_size,key='best'
        )
        if rank == 0:
            print(f"[ALL GPUS] Finished evaluating BEST model.")
            print(f"===== [FINAL RESULT 1/4] BEST Model ({os.path.basename(best_model_path_synced)}) =====")
            print_metrics(metrics_best)
    else:
        if rank == 0:
            print("No best model was saved. Skipping evaluation.")
            
    # 2. --- 所有卡并行评估 2ND BEST model ---
    if second_best_model_path_synced and os.path.exists(second_best_model_path_synced):
        if rank == 0:
             print(f"\n[ALL GPUS] Evaluating 2ND BEST model (in parallel): {os.path.basename(second_best_model_path_synced)}")
        # 所有进程再次调用 evaluate_model
        metrics_second_best = evaluate_model(
            cfg, second_best_model_path_synced, scaler_save_path, 
            device=f"cuda:{device_id}", rank=rank, world_size=world_size,key='second_best'
        )
        if rank == 0:
            print(f"[ALL GPUS] Finished evaluating 2ND BEST model.")
            print(f"\n===== [FINAL RESULT 2/4] 2ND BEST Model ({os.path.basename(second_best_model_path_synced)}) =====")
            print_metrics(metrics_second_best)
    else:
        if rank == 0:
            print(f"[ALL GPUS] No second best model was saved. Skipping evaluation.")

    # 3. --- 所有卡并行评估 BEST VAL model ---
    if best_model_path_for_val_synced and os.path.exists(best_model_path_for_val_synced):
        if rank == 0:
            print(f"\n[ALL GPUS] Evaluating BEST VAL model (in parallel): {os.path.basename(best_model_path_for_val_synced)}")
        # 所有进程都调用 evaluate_model，函数内部会处理 DDP
        metrics_best_val = evaluate_model(
            cfg, best_model_path_for_val_synced, scaler_save_path, 
            device=f"cuda:{device_id}", rank=rank, world_size=world_size,key='best_val'
        )
        if rank == 0:
            print(f"[ALL GPUS] Finished evaluating BEST VAL model.")
            print(f"===== [FINAL RESULT 3/4] BEST VAL Model ({os.path.basename(best_model_path_for_val_synced)}) =====")
            print_metrics(metrics_best_val)
    else:
        if rank == 0:
            print("No best val model was saved. Skipping evaluation.")
            
    # 4. --- 所有卡并行评估 2ND BEST VAL model ---
    if second_best_model_path_for_val_synced and os.path.exists(second_best_model_path_for_val_synced):
        if rank == 0:
             print(f"\n[ALL GPUS] Evaluating 2ND BEST VAL model (in parallel): {os.path.basename(second_best_model_path_for_val_synced)}")
        # 所有进程再次调用 evaluate_model
        metrics_second_best_val = evaluate_model(
            cfg, second_best_model_path_for_val_synced, scaler_save_path, 
            device=f"cuda:{device_id}", rank=rank, world_size=world_size,key='second_best_val'
        )
        if rank == 0:
            print(f"[ALL GPUS] Finished evaluating 2ND BEST VAL model.")
            print(f"\n===== [FINAL RESULT 4/4] 2ND BEST VAL Model ({os.path.basename(second_best_model_path_for_val_synced)}) =====")
            print_metrics(metrics_second_best_val)
    else:
        if rank == 0:
            print(f"[ALL GPUS] No second best val model was saved. Skipping evaluation.")

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

# =============================================================================
# ===================== 核心修改5: 全面替换为V2.3的评估逻辑 =====================
# =============================================================================

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

def evaluate_model(train_cfg, model_path, scaler_path, device, rank, world_size,key):
    cfg = EvalConfig()
    cfg.RUN_ID = train_cfg.RUN_ID
    cfg.NORMALIZATION_TYPE = train_cfg.NORMALIZATION_TYPE
    cfg.DEVICE = device

    set_seed(cfg.EVAL_SEED)
    
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
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    adj_matrix = np.load(cfg.ADJ_MATRIX_PATH)
    distances = adj_matrix[adj_matrix > 0]
    sigma = np.std(distances)

    # 应用高斯核函数
    adj_matrix = np.exp(-np.square(adj_matrix) / (sigma**2))
    adj_matrix[adj_matrix < 0.1] = 0
    
    original_edge_index, original_edge_weights = get_edge_index(adj_matrix)
    test_dataset = EVChargerDatasetV2(test_features, cfg.HISTORY_LEN, cfg.PRED_LEN, cfg, scaler=scaler)
    # --- 使用 DistributedSampler 切分测试集 ---
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, sampler=test_sampler)
    
    # original_test_samples = create_sliding_windows(test_features, cfg.HISTORY_LEN, cfg.PRED_LEN)
    # y_true_original = np.array([s[1][:, :, :cfg.TARGET_FEAT_DIM] for s in original_test_samples]).squeeze(-1)

    all_predictions_list, all_samples_list, all_true_list = [], [], []

    disable_tqdm = (dist.is_initialized() and dist.get_rank() != 0)
    with torch.no_grad(), amp.autocast():
        for tensors in tqdm(test_dataloader, desc="Evaluating", disable=disable_tqdm):
            history_c, static_c, future_x0_true, future_known_c = [d.to(cfg.DEVICE) for d in tensors]
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
                    shape=future_x0_true.permute(0, 2, 1, 3).shape, sampling_steps=cfg.SAMPLING_STEPS,
                    eta=cfg.SAMPLING_ETA
                )
                generated_samples.append(sample)
            
            stacked_samples = torch.stack(generated_samples, dim=0)
            median_prediction = torch.median(stacked_samples, dim=0).values

            
            # if scaler:
            #     denorm_pred = scaler.inverse_transform(denorm_pred.reshape(-1, 1)).reshape(denorm_pred.shape)
            #     denorm_samples = scaler.inverse_transform(denorm_samples.reshape(-1, 1)).reshape(denorm_samples.shape)
            
            # all_predictions_list.append(denorm_pred)
            # all_samples_list.append(denorm_samples)
            all_predictions_list.append(median_prediction.cpu().numpy())
            all_samples_list.append(stacked_samples.cpu().numpy())

    all_predictions = np.concatenate(all_predictions_list, axis=0).squeeze(-1).transpose(0, 2, 1)
    all_samples = np.concatenate(all_samples_list, axis=1).squeeze(-1).transpose(1, 0, 3, 2)

    if not all_predictions_list:
        # 如果某个 rank 没有数据 (例如数据集大小不能被 world_size 整除且 drop_last=False)
        # 我们创建一个空的数组以避免 gather 失败
        local_predictions = np.empty((0, cfg.PRED_LEN, cfg.NUM_NODES, cfg.TARGET_FEAT_DIM), dtype=np.float32)
        local_samples = np.empty((cfg.NUM_SAMPLES, 0, cfg.PRED_LEN, cfg.NUM_NODES, cfg.TARGET_FEAT_DIM), dtype=np.float32)
        local_true = np.empty((0, cfg.PRED_LEN, cfg.NUM_NODES, cfg.TARGET_FEAT_DIM), dtype=np.float32)
    else:
        local_predictions = np.concatenate(all_predictions_list, axis=0)
        local_samples = np.concatenate(all_samples_list, axis=1)
        local_true = np.concatenate(all_true_list, axis=0)

    gathered_preds = [None] * world_size
    gathered_samples = [None] * world_size
    gathered_true = [None] * world_size

    dist.gather_object(local_predictions, gathered_preds if rank == 0 else None, dst=0)
    dist.gather_object(local_samples, gathered_samples if rank == 0 else None, dst=0)
    dist.gather_object(local_true, gathered_true if rank == 0 else None, dst=0)

    if rank == 0:
        # 拼接所有 GPU 的结果
        all_predictions_norm = np.concatenate(gathered_preds, axis=0)
        all_samples_norm = np.concatenate(gathered_samples, axis=1)
        all_true_norm = np.concatenate(gathered_true, axis=0)

        # 1. 在 rank 0 上进行反归一化
        if scaler:
            # (B, N, L, 1) -> (B*N*L, 1) -> (B*N*L, 1) -> (B, N, L, 1)
            all_predictions = scaler.inverse_transform(all_predictions_norm.reshape(-1, 1)).reshape(all_predictions_norm.shape)
            y_true_original = scaler.inverse_transform(all_true_norm.reshape(-1, 1)).reshape(all_true_norm.shape)
            # (S, B, N, L, 1) -> (S*B*N*L, 1) -> (S*B*N*L, 1) -> (S, B, N, L, 1)
            all_samples = scaler.inverse_transform(all_samples_norm.reshape(-1, 1)).reshape(all_samples_norm.shape)
        else:
            all_predictions = all_predictions_norm
            y_true_original = all_true_norm
            all_samples = all_samples_norm

        # 2. 调整维度以匹配 calculate_metrics 函数的期望
        # (B, N, L, 1) -> (B, N, L) -> (B, L, N)
        all_predictions = all_predictions.squeeze(-1).transpose(0, 2, 1)
        y_true_original = y_true_original.squeeze(-1).transpose(0, 2, 1)
        # (S, B, N, L, 1) -> (S, B, N, L) -> (B, S, L, N)
        all_samples = all_samples.squeeze(-1).transpose(1, 0, 3, 2)

        print(f"y_true shape:{y_true_original.shape}")
        print(f"all_predictions shape:{all_predictions.shape}")
        print(f"all_samples shape:{all_samples.shape}")

        np.save(f'./results/pred_{cfg.RUN_ID}_{key}.npy', all_predictions)
        np.save(f'./results/samples_{cfg.RUN_ID}_{key}.npy', all_samples)

        try:
            # 注意：这个基线文件路径是硬编码的
            all_baseline_preds = np.load("./urbanev/TimeXer_predictions.npy")
            # 基线模型 shape 适配
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
