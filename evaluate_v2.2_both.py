import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
from tqdm import tqdm
import json
import os
import argparse  # 导入argparse
from datetime import datetime
from torch.cuda import amp
from scipy.stats import t
import pandas as pd
import properscoring as ps
from contextlib import nullcontext
import math
import random

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib

# --- 导入分布式训练所需的库 ---
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# --- 导入模型 ---
# 确保 model_v2_gcngat.py 在同一个目录下
try:
    from model_v2_gcngat import SpatioTemporalDiffusionModelV2
except ImportError:
    print("错误：无法导入 'model_v2_gcngat.py'。")
    print("请确保 'model_v2_gcngat.py' 文件与此脚本在同一目录中。")
    exit(1)

# =============================================================================
# 1. 从 train_v2.2_both.py 复制所有依赖项
# =============================================================================

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
    DYNAMIC_FEAT_DIM = 9
    STATIC_FEAT_DIM = 4
    FUTURE_KNOWN_FEAT_DIM = 7
    
    HISTORY_FEATURES = TARGET_FEAT_DIM + DYNAMIC_FEAT_DIM
    STATIC_FEATURES = STATIC_FEAT_DIM
    
    # 模型参数
    MODEL_DIM = 64
    NUM_HEADS = 4
    DEPTH = 4
    T = 1000
    
    # 训练参数 (评估时大部分用不到)
    EPOCHS = 100
    BATCH_SIZE = 4 
    LEARNING_RATE = 1e-4
    ACCUMULATION_STEPS = 1

    WARMUP_EPOCHS = 5
    COOLDOWN_EPOCHS = 50
    CYCLE_EPOCHS = 10
    
    MODEL_SAVE_PATH_TEMPLATE = "./weights/st_diffusion_model_v2_{run_id}_{rank}.pth"
    SCALER_SAVE_PATH_TEMPLATE = "./weights/scaler_v2_{run_id}.pkl"

    EVAL_ON_VAL = True
    EVAL_ON_VAL_EPOCH = 5
    EVAL_ON_VAL_BATCHES = 50
    EVAL_ON_VAL_SAMPLES = 5
    EVAL_ON_VAL_STEPS = 20
    SAMPLING_ETA = 0.0
    EVAL_SEED = 42 

    # 数据文件路径
    TRAIN_FEATURES_PATH = './urbanev/features_train_v2.npy'
    VAL_FEATURES_PATH = './urbanev/features_test_v2.npy'
    TEST_FEATURES_PATH = './urbanev/features_test_v2.npy'
    ADJ_MATRIX_PATH = './urbanev/dis.npy'

# --- 评估专用配置 ---
class EvalConfig(ConfigV2):
    BATCH_SIZE = 8
    NUM_SAMPLES = 20
    SAMPLING_STEPS = 50
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 辅助函数：图处理 ---
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

# --- 辅助函数：滑动窗口 ---
def create_sliding_windows(data, history_len, pred_len):
    samples = []
    total_len = len(data)
    for i in range(total_len - history_len - pred_len + 1):
        history = data[i : i + history_len]
        future = data[i + history_len : i + history_len + pred_len]
        samples.append((history, future))
    return samples

# --- 辅助函数：数据集类 ---
class EVChargerDatasetV2(Dataset):
    def __init__(self, features, history_len, pred_len, cfg, scaler=None):
        self.cfg = cfg
        dynamic_features = features[:, :, :cfg.HISTORY_FEATURES].copy()
        self.static_features = torch.tensor(features[0, :, cfg.HISTORY_FEATURES:], dtype=torch.float)

        target_col_original = dynamic_features[:, :, 0]

        if scaler is None:
            # 在评估模式下，我们不应该初始化scaler，只应该加载
            print("警告：评估时未提供Scaler。如果模型已归一化，结果将不准确。")
            self.scaler = None
        else:
            self.scaler = scaler

        if self.scaler and self.cfg.NORMALIZATION_TYPE != "none":
            target_col_reshaped = target_col_original.reshape(-1, 1)
            normalized_target = self.scaler.transform(target_col_reshaped)
            dynamic_features[:, :, 0] = normalized_target.reshape(target_col_original.shape)
        
        self.samples = create_sliding_windows(dynamic_features, history_len, pred_len)

    def _initialize_scaler(self, data):
        # 评估脚本不负责创建 scaler
        raise RuntimeError("评估脚本不应初始化 Scaler。请提供已训练的 Scaler。")

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

# --- 辅助函数：U-Net 长度计算 ---
def calc_layer_lengths(L_in, depth, kernel_size=3, stride=2, padding=1, dilation=1):
    lengths = [L_in]
    for i in range(depth):
        L_prev = lengths[-1]
        L_out = math.floor((L_prev + 2*padding - dilation*(kernel_size - 1) - 1) / stride + 1)
        lengths.append(L_out)
    return lengths

# --- 辅助函数：图批处理 (核心优化) ---
def batch_time_edge_index(edge_index, edge_weights, num_nodes, batch_size, time_steps, device):
    if time_steps <= 0: return torch.empty(2, 0, dtype=torch.long, device=device)
    num_total_graphs = batch_size * time_steps
    edge_index_list = [edge_index + i * num_nodes for i in range(num_total_graphs)]
    batched_edge_index = torch.cat(edge_index_list, dim=1).to(device)
    batched_edge_weights = edge_weights.repeat(num_total_graphs).to(device)
    return batched_edge_index, batched_edge_weights

# --- 辅助函数：设置随机种子 ---
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

# --- 辅助函数：指标计算 ---
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

# --- 辅助函数：指标打印 ---
def print_metrics(metrics):
    print("\n--- 总体指标 (Overall Metrics) ---")
    print(f"MAE:  {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"RAE:  {metrics['rae']:.4f}")
    print(f"MAPE: {metrics['mape']:.2f}%")
    print(f"CRPS: {metrics['crps']:.4f}")
    print("----------------------------------\n")
    
    if 'horizon_metrics' in metrics:
        horizon_df = pd.DataFrame(metrics['horizon_metrics'])
        print("--- 分步指标 (Horizon-wise Metrics) ---")
        print(horizon_df.to_string(index=False))
        print("---------------------------------------\n")

    if 'dm_stat' in metrics:
        print("--- 显著性检验 (Diebold-Mariano) ---")
        print(f"对比模型 vs. Naive Baseline:")
        print(f"DM 统计量: {metrics['dm_stat']:.4f}, P-value: {metrics['p_value']:.7f}")
        if metrics['p_value'] < 0.05:
            print("结论: 模型表现**显著优于** Naive baseline (p < 0.05)。")
        else:
            print("结论: 没有统计证据表明模型优于 Naive baseline (p >= 0.05)。")
        print("--------------------------------------------\n")

# --- 辅助函数：DM 检验 ---
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

# =============================================================================
# 2. 从 train_v2.2_both.py 复制完整的 evaluate_model 函数
# =============================================================================

def evaluate_model(train_cfg, model_path, scaler_path, device, rank, world_size, key):
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
    
    # --- 核心：加载模型 ---
    try:
        model.load_state_dict(torch.load(model_path, map_location=cfg.DEVICE))
    except FileNotFoundError:
        if rank == 0:
            print(f"错误：模型文件未找到于: {model_path}")
        return None # 评估失败
    except Exception as e:
        if rank == 0:
            print(f"加载模型时出错: {e}")
        return None
        
    model.eval()
    if rank == 0:
        print("模型加载成功。")

    # --- 核心：加载Scaler ---
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        if rank == 0:
             print("Scaler 加载成功。")
    else:
        if rank == 0:
            print(f"警告：Scaler 文件未找到于: {scaler_path}")
            print("如果模型使用了归一化，评估结果将不准确。")
        scaler = None

    # --- 加载数据和图 ---
    try:
        adj_matrix = np.load(cfg.ADJ_MATRIX_PATH)
        test_features = np.load(cfg.TEST_FEATURES_PATH)
    except FileNotFoundError as e:
        if rank == 0:
            print(f"错误：数据文件未找到: {e.filename}。请检查 'urbanev' 文件夹。")
        return None

    distances = adj_matrix[adj_matrix > 0]
    sigma = np.std(distances)

    # 应用高斯核函数
    adj_matrix = np.exp(-np.square(adj_matrix) / (sigma**2))
    adj_matrix[adj_matrix < 0.1] = 0
    
    original_edge_index, original_edge_weights = get_edge_index(adj_matrix)
    test_dataset = EVChargerDatasetV2(test_features, cfg.HISTORY_LEN, cfg.PRED_LEN, cfg, scaler=scaler)
    
    # --- 使用 DistributedSampler 切分测试集 ---
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, sampler=test_sampler, num_workers=4, pin_memory=True)
    
    all_predictions_list, all_samples_list, all_true_list = [], [], []

    # 只有 rank 0 显示 TQDM
    disable_tqdm = (rank != 0)
    with torch.no_grad(), amp.autocast():
        for tensors in tqdm(test_dataloader, desc=f"Evaluating (Rank {rank})", disable=disable_tqdm, ncols=100):
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

            all_predictions_list.append(median_prediction.cpu().numpy())
            all_samples_list.append(stacked_samples.cpu().numpy())

    # --- 收集所有进程 (GPU) 的结果 ---

    if not all_predictions_list:
        # 如果某个 rank 没有数据 (例如数据集大小不能被 world_size 整除)
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

    # --- 关键：DDP 收集操作 ---
    dist.gather_object(local_predictions, gathered_preds if rank == 0 else None, dst=0)
    dist.gather_object(local_samples, gathered_samples if rank == 0 else None, dst=0)
    dist.gather_object(local_true, gathered_true if rank == 0 else None, dst=0)

    # --- 只在 Rank 0 上进行指标计算和打印 ---
    if rank == 0:
        print("\n所有进程评估完成，正在 Rank 0 上汇总和计算指标...")
        
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
            print("警告：未执行反归一化。")
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

        # 确保结果目录存在
        os.makedirs('./results', exist_ok=True)
        np.save(f'./results/pred_{cfg.RUN_ID}_{key}.npy', all_predictions)
        np.save(f'./results/samples_{cfg.RUN_ID}_{key}.npy', all_samples)
        np.save(f'./results/true_{cfg.RUN_ID}_{key}.npy', y_true_original) # 额外保存真实值
        print(f"预测结果已保存到 ./results/pred_{cfg.RUN_ID}_{key}.npy")

        # 3. 加载基线模型进行 DM 检验
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

        # 4. 计算最终指标
        final_metrics = calculate_metrics(y_true_original, all_predictions, all_samples, cfg.DEVICE)
        
        if perform_significance:
            errors_model = np.abs(y_true_original.flatten() - all_predictions.flatten())
            errors_baseline = np.abs(y_true_original.flatten() - all_baseline_preds.flatten())
            dm_statistic, p_value = dm_test(errors_baseline, errors_model)
            final_metrics['dm_stat'] = dm_statistic
            final_metrics['p_value'] = p_value
            
        return final_metrics
    else:
        # 其他进程 (rank != 0) 不返回任何东西
        return None

# =============================================================================
# 3. 新的 main 函数，用于驱动评估
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="时空扩散模型 V2 评估脚本")
    parser.add_argument("--run_id", type=str, required=True, 
                        help="训练时生成的唯一 Run ID (例如: 20251026_111500)")
    args = parser.parse_args()

    # --- DDP 初始化 ---
    try:
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        device_id = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(device_id)
        is_ddp = True
        if rank == 0:
            print(f"DDP 环境初始化成功。Rank: {rank}/{world_size}, Device: {device_id}")
    except (KeyError, ValueError):
        # 回退到单机模式 (以防万一)
        rank = 0
        device_id = 0
        world_size = 1
        is_ddp = False
        print("警告：未检测到 DDP 环境变量。在单机模式下运行。")
        print("请使用 'torchrun' 来启动此脚本以进行分布式评估。")


    run_id = args.run_id
    train_cfg = ConfigV2()
    train_cfg.RUN_ID = run_id
    
    # --- 构建路径 ---
    base_path = "./weights/"
    scaler_path = os.path.join(base_path, f"scaler_v2_{run_id}.pkl")
    
    models_to_evaluate = [
        {
            "name": "2nd Best (Loss)",
            "key": "second_best",
            "path": os.path.join(base_path, f"st_diffusion_model_v2_{run_id}_second_best.pth")
        },
        {
            "name": "Best (MAE)",
            "key": "mae_best",
            "path": os.path.join(base_path, f"st_diffusion_model_v2_{run_id}_mae_best.pth")
        },
        # {
        #     "name": "Best (Loss)",
        #     "key": "best",
        #     "path": os.path.join(base_path, f"st_diffusion_model_v2_{run_id}_best.pth")
        # },
        {
            "name": "2nd Best (MAE)",
            "key": "mae_second_best",
            "path": os.path.join(base_path, f"st_diffusion_model_v2_{run_id}_mae_second_best.pth")
        },
        

    ]

    # 只有 Rank 0 检查 Scaler
    if rank == 0 and not os.path.exists(scaler_path):
        print("="*80)
        print(f"严重错误：Scaler 文件未找到于: {scaler_path}")
        print("评估无法继续，因为反归一化是必需的。")
        print("="*80)
        # 通知其他进程退出
        if is_ddp: dist.barrier()
        return

    # --- 依次评估所有模型 ---
    for model_info in models_to_evaluate:
        if rank == 0:
            print("\n" + "="*80)
            print(f"开始评估: {model_info['name']} ({os.path.basename(model_info['path'])})")
            print("="*80)
            if not os.path.exists(model_info['path']):
                print(f"警告：模型文件未找到: {model_info['path']}")
                print("跳过此模型评估。")
                metrics = None # 发送一个 None 信号
            else:
                metrics = "START" # 发送一个 "开始" 信号
        
        if is_ddp:
            # 广播信号，确保所有进程都知道是否要评估
            signal = [metrics] if rank == 0 else [None]
            dist.broadcast_object_list(signal, src=0)
            
            if signal[0] is None:
                if rank != 0: print(f"Rank {rank} 跳过评估 {model_info['key']} (文件未找到)。")
                dist.barrier()
                continue
            
            # 同步所有进程
            dist.barrier()

        # --- 所有进程并行执行评估 ---
        metrics_result = evaluate_model(
            train_cfg=train_cfg,
            model_path=model_info['path'],
            scaler_path=scaler_path,
            device=f"cuda:{device_id}",
            rank=rank,
            world_size=world_size,
            key=model_info['key'] 
        )

        # --- Rank 0 打印结果 ---
        if rank == 0:
            if metrics_result:
                print(f"\n===== [最终评估结果] {model_info['name']} =====")
                print_metrics(metrics_result)
                print(f"==========================================")
            else:
                # 这种情况可能是 evaluate_model 内部出错
                print(f"模型 {model_info['name']} 评估失败 (未返回指标)。")
        
        if is_ddp:
            dist.barrier() # 确保所有进程在进入下一个循环前都已完成

    if rank == 0:
        print("\n" + "="*80)
        print("所有评估任务已完成。")
        print("="*80)

    # --- 清理DDP ---
    if is_ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
      
    main()
