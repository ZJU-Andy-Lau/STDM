import torch
import numpy as np
import os
import joblib
import sys
from tqdm import tqdm
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda import amp
import torch.distributed as dist

# 获取当前脚本的绝对路径 (e.g., /path/to/project/train_sigma/evaluation.py)
current_file_path = os.path.abspath(__file__)
# 获取当前脚本所在的目录 (e.g., /path/to/project/train_sigma)
current_dir = os.path.dirname(current_file_path)
# 获取项目根目录 (e.g., /path/to/project)
project_root = os.path.dirname(current_dir)

# 将项目根目录添加到 sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 确保 train_sigma 包本身可以被解析
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from train_sigma.config import EvalConfig
    from train_sigma.utils import calc_layer_lengths, batch_time_edge_index, get_edge_index
    from train_sigma.dataset import EVChargerDatasetV2
    from train_sigma.metrics import calculate_metrics, dm_test, print_metrics
except ImportError:
    from config import EvalConfig
    from utils import calc_layer_lengths, batch_time_edge_index, get_edge_index
    from dataset import EVChargerDatasetV2
    from metrics import calculate_metrics, dm_test, print_metrics

# 这里注意 model_sigma 需要从上级目录导入，运行 main.py 时路径会自动处理
from model_sigma import SpatioTemporalDiffusionModelV2

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