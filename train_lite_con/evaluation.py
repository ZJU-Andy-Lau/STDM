import warnings
warnings.filterwarnings("ignore")
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
    from train_lite_con.config import EvalConfig
    from train_lite_con.dataset import EVChargerDatasetV2
    from train_lite_con.metrics import calculate_metrics, dm_test, print_metrics
except ImportError:
    from config import EvalConfig
    from dataset import EVChargerDatasetV2
    from metrics import calculate_metrics, dm_test, print_metrics

from model_lite import SpatioTemporalDiffusionModelV2

@torch.no_grad()
def periodic_evaluate_mae(model, loader, scaler_e, adj_tensor, cfg, device):
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
    all_true_list = []
    for (history_c, static_c, target_e0, mu_future, future_known_c, idx, start_idx) in progress_bar:
        tensors = [d.to(device) for d in (history_c, static_c, target_e0, mu_future, future_known_c)]
        history_c, static_c, target_e0, mu_future, future_known_c = tensors
        
        generated_samples = []
        for _ in range(cfg.EVAL_ON_VAL_SAMPLES):
            e_norm_hat = model.ddim_sample(
                history_c=history_c.permute(0, 2, 1, 3), 
                static_c=static_c,
                future_known_c=future_known_c.permute(0, 2, 1, 3),
                adj=adj_tensor, # 传入 adj
                shape=target_e0.permute(0, 2, 1, 3).shape, 
                sampling_steps=cfg.EVAL_ON_VAL_STEPS, 
                eta=cfg.SAMPLING_ETA 
            )
            generated_samples.append(e_norm_hat)
        
        stacked = torch.stack(generated_samples, dim=0)                 # (S,B,N,L,1)
        e_norm_pred = torch.median(stacked, dim=0).values       # (B,N,L,1)

        e_norm_true = target_e0.permute(0,2,1,3)                # (B,N,L,1)
        mu_raw      = mu_future.permute(0,2,1,3)                # (B,N,L,1)

        # === 关键：用 scaler_e 反归一化 e，再加 mu ===
        e_pred = e_norm_pred.detach().cpu().numpy()
        e_true = e_norm_true.detach().cpu().numpy()
        mu = mu_raw.detach().cpu().numpy()

        if scaler_e is not None:
            e_pred = scaler_e.inverse_transform(e_pred.reshape(-1,1)).reshape(e_pred.shape)
            e_true = scaler_e.inverse_transform(e_true.reshape(-1,1)).reshape(e_true.shape)

        y_pred = e_pred + mu
        y_true = e_true + mu
        y_pred = np.clip(y_pred, 0.0, 1.0) 

        all_predictions_list.append(y_pred)
        all_true_list.append(y_true)


    model.train() 
    return np.concatenate(all_predictions_list, axis=0).squeeze(-1).transpose(0, 2, 1), np.concatenate(all_true_list, axis=0).squeeze(-1).transpose(0, 2, 1)

def evaluate_model(train_cfg, model_path, scaler_y_path, scaler_e_path,scaler_mm_path, scaler_z_path, device, rank, world_size,key):
    cfg = EvalConfig()
    cfg.RUN_ID = train_cfg.RUN_ID
    cfg.NORMALIZATION_TYPE = train_cfg.NORMALIZATION_TYPE
    cfg.DEVICE = device

    # 初始化模型时传入 MAX_CHANNELS
    model = SpatioTemporalDiffusionModelV2(
        in_features=cfg.TARGET_FEAT_DIM, out_features=cfg.TARGET_FEAT_DIM,
        history_features=cfg.HISTORY_FEATURES, static_features=cfg.STATIC_FEATURES,
        future_known_features=cfg.FUTURE_KNOWN_FEAT_DIM, model_dim=cfg.MODEL_DIM,
        num_heads=cfg.NUM_HEADS, T=cfg.T, depth=cfg.DEPTH,
        max_channels=cfg.MAX_CHANNELS # 传入
    ).to(cfg.DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=cfg.DEVICE))
    model.eval()
    print("Model loaded successfully.")

    # [修改] 密集矩阵加载
    adj_matrix = np.load(cfg.ADJ_MATRIX_PATH)
    distances = adj_matrix[adj_matrix > 0]
    sigma = np.std(distances)
    adj_matrix = np.exp(-np.square(adj_matrix) / (sigma**2))
    adj_matrix[adj_matrix < 0.1] = 0
    adj_tensor = torch.from_numpy(adj_matrix).float().to(cfg.DEVICE)
    
    test_features = np.load(cfg.TEST_FEATURES_PATH)
    scaler_y = joblib.load(scaler_y_path) if os.path.exists(scaler_y_path) else None  
    scaler_e = joblib.load(scaler_e_path) if os.path.exists(scaler_e_path) else None
    scaler_mm = joblib.load(scaler_mm_path) if os.path.exists(scaler_mm_path) else None
    scaler_z = joblib.load(scaler_z_path) if os.path.exists(scaler_z_path) else None


    did_beta_8am_path = cfg.DID_BETA_8_PATH
    did_beta_12am_path = cfg.DID_BETA_12_PATH
    test_did_policy_8am_path = cfg.TEST_DID_POLICY_8_PATH
    test_did_policy_12am_path = cfg.TEST_DID_POLICY_12_PATH

    test_dataset = EVChargerDatasetV2(test_features, 
                                      cfg.HISTORY_LEN, 
                                      cfg.PRED_LEN, cfg, 
                                      scaler_y=scaler_y, 
                                      scaler_e=scaler_e, 
                                      scaler_mm=scaler_mm, 
                                      scaler_z=scaler_z,
                                      did_policy_8am_path=test_did_policy_8am_path,
                                      did_policy_12am_path=test_did_policy_12am_path,
                                      did_beta_8am_path=did_beta_8am_path,
                                      did_beta_12am_path=did_beta_12am_path,
                                      )
    print(f"Test dataset size: {len(test_dataset)} samples.")
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, sampler=test_sampler)
    
    all_predictions_list, all_samples_list, all_true_list, all_idx_list,all_mu_list = [], [], [], [], []

    disable_tqdm = (dist.is_initialized() and dist.get_rank() != 0)
    with torch.no_grad(), amp.autocast():
        for tensors in tqdm(test_dataloader, desc="Evaluating", disable=disable_tqdm):
            history_c, static_c, target_e0, mu_future, future_known_c, idx, start_idx = tensors
            history_c = history_c.to(cfg.DEVICE)
            static_c = static_c.to(cfg.DEVICE)
            target_e0 = target_e0.to(cfg.DEVICE)
            mu_future = mu_future.to(cfg.DEVICE)
            future_known_c = future_known_c.to(cfg.DEVICE)

            all_true_list.append(target_e0.permute(0, 2, 1, 3).cpu().numpy())
            all_idx_list.append(idx.cpu().numpy()) 
            all_mu_list.append(mu_future.permute(0, 2, 1, 3).cpu().numpy())

            # [修改] 使用 adj_tensor
            generated_samples = []
            for _ in range(cfg.NUM_SAMPLES):
                sample = model.ddim_sample(
                    history_c=history_c.permute(0, 2, 1, 3), static_c=static_c,
                    future_known_c=future_known_c.permute(0, 2, 1, 3),
                    adj=adj_tensor, # 传入
                    shape=target_e0.permute(0, 2, 1, 3).shape, sampling_steps=cfg.SAMPLING_STEPS,
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
        local_mu = np.empty((0, cfg.NUM_NODES, cfg.PRED_LEN, cfg.TARGET_FEAT_DIM), dtype=np.float32)  # (0,N,L,1)
        local_idx = np.empty((0,), dtype=np.int64)
    else:
        local_predictions = np.concatenate(all_predictions_list, axis=0)
        local_samples = np.concatenate(all_samples_list, axis=1)
        local_true = np.concatenate(all_true_list, axis=0)
        local_mu = np.concatenate(all_mu_list, axis=0)
        local_idx = np.concatenate(all_idx_list, axis=0) 

    gathered_preds = [None] * world_size
    gathered_samples = [None] * world_size
    gathered_true = [None] * world_size
    gathered_idx = [None] * world_size
    gathered_mu = [None] * world_size

    dist.gather_object(local_predictions, gathered_preds if rank == 0 else None, dst=0)
    dist.gather_object(local_samples, gathered_samples if rank == 0 else None, dst=0)
    dist.gather_object(local_true, gathered_true if rank == 0 else None, dst=0)
    dist.gather_object(local_mu, gathered_mu if rank == 0 else None, dst=0)
    dist.gather_object(local_idx, gathered_idx if rank == 0 else None, dst=0)
    
    if rank == 0:
        all_predictions_norm = np.concatenate(gathered_preds, axis=0)
        all_samples_norm = np.concatenate(gathered_samples, axis=1)
        all_true_norm = np.concatenate(gathered_true, axis=0)
        all_mu_raw = np.concatenate(gathered_mu, axis=0)
        all_idx = np.concatenate(gathered_idx, axis=0)
        print(f"gather index:{all_idx}")

        order = np.argsort(all_idx)
        print(f"order:{order}")
        all_predictions_norm = all_predictions_norm[order]
        all_true_norm = all_true_norm[order]
        all_samples_norm = all_samples_norm[:, order]
        all_mu_raw = all_mu_raw[order]

        if scaler_e:
            e_pred = scaler_e.inverse_transform(all_predictions_norm.reshape(-1, 1)).reshape(all_predictions_norm.shape)
            e_true = scaler_e.inverse_transform(all_true_norm.reshape(-1, 1)).reshape(all_true_norm.shape)
            e_samp = scaler_e.inverse_transform(all_samples_norm.reshape(-1, 1)).reshape(all_samples_norm.shape)
        else:
            e_pred = all_predictions_norm
            e_true = all_true_norm
            e_samp = all_samples_norm

        y_pred = e_pred + all_mu_raw
        y_true = e_true + all_mu_raw
        y_samp = e_samp + all_mu_raw[None, ...]

        y_pred = np.clip(y_pred, 0.0, 1.0)
        y_true = np.clip(y_true, 0.0, 1.0)

        y_pred = y_pred.squeeze(-1).transpose(0, 2, 1)      # (B, L, N)
        y_true = y_true.squeeze(-1).transpose(0, 2, 1)      # (B, L, N)
        y_samp = y_samp.squeeze(-1).transpose(1, 0, 3, 2)   # (B, S, L, N)

        try:
            all_baseline_preds = np.load("./urbanev/TimeXer_predictions.npy")
            all_baseline_preds = np.concatenate([all_baseline_preds[:, :, -1:], all_baseline_preds], axis=-1)[:, :, :-1]
            all_baseline_preds = all_baseline_preds[:y_pred.shape[0]]
            y_true = y_true[:all_baseline_preds.shape[0]]
            y_pred = y_pred[:all_baseline_preds.shape[0]]
            y_samp = y_samp[:all_baseline_preds.shape[0]]
            print("\n已加载基线模型 (TimeXer) 预测，用于 DM 显著性检验。")
            print(f"all_baseline_preds shape:{all_baseline_preds.shape}")
            perform_significance = True
        except FileNotFoundError:
            print("\n警告：基线预测文件 ./urbanev/TimeXer_predictions.npy 未找到。")
            print("跳过 DM 显著性检验。")
            perform_significance = False
        
        print(f"y_true shape:{y_true.shape}")
        print(f"y_pred shape:{y_pred.shape}")
        print(f"y_samp shape:{y_samp.shape}")
        print(f"all_baseline_shape:{all_baseline_preds.shape}")

        np.save(f'./results/truths_{cfg.RUN_ID}_{key}.npy', y_true)
        np.save(f'./results/pred_{cfg.RUN_ID}_{key}.npy', y_pred)
        np.save(f'./results/samples_{cfg.RUN_ID}_{key}.npy', y_samp)

        final_metrics = calculate_metrics(y_true, y_pred, y_samp, cfg.DEVICE)
        if perform_significance:
            errors_model = np.abs(y_true.flatten() - y_pred.flatten())
            errors_baseline = np.abs(y_true.flatten() - all_baseline_preds.flatten())
            dm_statistic, p_value = dm_test(errors_baseline, errors_model)
            final_metrics['dm_stat'] = dm_statistic
            final_metrics['p_value'] = p_value
            
        return final_metrics
    else:
        return None