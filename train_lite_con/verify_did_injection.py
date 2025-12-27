import torch
import numpy as np
import os
import sys

# 确保路径包含项目根目录，以便导入模块
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from train_lite_con.config import ConfigV2
from train_lite_con.dataset import EVChargerDatasetV2

def verify_did():
    print("开始验证 DID 注入逻辑...")
    
    # 1. 初始化配置和数据路径
    cfg = ConfigV2()
    # 确保这些路径指向真实存在的测试文件或样本文件
    # 建议使用验证集或训练集的一小部分
    features_path = cfg.TRAIN_FEATURES_PATH 
    
    # 检查文件是否存在
    if not os.path.exists(features_path):
        print(f"错误: 找不到特征文件 {features_path}")
        return

    features = np.load(features_path)
    
    # 实例化 Dataset
    # 注意：这里会加载 DID 资源，请确保 config 中路径配置正确
    dataset = EVChargerDatasetV2(
        features, 
        cfg.HISTORY_LEN, 
        cfg.PRED_LEN, 
        cfg,
        did_policy_8am_path=cfg.TRAIN_DID_POLICY_8_PATH,
        did_policy_12am_path=cfg.TRAIN_DID_POLICY_12_PATH,
        did_beta_8am_path=cfg.DID_BETA_8_PATH,
        did_beta_12am_path=cfg.DID_BETA_12_PATH,
        enable_did=True
    )
    
    print(f"Dataset 加载完成，样本数: {len(dataset)}")
    
    if not dataset.did_ready:
        print("错误: DID 资源未正确加载 (did_ready=False)")
        return

    # 计数器
    total_checked = 0
    nonzero_found = 0
    
    # 随机抽取样本或遍历前 N 个样本
    num_samples_to_check = 1000 
    indices = np.random.choice(len(dataset), num_samples_to_check, replace=False)

    for idx in indices:
        # 获取样本
        # future_known_c shape: (Lf, N, 18)
        # 最后的 5 个通道是 [own, amp, ec, tc, total]
        _, _, _, _, future_known_c, _, start_idx = dataset[idx]
        
        # 提取 DID 特征部分
        did_feats = future_known_c[:, :, -5:] # (12, 275, 5)
        
        # 计算该样本对应的时间信息
        t0 = start_idx + cfg.HISTORY_LEN
        future_idx = np.arange(t0, t0 + cfg.PRED_LEN)
        hours = future_idx % 24
        
        # --- 验证 1: 时空位置 (Time Windows) ---
        # 8点事件窗口: [-3, -2, -1, 0, 1] -> 小时 [5, 6, 7, 8, 9]
        # 12点事件窗口: [0, 1] -> 小时 [12, 13]
        
        valid_hours_8 = [5, 6, 7, 8, 9]
        valid_hours_12 = [12, 13]
        valid_hours_all = set(valid_hours_8 + valid_hours_12)
        
        for t in range(cfg.PRED_LEN):
            h = hours[t]
            current_did_slice = did_feats[t] # (N, 5)
            
            # 如果当前小时完全不在任何事件窗口内，所有 DID 特征必须为 0
            if h not in valid_hours_all:
                if torch.any(current_did_slice != 0):
                    print(f"[FAIL] 样本 {idx} 时间步 {t} (小时 {h}) 不在事件窗口内，但发现了非零 DID 特征！")
                    print(f"Max value: {current_did_slice.max()}")
                    return
            
            # 如果在窗口内，检查是否有值 (不一定有值，取决于是否变价日/处理组)
            if torch.any(current_did_slice != 0):
                nonzero_found += 1
                
                # --- 验证 2: Mask 正确性 (简单抽查) ---
                # 随机抽查一个非零节点，反查 policy 确认其是否应该有值
                nonzero_nodes = torch.nonzero(current_did_slice[:, 4], as_tuple=True)[0]
                if len(nonzero_nodes) > 0:
                    node_idx = nonzero_nodes[0].item()
                    
                    # 反查 Policy
                    d0 = min((start_idx + cfg.HISTORY_LEN + cfg.PRED_LEN//2)//24, dataset.policy["D8"].shape[0]-1)
                    
                    # 检查 8点逻辑
                    if h in valid_hours_8:
                        is_treated = dataset.policy["D8"][d0][node_idx]
                        has_expo = dataset.policy["expo_ctrl8"][d0][node_idx] or dataset.policy["expo_tc8"][d0][node_idx]
                        
                        # 自身效应通道有值 => 必须是处理组 (D=1)
                        if current_did_slice[node_idx, 0] != 0 and is_treated == 0:
                            print(f"[FAIL] 样本 {idx} 节点 {node_idx} 有 Own 效应，但 Policy D8=0")
                            return

        total_checked += 1

    print("-" * 30)
    print(f"验证完成！")
    print(f"检查样本数: {total_checked}")
    print(f"发现含 DID 特征的时间步数: {nonzero_found}")
    
    if nonzero_found == 0:
        print("[WARNING] 在检查的样本中未发现任何 DID 特征。请确认是否碰巧全抽到了非变价日？")
    else:
        print("[PASS] 基础验证通过：非事件窗口严格为0，事件窗口内检测到有效注入。")

if __name__ == "__main__":
    verify_did()