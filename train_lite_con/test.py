import warnings
warnings.filterwarnings("ignore")
import sys
import os
import argparse
import torch
import torch.distributed as dist
from datetime import datetime

# --- 1. 路径设置 (确保能导入根目录模块) ---
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path) # train_sigma/
project_root = os.path.dirname(current_dir)      # 根目录

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# --- 2. 导入模块 ---
# 尝试包内导入或直接导入
try:
    from train_lite_con.config import ConfigV2
    from train_lite_con.metrics import print_metrics
    from train_lite_con.evaluation import evaluate_model
except ImportError:
    from config import ConfigV2
    from metrics import print_metrics
    from evaluation import evaluate_model

def main():
    # --- 3. 参数解析 ---
    parser = argparse.ArgumentParser(description="单独测试已训练好的时空扩散模型")
    parser.add_argument("--run_id", type=str, required=True, 
                        help="训练时生成的唯一 Run ID (例如: 20251028_100000)")
    args = parser.parse_args()
    
    target_run_id = args.run_id

    # --- 4. DDP 初始化 (评估代码依赖 DDP 环境) ---
    try:
        # 如果使用 torchrun 启动，环境变量会自动设置
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        device_id = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(device_id)
    except (KeyError, ValueError):
        # 兼容单机调试模式 (非 DDP)
        print("警告: 未检测到 DDP 环境，尝试以单机模式运行...")
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group("nccl")
        rank = 0
        device_id = 0
        world_size = 1
        torch.cuda.set_device(device_id)

    # --- 5. 配置加载 ---
    cfg = ConfigV2()
    cfg.RUN_ID = target_run_id
    
    if rank == 0:
        print("="*60)
        print(f"开始独立测试流程 | RUN ID: {target_run_id}")
        print("="*60)

    
    # Scaler 路径
    scaler_y_path = cfg.SCALER_Y_SAVE_PATH_TEMPLATE.format(run_id=target_run_id)
    scaler_mm_path = cfg.SCALER_MM_SAVE_PATH_TEMPLATE.format(run_id=target_run_id)
    scaler_z_path = cfg.SCALER_Z_SAVE_PATH_TEMPLATE.format(run_id=target_run_id)

    # 检查 Scaler 是否存在 (必须)
    if rank == 0:
        if not (os.path.exists(scaler_y_path) and os.path.exists(scaler_mm_path) and os.path.exists(scaler_z_path)):
            print(f"错误: 找不到对应的 Scaler 文件。\n请检查路径:\n{scaler_y_path}\n{scaler_mm_path}\n{scaler_z_path}")
            dist.destroy_process_group()
            return

    # 定义要测试的模型列表
    models_to_test = [
        {
            "name": "2nd Best Val Loss Model",
            "key": "second_best_val",
            "path": cfg.MODEL_SAVE_PATH_TEMPLATE.format(run_id=target_run_id, rank="second_best")
        },
        {
            "name": "Best Val MAE Model",
            "key": "best",
            "path": cfg.MODEL_SAVE_PATH_TEMPLATE.format(run_id=target_run_id, rank="mae_best")
        },
        {
            "name": "2nd Best Val MAE Model",
            "key": "second_best",
            "path": cfg.MODEL_SAVE_PATH_TEMPLATE.format(run_id=target_run_id, rank="mae_second_best")
        },
        {
            "name": "Best Val Loss Model",
            "key": "best_val",
            "path": cfg.MODEL_SAVE_PATH_TEMPLATE.format(run_id=target_run_id, rank="best")
        },
        
    ]

    # --- 7. 循环评估 ---
    # 使用 broadcast 确保所有进程同步模型列表 (虽然这里是硬编码的，但为了 DDP 规范)
    dist.barrier()

    for model_info in models_to_test:
        model_path = model_info["path"]
        model_name = model_info["name"]
        model_key = model_info["key"]

        # 检查文件是否存在
        file_exists = os.path.exists(model_path)
        
        if file_exists:
            if rank == 0:
                print(f"\n>>> 正在评估: {model_name}")
                print(f"    路径: {model_path}")
            
            # 调用现有的评估函数
            metrics = evaluate_model(
                train_cfg=cfg,
                model_path=model_path,
                scaler_y_path=scaler_y_path,
                scaler_mm_path=scaler_mm_path,
                scaler_z_path=scaler_z_path,
                device=f"cuda:{device_id}",
                rank=rank,
                world_size=world_size,
                key=model_key+'_post_test'
            )

            if rank == 0 and metrics:
                print(f"--- {model_name} 结果 ---")
                print_metrics(metrics)
        else:
            if rank == 0:
                print(f"\n[跳过] 未找到模型文件: {model_name} ({model_path})")
        
        # 确保所有进程同步进入下一个模型
        dist.barrier()

    if rank == 0:
        print("\n" + "="*60)
        print("所有指定模型的测试已完成。")
        print("="*60)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
