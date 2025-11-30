import torch

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
    
    # --- 集成训练参数 ---
    ENSEMBLE_K = 4          # 对于每个样本，生成 K 个不同的预测
    ENSEMBLE_LAMBDA = 1.0   # 集成损失的权重
    LOGVAR_LAMBDA = 0.1     # 原始 LogVar 辅助损失的权重 (防止 LogVar 分支坍缩)
    
    # 训练参数
    EPOCHS = 100
    BATCH_SIZE = 4 # 注意：这是【单张卡】的batch size。显存仅需支持 1*K
    LEARNING_RATE = 1e-4
    ACCUMULATION_STEPS = 5

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

class EvalConfig(ConfigV2):
    BATCH_SIZE = 8
    NUM_SAMPLES = 20
    SAMPLING_STEPS = 50
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"