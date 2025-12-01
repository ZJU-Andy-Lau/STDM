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
    
    # [显存优化核心] 限制最大通道数，防止深层特征爆炸
    # 结合 DEPTH=4，原逻辑会达到 1024，现在限制在 256
    MAX_CHANNELS = 1024
    
    # --- 集成训练参数 ---
    ENSEMBLE_K = 8          
    
    # 损失函数权重配置
    MEAN_MSE_LAMBDA = 10.0       
    INDIVIDUAL_L1_LAMBDA = 1.0   
    REPULSION_LAMBDA = 0.5
    
    # 训练参数
    EPOCHS = 100
    BATCH_SIZE = 8 # 优化后建议尝试开大，如 8 或 16
    LEARNING_RATE = 1e-4
    ACCUMULATION_STEPS = 2

    WARMUP_EPOCHS = 5      
    COOLDOWN_EPOCHS = 50    
    CYCLE_EPOCHS = 10    
    
    MODEL_SAVE_PATH_TEMPLATE = "./weights/st_diffusion_model_v2_{run_id}_{rank}.pth"
    SCALER_Y_SAVE_PATH_TEMPLATE = "./weights/scaler_y_v2_{run_id}.pkl"
    SCALER_MM_SAVE_PATH_TEMPLATE = "./weights/scaler_mm_v2_{run_id}.pkl"
    SCALER_Z_SAVE_PATH_TEMPLATE = "./weights/scaler_z_v2_{run_id}.pkl"

    EVAL_ON_VAL = True               
    EVAL_ON_VAL_EPOCH = 5            
    EVAL_ON_VAL_BATCHES = 48         
    EVAL_ON_VAL_SAMPLES = 5          
    EVAL_ON_VAL_STEPS = 20           
    SAMPLING_ETA = 0.0               
    EVAL_SEED = 42 

    TRAIN_FEATURES_PATH = './urbanev/features_train_wea_poi.npy'
    VAL_FEATURES_PATH = './urbanev/features_valid_wea_poi.npy'
    TEST_FEATURES_PATH = './urbanev/features_test_wea_poi.npy'
    ADJ_MATRIX_PATH = './urbanev/dis.npy'

class EvalConfig(ConfigV2):
    BATCH_SIZE = 8
    NUM_SAMPLES = 20
    SAMPLING_STEPS = 50
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"