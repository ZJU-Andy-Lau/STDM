import torch
import numpy as np

from model_sigma import SpatioTemporalDiffusionModelV2

# ==========================
# 加载模型权重路径
# ==========================
model_path = "./weights/st_diffusion_model_v2_20251122_163129_mae_best.pth"   # ← 修改这里

# ==========================
# 模型结构必须和训练时一致
# ==========================
B = 1
N = 275
L = 12

history_dim = 13
static_dim = 7
future_known_dim = 13

model_dim = 64   # ← 如果训练时不是 64，请告诉我
num_heads = 4
depth = 4        # ← 根据你的信息已经确认
T = 1000

# ----------------------------
# 构造假输入（严格符合模型要求）
# ----------------------------
history_c = torch.randn(B, N, L, history_dim)
static_c = torch.randn(B, N, static_dim)
future_known_c = torch.randn(B, N, L, future_known_dim)

x_k = torch.randn(B, N, L, 1)
k = torch.randint(0, T, (B,), dtype=torch.long)

# ----------------------------
# 构造 edge_data_list (长度必须 = depth + 1 = 5)
# ----------------------------
edge_index = torch.zeros((2, 2), dtype=torch.long)
edge_weight = torch.ones((2,), dtype=torch.float32)

# 5 层：4 个 down_blocks + 1 个 bottleneck
edge_data_list = [(edge_index, edge_weight)] * (depth + 1)

# ----------------------------
# 加载模型
# ----------------------------
model = SpatioTemporalDiffusionModelV2(
    in_features=1,
    out_features=1,
    history_features=history_dim,
    static_features=static_dim,
    future_known_features=future_known_dim,
    model_dim=model_dim,
    num_heads=num_heads,
    T=T,
    depth=depth
)

state = torch.load(model_path, map_location="cpu")
missing, unexpected = model.load_state_dict(state, strict=False)
print("Missing keys:", missing)
print("Unexpected keys:", unexpected)

model.eval()

# ----------------------------
# 前向运行（不计算梯度）
# ----------------------------
with torch.no_grad():
    predicted_noise, predicted_logvar = model(
        x_k,
        k,
        history_c,
        static_c,
        future_known_c,
        edge_data_list,
        edge_data_list,
    )

# ----------------------------
# 打印 logvar 统计
# ----------------------------
logvar_np = predicted_logvar.numpy().flatten()

print("\n============== Predicted logvar Diagnostics ==============")
print("min logvar:", logvar_np.min())
print("max logvar:", logvar_np.max())
print("mean logvar:", logvar_np.mean())
print("std logvar:", logvar_np.std())
print("===========================================================\n")

if logvar_np.mean() < -10:
    print("❗【严重塌缩】logvar 平均值 < -10 → σθ 极小 → Under-dispersion 很严重")
elif logvar_np.mean() < -5:
    print("⚠️【较小】logvar < -5 → σθ 偏小 → 存在 under-dispersion")
else:
    print("✔ logvar 正常，没有塌缩")
