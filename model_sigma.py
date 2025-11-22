import torch
import torch.nn as nn
import math
from tqdm import tqdm
from torch_geometric.utils import softmax
from torch_geometric.nn import GCNConv, GATConv


def get_cosine_schedule_buffers(timesteps, s=0.008):
    """
    根据余弦调度计算 alphas_cumprod (alpha_bar_t)。
    f(t) = cos^2( ((t/T + s) / (1 + s)) * pi/2 )
    alpha_bar_t = f(t) / f(0)
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps)
    f_t = torch.cos(((t / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod_cos = f_t / f_t[0]
    alphas_cumprod_prev = alphas_cumprod_cos[:-1]
    alphas_cumprod = alphas_cumprod_cos[1:]

    betas = 1. - (alphas_cumprod / alphas_cumprod_prev)
    betas = torch.clamp(betas, 1e-8, 0.999)

    alphas = 1. - betas
    alphas_cumprod_final = torch.cumprod(alphas, axis=0)
    
    return betas, alphas, alphas_cumprod_final

# --- 基础模块 (Foundation Modules) ---

class GCN_GAT_Layer(nn.Module):
    """融合版图卷积层：GCN + GAT"""
    def __init__(self, in_channels, out_channels, heads=4, dropout=0.1, concat=True):
        super(GCN_GAT_Layer, self).__init__()
        self.gcn = GCNConv(in_channels, out_channels)
        self.gat = GATConv(in_channels, out_channels, heads=heads, dropout=dropout, concat=concat)

        fusion_dim = out_channels + (out_channels * heads if concat else out_channels)
        self.fusion = nn.Linear(fusion_dim, out_channels)

        self.dropout = nn.Dropout(dropout)
        self.act = nn.ELU()

    def forward(self, x, edge_index,edge_weight=None):
        """
        x: (N, F) 或 (B*N*L, F)
        edge_index: (2, E)
        """
        x_gcn = self.gcn(x, edge_index, edge_weight=edge_weight)
        x_gat = self.gat(x, edge_index)

        x_cat = torch.cat([x_gcn, x_gat], dim=-1)
        out = self.fusion(x_cat)
        return self.act(self.dropout(out))
        # return self.act(self.dropout(x_gcn))


class SinusoidalPosEmb(nn.Module):
    """用于扩散时间步长 k 的正弦位置嵌入。"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Downsample(nn.Module):
    """下采样模块，同时处理通道数变化。"""
    # --- 核心改动：接收输入和输出通道数 ---
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        return x.permute(0, 2, 1)

class Upsample(nn.Module):
    """上采样模块，用于恢复时间序列长度。"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.ConvTranspose1d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        return x.permute(0, 2, 1)

# --- V2.5 核心改进模块 ---

class FiLMLayer(nn.Module):
    """特征级线性调制 (Feature-wise Linear Modulation) 层。"""
    def __init__(self, channels, context_dim):
        super().__init__()
        self.layer = nn.Linear(context_dim, channels * 2)

    def forward(self, x, context):
        gamma_beta = self.layer(context).unsqueeze(1).chunk(2, dim=-1)
        gamma, beta = gamma_beta[0], gamma_beta[1]
        return gamma * x + beta

class SpatioTemporalTransformerBlock(nn.Module):
    """时空Transformer块 (空间注意力已优化, 并使用Pre-LN)。"""
    def __init__(self, channels, context_dim, num_heads):
        super().__init__()
        assert channels % num_heads == 0, f"Channels ({channels}) must be divisible by num_heads ({num_heads})"
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        # --- 空间注意力 (融合 GCN + GAT) ---
        self.spatial_layer = GCN_GAT_Layer(channels, channels, heads=num_heads, dropout=0.1)
        self.spatial_norm = nn.LayerNorm(channels)  # Pre-LN

        
        # 时间注意力组件
        self.temporal_attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.temporal_norm = nn.LayerNorm(channels)

        # 条件注入
        self.use_film = context_dim > 0
        if self.use_film:
            self.film_layer = FiLMLayer(channels, context_dim)
        else:
            self.film_layer = None

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels)
        )
        self.ffn_norm = nn.LayerNorm(channels)

    def forward(self, x, context, edge_data):
        # x 形状: (B*N, L, C)
        # edge_index 形状: (2, E * B * L), 已经过批处理

        # print(f"Debug(5): x shape: {x.shape}, edge_index shape: {edge_index.shape}")
        # if context is not None:
            # print(f"Debug(6): context shape: {context.shape}")
        # x shape: torch.Size([550, 12, 64])
        # context shape: torch.Size([550, 256])     定义了context_dim = model_dim * 4

        x_res = x
        
        if self.use_film and context is not None:
            x = self.film_layer(x, context)
        
        # print(f"Debug(7): After FiLM, x shape: {x.shape}")
        # After FiLM, x shape: torch.Size([550, 12, 64])    [550, 6, 128]  [550, 3, 256]...


        # --- 空间注意力 (融合 GCN + GAT) ---
        x_spatial_res = x
        x_norm_spatial = self.spatial_norm(x)  # Pre-LN
        # print(f"Debug(8): x_norm shape before flatten: {x_norm.shape}")
        # x_norm shape before flatten: torch.Size([550, 12, 64])  [550, 6, 128]  [550, 3, 256]...

        BN, L, C = x_norm_spatial.shape
        B=BN // 275
        N=275
        x_reshaped = x_norm_spatial.reshape(B,N, L, C)
        x_permuted = x_reshaped.permute(0, 2, 1, 3)
        #  (B, L, N, C) -> (B*L*N, C) 以便GNN处理
        x_flat = x_permuted.reshape(-1, C)

        # print(f"Debug(9): x_flat shape: {x_flat.shape}")
        # x_flat shape: torch.Size([6600, 64])  [3300, 128]  [1650, 256]...

        edge_index, edge_weight = edge_data
        x_spatial = self.spatial_layer(x_flat, edge_index, edge_weight=edge_weight)  # (B*N*L, C)
        # print(f"Debug(10): x_spatial shape after GCN_GAT_Layer: {x_spatial.shape}")

        x_spatial_reshaped = x_spatial.reshape(B, L, N, C)
        x_spatial_permuted = x_spatial_reshaped.permute(0, 2, 1, 3)
        x_spatial = x_spatial_permuted.reshape(BN, L, C)

        x = x_spatial_res + x_spatial  # 残差连接

        
        # --- 时间注意力 (Pre-LN) ---
        x_temporal_res = x
        x_norm = self.temporal_norm(x)
        x_attn, _ = self.temporal_attn(x_norm, x_norm, x_norm)
        x = x_temporal_res + x_attn

        # --- 前馈网络 (Pre-LN) ---
        x_ffn_res = x
        x_norm = self.ffn_norm(x)
        x_ffn = self.ffn(x_norm)
        x = x_ffn_res + x_ffn

        return x + x_res

class ContextEncoder(nn.Module):
    """统一的上下文编码器 (已增加未来已知特征编码)。"""
    def __init__(self, time_dim, history_dim, static_dim, future_known_dim, model_dim, context_dim, num_heads):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4), nn.Mish(),
            nn.Linear(time_dim * 4, time_dim)
        )
        
        self.history_projection = nn.Linear(history_dim, model_dim)
        self.history_encoder = SpatioTemporalTransformerBlock(model_dim, context_dim=0, num_heads=num_heads)
        
        self.static_encoder = nn.Linear(static_dim, model_dim)
        
        # <<< 新增：使用GRU编码未来已知特征序列 >>>
        self.future_encoder = nn.GRU(
            input_size=future_known_dim,
            hidden_size=model_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.05
        )
        self.dropout = nn.Dropout(0.05)
        # <<< 修改：融合MLP的输入维度增加 >>>
        fusion_input_dim = time_dim + model_dim + model_dim + model_dim # t_emb, hist_emb, static_emb, future_emb
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, context_dim),
            nn.Mish(),
            nn.Dropout(0.05),
            nn.Linear(context_dim, context_dim),
            nn.Dropout(0.05)
        )

    def forward(self, k, history_c, static_c, future_known_c, edge_data):
        # print(f"Debug(3): history_c shape: {history_c.shape}, static_c shape: {static_c.shape}, future_known_c shape: {future_known_c.shape}")
        # history_c shape: torch.Size([550, 12, 8]),  B*N, L, F_hist
        # static_c shape: torch.Size([550, 4]), 
        # future_known_c shape: torch.Size([550, 12, 5])

        t_emb = self.time_mlp(k)
        
        history_c_proj = self.history_projection(history_c)
        hist_emb = self.history_encoder(history_c_proj, None, edge_data)[:, -1, :]
        
        static_emb = self.static_encoder(static_c)

        # print(f"Debug(4): history_c_proj shape: {history_c_proj.shape}, hist_emb shape: {hist_emb.shape}, static_emb shape: {static_emb.shape}")
        # history_c_proj shape: torch.Size([550, 12, 64]), 
        # hist_emb shape: torch.Size([550, 64]), 
        # static_emb shape: torch.Size([550, 64])

        # <<< 新增：编码未来已知特征 >>>
        # B*N, L, F_known -> (num_layers, B*N, H_dim)
        _, future_hidden_state = self.future_encoder(future_known_c)
        # 取最后一层的隐藏状态作为总结
        future_emb = future_hidden_state[-1]

        future_emb = self.dropout(future_emb)
        
        # 对齐t_emb维度
        num_nodes = hist_emb.size(0) // t_emb.size(0)
        t_emb_expanded = t_emb.repeat_interleave(num_nodes, dim=0)
        
        # <<< 修改：将 future_emb 加入拼接 >>>
        combined = torch.cat([t_emb_expanded, hist_emb, static_emb, future_emb], dim=-1)
        context = self.fusion_mlp(combined)
        return context

class UGnetV2(nn.Module):
    """V2版本的时空U-Net去噪网络。"""
    def __init__(self, in_features, out_features, context_dim, model_dim, num_heads, depth=2):
        super().__init__()
        self.init_conv = nn.Conv1d(in_features, model_dim, kernel_size=1)
        
        self.down_blocks, self.up_blocks = nn.ModuleList(), nn.ModuleList()
        self.downsamples, self.upsamples = nn.ModuleList(), nn.ModuleList()
        self.up_projections = nn.ModuleList()

        dims = [model_dim]
        current_dim = model_dim
        for _ in range(depth):
            self.down_blocks.append(SpatioTemporalTransformerBlock(current_dim, context_dim, num_heads))
            # --- 核心改动：Downsample现在负责增加通道数 ---
            self.downsamples.append(Downsample(current_dim, current_dim * 2))
            dims.append(current_dim * 2)
            current_dim *= 2
        
        self.bottleneck = SpatioTemporalTransformerBlock(dims[-1], context_dim, num_heads)
        
        for i in range(depth):
            self.upsamples.append(Upsample(dims[-1]))
            
            in_ch = dims[-1] + dims[-2]
            out_ch = dims[-2]
            self.up_projections.append(nn.Conv1d(in_ch, out_ch, kernel_size=1))
            self.up_blocks.append(SpatioTemporalTransformerBlock(out_ch, context_dim, num_heads))
            
            dims.pop()
            
        self.final_conv = nn.Conv1d(model_dim, out_features*2, kernel_size=1)

    def forward(self, x, context, edge_data_list):
        batch_size, num_nodes, seq_len, _ = x.shape
        # print(f"Debug(11): Input x shape: {x.shape}")
        # Input x shape: torch.Size([2, 275, 12, 1])
        
        x = x.reshape(batch_size * num_nodes, seq_len, -1).permute(0, 2, 1)
        # print(f"Debug(12): Reshaped x shape: {x.shape}")
        # Reshaped x shape: torch.Size([550, 1, 12])
        
        x = self.init_conv(x).permute(0, 2, 1)
        # print(f"Debug(13): After init_conv, x shape: {x.shape}")
        # After init_conv, x shape: torch.Size([550, 12, 64])
        
        skip_connections = []
        
        # --- 编码器路径 ---
        for i, (block, downsampler) in enumerate(zip(self.down_blocks, self.downsamples)):
            # 为当前层选择正确的 edge_index
            current_edge_data = edge_data_list[i]
            x = block(x, context, current_edge_data)
            skip_connections.append(x)
            x = downsampler(x)
            
        # --- 瓶颈层 ---
        # 为瓶颈层选择最深层的 edge_index
        bottleneck_edge_data = edge_data_list[-1]
        x = self.bottleneck(x, context, bottleneck_edge_data)
        # print("Bottleneck output shape:", x.shape)
        
        # --- 解码器路径 ---
        skip_connections.reverse()
        for i, (upsampler, proj, block) in enumerate(zip(self.upsamples, self.up_projections, self.up_blocks)):
            x = upsampler(x)
            skip = skip_connections[i]
            
            if x.shape[1] != skip.shape[1]:
                x = nn.functional.interpolate(x.permute(0, 2, 1), size=skip.shape[1], mode='linear', align_corners=False).permute(0, 2, 1)

            x = torch.cat([x, skip], dim=-1)
            
            x = x.permute(0, 2, 1)
            x = proj(x)
            x = x.permute(0, 2, 1)

            # 为解码器当前层选择正确的 edge_index
            # 解码器从深到浅，所以 edge_index 的索引也要倒过来
            current_edge_data = edge_data_list[len(self.down_blocks) - 1 - i]
            x = block(x, context, current_edge_data)

        x = x.permute(0, 2, 1)
        x = self.final_conv(x)
        x = x.permute(0, 2, 1).reshape(batch_size, num_nodes, seq_len, -1)
        return x

class SpatioTemporalDiffusionModelV2(nn.Module):
    """完整的V2.5版时空扩散模型。"""
    def __init__(self, in_features, out_features, history_features, static_features, future_known_features, model_dim, num_heads, T=1000,depth=2):
        super().__init__()
        self.T = T
        self.out_features = out_features
        
        context_dim = model_dim * 4
        self.context_encoder = ContextEncoder(
            time_dim=model_dim, 
            history_dim=history_features, 
            static_dim=static_features, 
            future_known_dim=future_known_features,
            model_dim=model_dim,
            context_dim=context_dim,
            num_heads=num_heads
        )
        
        self.denoise_net = UGnetV2(
            in_features=in_features, 
            out_features=out_features, 
            context_dim=context_dim, 
            model_dim=model_dim,
            num_heads=num_heads,
            depth=depth
        )
        
        # 线性调度
        betas = torch.linspace(1e-4, 0.02, T)

        # # 二次方调度：在 beta 的平方根上进行线性插值，然后再平方
        # beta_start = 1e-4
        # beta_end = 0.5
        # betas = torch.linspace(beta_start**0.5, beta_end**0.5, T)**2

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)

        # 余弦调度
        # betas, alphas,  alphas_cumprod = get_cosine_schedule_buffers(T)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

    def forward(self, x_k, k, history_c, static_c, future_known_c, history_edge_data, future_edge_data):
        batch_size, num_nodes, _, _ = x_k.shape
        
        history_c_flat = history_c.reshape(batch_size * num_nodes, history_c.shape[2], -1)
        future_known_c_flat = future_known_c.reshape(batch_size * num_nodes, future_known_c.shape[2], -1)
        
        if static_c.dim() == 3 and static_c.shape[0] == batch_size:
            static_c_flat = static_c.reshape(batch_size * num_nodes, -1)
        elif static_c.dim() == 2:
            static_c_flat = static_c.repeat(batch_size, 1)
        else:
            raise ValueError(f"Unexpected shape for static_c: {static_c.shape}. Expected (B, N, F) or (N, F).")

        # context = self.context_encoder(k, history_c_flat, static_c_flat, history_edge_index[0])
        context = self.context_encoder(k, history_c_flat, static_c_flat, future_known_c_flat, history_edge_data[0])
        predicted_out = self.denoise_net(x_k, context, future_edge_data)
        # 按最后一维拆成两半：[..., :out_features] 是噪声，[..., out_features:] 是 logvar
        predicted_noise = predicted_out[..., :self.out_features]
        predicted_logvar = predicted_out[..., self.out_features:]
        return predicted_noise, predicted_logvar

    @torch.no_grad()
    def ddim_sample(self, history_c, static_c, future_known_c, history_edge_data, future_edge_data, shape, sampling_steps=50, eta=0.0):
        device = self.betas.device
        b, n, l, _ = shape
        
        time_steps = torch.linspace(-int(1), self.T - 1, steps=sampling_steps + 1).round().to(torch.long)
        time_steps = list(reversed(time_steps.tolist()))
        
        x_k = torch.randn(shape, device=device)

        for i in tqdm(range(sampling_steps), desc="DDIM Sampling", leave=False, ncols=100,disable=True):
            t_current = time_steps[i]
            t_prev = time_steps[i+1]
            
            k_tensor = torch.full((b,), t_current, device=device, dtype=torch.long)
            
            # <<< 核心修改：将两个edge_index列表传入self.forward >>>
            # predicted_noise = self.forward(x_k, k_tensor, history_c, static_c, history_edge_indices, future_edge_indices)
            predicted_noise, predicted_logvar = self.forward(
                x_k, k_tensor, history_c, static_c, future_known_c, history_edge_data, future_edge_data
            )

            alpha_cumprod_t = self.alphas_cumprod[t_current] if t_current >= 0 else torch.tensor(1.0, device=device)
            alpha_cumprod_t_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)
            
            pred_x0 = (x_k - torch.sqrt(1. - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
            # sigma_t = eta * torch.sqrt((1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev))
            
            min_logvar, max_logvar = -4.0, 0.5   # 可微调
            logvar = torch.clamp(predicted_logvar, min_logvar, max_logvar)
    
            var_scale = 2.0  
            sigma_t = eta * torch.exp(0.5 * logvar) * var_scale

            # 4) 限制 sigma_t^2 不超过 (1 - alpha_cumprod_t_prev)
            max_sigma_sq = (1. - alpha_cumprod_t_prev - 1e-5).clamp(min=1e-5)  # 标量 → 自动广播
            sigma_sq = torch.clamp(sigma_t**2, max=max_sigma_sq)

            # 5) 计算 pred_dir_xt 时，确保 sqrt 里非负
            coeff = torch.sqrt((1. - alpha_cumprod_t_prev - sigma_sq).clamp(min=0.0))
            pred_dir_xt = coeff * predicted_noise

            # 6) 合成 x_{t-1}
            x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir_xt

            if eta > 0:
                x_prev = x_prev + torch.sqrt(sigma_sq) * torch.randn_like(x_k)
                
            x_k = x_prev

        return x_k
