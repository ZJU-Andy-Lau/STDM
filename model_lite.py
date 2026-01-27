import torch
import torch.nn as nn
import math
from tqdm import tqdm
import torch.utils.checkpoint as checkpoint

def get_cosine_schedule_buffers(timesteps, s=0.008):
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

# --- 基础模块 (Dense Ops) ---

class DenseSpatialLayer(nn.Module):
    """
    密集图计算层：替代原来的 GCN_GAT_Layer (稀疏)。
    支持 GCN (矩阵乘法) 和 GAT (Masked Self-Attention)。
    """
    def __init__(self, in_channels, out_channels, heads=4, dropout=0.1):
        super(DenseSpatialLayer, self).__init__()
        
        # GCN 分支: Linear + MatMul
        self.gcn_lin = nn.Linear(in_channels, out_channels)
        
        # GAT 分支: Masked Multi-Head Attention
        # 为了适配 MHA，我们先将输入投影到 out_channels，这样 embed_dim = out_channels
        self.gat_lin = nn.Linear(in_channels, out_channels)
        self.gat_attn = nn.MultiheadAttention(embed_dim=out_channels, num_heads=heads, batch_first=True, dropout=dropout)
        
        # 融合层
        # 输入维度是 GCN输出(out_channels) + GAT输出(out_channels)
        fusion_dim = out_channels * 2 
        self.fusion = nn.Linear(fusion_dim, out_channels)
        
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ELU()

    def forward(self, x, adj, attn_mask):
        """
        x: (Batch * Time, Nodes, In_Channels)
        adj: (Nodes, Nodes) - 密集邻接矩阵 (用于 GCN)
        attn_mask: (Nodes, Nodes) - 密集 Mask 矩阵 (用于 GAT), 非邻居设为 -inf, 邻居设为 0
        """
        # --- GCN 分支 ---
        # 公式: ReLU(A * (XW))
        x_gcn = self.gcn_lin(x) # (B, N, C_out)
        x_gcn = torch.matmul(adj, x_gcn) # 广播乘法: (N,N) @ (B,N,C) -> (B,N,C)
        
        # --- GAT 分支 ---
        # 使用 Masked Self-Attention 模拟 GAT
        x_gat_in = self.gat_lin(x)
        # need_weights=False 节省显存，不返回 Attention Map
        x_gat, _ = self.gat_attn(x_gat_in, x_gat_in, x_gat_in, attn_mask=attn_mask, need_weights=False)
        
        # --- 融合 ---
        x_cat = torch.cat([x_gcn, x_gat], dim=-1)
        out = self.fusion(x_cat)
        
        return self.act(self.dropout(out))

class SinusoidalPosEmb(nn.Module):
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
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        return x.permute(0, 2, 1)

class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.ConvTranspose1d(channels, channels, kernel_size=3, stride=2, padding=1)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        return x.permute(0, 2, 1)

class FiLMLayer(nn.Module):
    def __init__(self, channels, context_dim):
        super().__init__()
        self.layer = nn.Linear(context_dim, channels * 2)
        
    def forward(self, x, context):
        """
        x: (B, N, L, C)
        context: (B*N, context_dim)
        """
        gamma_beta = self.layer(context) # (B*N, 2C)
        
        # 必须根据 x 的维度重塑 gamma_beta，否则无法广播
        if x.dim() == 4:
            B, N, L, C = x.shape
            # 将 (B*N, 2C) -> (B, N, 1, 2C)
            # 这里的 1 是为了对齐 L 维度进行广播
            gamma_beta = gamma_beta.view(B, N, 1, -1)
        else:
            # 兼容其他可能的维度情况
            gamma_beta = gamma_beta.unsqueeze(1)

        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return gamma * x + beta

class SpatioTemporalTransformerBlock(nn.Module):
    def __init__(self, channels, context_dim, num_heads):
        super().__init__()
        assert channels % num_heads == 0, f"Channels ({channels}) must be divisible by num_heads ({num_heads})"
        self.channels = channels
        self.num_heads = num_heads
        
        # 密集计算层
        self.spatial_layer = DenseSpatialLayer(channels, channels, heads=num_heads, dropout=0.1)
        
        self.spatial_norm = nn.LayerNorm(channels)
        self.temporal_attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.temporal_norm = nn.LayerNorm(channels)
        self.use_film = context_dim > 0
        if self.use_film:
            # self.film_layer = FiLMLayer(channels, context_dim)
            self.context_bias = nn.Linear(context_dim, channels)
        else:
            # self.film_layer = None
            self.context_bias = None
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels)
        )
        self.ffn_norm = nn.LayerNorm(channels)

    def forward(self, x, context, adj):
        """
        x: (B, N, L, C)
        adj: (N, N) 密集矩阵
        """
        
        # 预计算 Attention Mask
        attn_mask = torch.zeros_like(adj)
        attn_mask[adj == 0] = float('-inf')

        # 定义内部前向传播函数，用于 checkpoint
        def _inner_forward(x, context, adj, attn_mask):
            x_res = x
            if self.use_film and context is not None:
                B, N, L, C = x.shape
                bias = self.context_bias(context).view(B, N, 1, C)
                x = x + bias
            x_spatial_res = x
            x_norm_spatial = self.spatial_norm(x)
            
            # --- Spatial Layer (密集计算) ---
            # Input: (B, N, L, C)
            # Need: (B*L, N, C) for Spatial Layer
            B, N, L, C = x_norm_spatial.shape
            
            # 将 Batch 和 Time 维度合并
            x_flat = x_norm_spatial.permute(0, 2, 1, 3).reshape(B * L, N, C)
            
            x_spatial = self.spatial_layer(x_flat, adj, attn_mask)
            
            # Restore: (B*L, N, C) -> (B, L, N, C) -> (B, N, L, C)
            x_spatial = x_spatial.reshape(B, L, N, C).permute(0, 2, 1, 3)
            x = x_spatial_res + x_spatial

            # --- Temporal Layer ---
            x_temporal_res = x
            x_norm = self.temporal_norm(x)
            # Input to MHA: (Batch*Nodes, Seq, Channels)
            x_norm_temp = x_norm.reshape(B * N, L, C)
            x_attn, _ = self.temporal_attn(x_norm_temp, x_norm_temp, x_norm_temp)
            x_attn = x_attn.reshape(B, N, L, C)
            x = x_temporal_res + x_attn

            # --- FFN ---
            x_ffn_res = x
            x_norm = self.ffn_norm(x)
            x_ffn = self.ffn(x_norm)
            x = x_ffn_res + x_ffn
            
            return x + x_res

        # [恢复] Gradient Checkpointing 逻辑
        if self.training and x.requires_grad:
            # 使用 use_reentrant=False 是推荐做法，可以减少一些潜在的副作用
            return checkpoint.checkpoint(_inner_forward, x, context, adj, attn_mask, use_reentrant=False)
        else:
            return _inner_forward(x, context, adj, attn_mask)

class ContextEncoder(nn.Module):
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
        self.future_encoder = nn.GRU(
            input_size=future_known_dim,
            hidden_size=model_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.05
        )
        self.dropout = nn.Dropout(0.05)
        fusion_input_dim = time_dim + model_dim + model_dim + model_dim
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, context_dim),
            nn.Mish(),
            nn.Dropout(0.05),
            nn.Linear(context_dim, context_dim),
            nn.Dropout(0.05)
        )

    def forward(self, k, history_c, static_c, future_known_c, adj):
        t_emb = self.time_mlp(k)
        
        # 1. 投影
        history_c_proj = self.history_projection(history_c) # 形状目前是 (BN, L, C)
        
        # Block 需要 (B, N, L, C) 4维输入
        
        # 从 adj 获取节点数 N (adj shape: [N, N])
        N = adj.shape[0]
        BN, L, C = history_c_proj.shape
        B = BN // N
        
        # 将 3D (BN, L, C) 重塑为 4D (B, N, L, C)
        history_c_reshaped = history_c_proj.reshape(B, N, L, C)
        
        # 传入 Block 进行处理, context 为 None
        hist_enc_out = self.history_encoder(history_c_reshaped, None, adj)
        
        # 我们只需要最后一个时间步的特征，并需要将其还原为 (BN, C) 以便后续融合
        hist_emb = hist_enc_out[:, :, -1, :].reshape(BN, C)

        static_emb = self.static_encoder(static_c)
        _, future_hidden_state = self.future_encoder(future_known_c)
        future_emb = future_hidden_state[-1]
        future_emb = self.dropout(future_emb)
        num_nodes = hist_emb.size(0) // t_emb.size(0)
        t_emb_expanded = t_emb.repeat_interleave(num_nodes, dim=0)
        combined = torch.cat([t_emb_expanded, hist_emb, static_emb, future_emb], dim=-1)
        context = self.fusion_mlp(combined)
        return context

class UGnetV2(nn.Module):
    def __init__(self, in_features, out_features, context_dim, model_dim, num_heads, depth=2, max_channels=256):
        super().__init__()
        self.init_conv = nn.Conv1d(in_features, model_dim, kernel_size=1)
        self.down_blocks, self.up_blocks = nn.ModuleList(), nn.ModuleList()
        self.downsamples, self.upsamples = nn.ModuleList(), nn.ModuleList()
        self.up_projections = nn.ModuleList()
        
        dims = [model_dim]
        current_dim = model_dim
        
        # 显存优化：限制最大通道数
        for _ in range(depth):
            self.down_blocks.append(SpatioTemporalTransformerBlock(current_dim, context_dim, num_heads))
            
            # 限制下一层的通道数增长
            next_dim = min(current_dim * 2, max_channels)
            
            self.downsamples.append(Downsample(current_dim, next_dim))
            dims.append(next_dim)
            current_dim = next_dim
            
        self.bottleneck = SpatioTemporalTransformerBlock(dims[-1], context_dim, num_heads)
        
        for i in range(depth):
            self.upsamples.append(Upsample(dims[-1]))
            in_ch = dims[-1] + dims[-2]
            out_ch = dims[-2]
            self.up_projections.append(nn.Conv1d(in_ch, out_ch, kernel_size=1))
            self.up_blocks.append(SpatioTemporalTransformerBlock(out_ch, context_dim, num_heads))
            dims.pop()
        
        self.final_conv = nn.Conv1d(model_dim, out_features, kernel_size=1)
        nn.init.zeros_(self.final_conv.bias)
        nn.init.normal_(self.final_conv.weight, mean=0.0, std=0.02)

    def forward(self, x, context, adj):
        batch_size, num_nodes, seq_len, _ = x.shape
        # x input: (B, N, L, C)
        
        # Conv1d expects (Batch, Channels, Length). We treat (B*N) as batch.
        x = x.reshape(batch_size * num_nodes, seq_len, -1).permute(0, 2, 1) # (BN, C, L)
        x = self.init_conv(x).permute(0, 2, 1) # (BN, L, C)
        
        # Restore to (B, N, L, C) for ST-Block which needs 4D input for DenseSpatialLayer
        x = x.reshape(batch_size, num_nodes, seq_len, -1)
        
        skip_connections = []
        for block, downsampler in zip(self.down_blocks, self.downsamples):
            x = block(x, context, adj)
            skip_connections.append(x)
            
            # Downsample: (B, N, L, C) -> (BN, L, C) -> Conv1d -> (BN, L/2, C_out) -> (B, N, L/2, C_out)
            B, N, L, C = x.shape
            x_flat = x.reshape(B * N, L, C)
            x_down = downsampler(x_flat)
            x = x_down.reshape(B, N, -1, x_down.shape[-1])

        x = self.bottleneck(x, context, adj)
        skip_connections.reverse()
        
        for i, (upsampler, proj, block) in enumerate(zip(self.upsamples, self.up_projections, self.up_blocks)):
            # Upsample
            B, N, L, C = x.shape
            x_flat = x.reshape(B * N, L, C)
            x_up = upsampler(x_flat)
            x = x_up.reshape(B, N, -1, x_up.shape[-1])
            
            skip = skip_connections[i]
            # Interpolate if needed (safety check for odd lengths)
            if x.shape[2] != skip.shape[2]:
                 x_flat = x.reshape(B*N, x.shape[2], -1).permute(0, 2, 1)
                 x_flat = nn.functional.interpolate(x_flat, size=skip.shape[2], mode='linear', align_corners=False)
                 x = x_flat.permute(0, 2, 1).reshape(B, N, skip.shape[2], -1)

            x = torch.cat([x, skip], dim=-1)
            
            # Projection
            B, N, L, C_concat = x.shape
            x_flat = x.reshape(B * N, L, C_concat).permute(0, 2, 1)
            x_proj = proj(x_flat).permute(0, 2, 1)
            x = x_proj.reshape(B, N, L, -1)
            
            x = block(x, context, adj)
            
        B, N, L, C = x.shape
        x_flat = x.reshape(B * N, L, C).permute(0, 2, 1)
        x_out = self.final_conv(x_flat)
        x = x_out.permute(0, 2, 1).reshape(batch_size, num_nodes, seq_len, -1)
        return x

class SpatioTemporalDiffusionModelV2(nn.Module):
    def __init__(self, in_features, out_features, history_features, static_features, future_known_features, model_dim, num_heads, T=1000, depth=2, max_channels=256):
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
            depth=depth,
            max_channels=max_channels 
        )
        
        betas = torch.linspace(1e-4, 0.02, T)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

    def forward(self, x_k, k, history_c, static_c, future_known_c, adj):
        """
        adj: (N, N) 密集邻接矩阵
        """
        batch_size, num_nodes, _, _ = x_k.shape
        history_c_flat = history_c.reshape(batch_size * num_nodes, history_c.shape[2], -1)
        future_known_c_flat = future_known_c.reshape(batch_size * num_nodes, future_known_c.shape[2], -1)
        
        if static_c.dim() == 3 and static_c.shape[0] == batch_size:
            static_c_flat = static_c.reshape(batch_size * num_nodes, -1)
        elif static_c.dim() == 2:
            # (N, F) -> (B, N, F) -> (BN, F)
            static_c_flat = static_c.repeat(batch_size, 1)
        else:
            raise ValueError(f"Unexpected shape for static_c: {static_c.shape}.")

        context = self.context_encoder(k, history_c_flat, static_c_flat, future_known_c_flat, adj)
        predicted_noise = self.denoise_net(x_k, context, adj)
        return predicted_noise

    @torch.no_grad()
    def ddim_sample(self, history_c, static_c, future_known_c, adj, shape, sampling_steps=50, eta=0.0):
        device = self.betas.device
        b, n, l, _ = shape
        
        time_steps = torch.linspace(-int(1), self.T - 1, steps=sampling_steps + 1).round().to(torch.long)
        time_steps = list(reversed(time_steps.tolist()))
        
        x_k = torch.randn(shape, device=device)

        for i in tqdm(range(sampling_steps), desc="DDIM Sampling", leave=False, ncols=100, disable=True):
            t_current = time_steps[i]
            t_prev = time_steps[i+1]
            k_tensor = torch.full((b,), t_current, device=device, dtype=torch.long)
            
            # 传入 adj
            predicted_noise = self.forward(
                x_k, k_tensor, history_c, static_c, future_known_c, adj
            )

            alpha_cumprod_t = self.alphas_cumprod[t_current] if t_current >= 0 else torch.tensor(1.0, device=device)
            alpha_cumprod_t_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)
            
            pred_x0 = (x_k - torch.sqrt(1. - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
            
            if eta > 0 and t_prev >= 0:
                variance = (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev)
                sigma_t = eta * torch.sqrt(variance)
            else:
                sigma_t = torch.tensor(0.0, device=device)

            sigma_sq = sigma_t ** 2
            max_sigma_sq = (1. - alpha_cumprod_t_prev).clamp(min=0.0)
            sigma_sq = torch.clamp(sigma_sq, max=max_sigma_sq)
            
            coeff = torch.sqrt((1. - alpha_cumprod_t_prev - sigma_sq).clamp(min=0.0))
            pred_dir_xt = coeff * predicted_noise
            x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir_xt

            if eta > 0:
                x_prev = x_prev + sigma_t * torch.randn_like(x_k)
                
            x_k = x_prev

        return x_k