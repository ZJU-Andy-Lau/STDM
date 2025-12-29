import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib import pyplot as plt

def create_sliding_windows(data, y_raw, history_len, pred_len):
    samples = []
    total_len = len(data)
    for i in range(total_len - history_len - pred_len + 1):
        history = data[i : i + history_len].copy()
        future = data[i + history_len : i + history_len + pred_len].copy()
        future[:, :, 0] = y_raw[i + history_len : i + history_len + pred_len]
        samples.append((i, history, future))
    return samples

class EVChargerDatasetV2(Dataset):
    """
    在原版基础上新增：
    - self.samples 存储 (start_idx, history, future)
    - 可选加载 DID beta / DID policy 资源，构造 mu_future
    - 输出:
        history_c       : (Lh, N, HISTORY_FEATURES)
        static_features : (N, STATIC_FEATURES)
        target_e0       : (Lf, N, TARGET_FEAT_DIM)   = future_x0 - mu_future
        mu_future       : (Lf, N, TARGET_FEAT_DIM)
        future_known_c  : (Lf, N, FUTURE_KNOWN_FEAT_DIM)
        idx             : int
        start_idx       : int  (新增)
    若未提供 DID 资源，则 mu_future=0，target_e0=future_x0
    """
    def __init__(self, features, history_len, pred_len, cfg,
                scaler_y=None, scaler_e=None, scaler_mm=None, scaler_z=None,
                did_policy_8am_path=None,
                did_policy_12am_path=None,
                did_beta_8am_path=None,
                did_beta_12am_path=None,
                enable_did=True):
        self.cfg = cfg
        self.history_len = int(history_len)
        self.pred_len = int(pred_len)
        self.enable_did = bool(enable_did)
        self.did_ready = False
        self.policy = None     # dict: keys -> (T, N)
        self.beta_8 = None     
        self.beta_12 = None    

        minmax_features = features[:, :, cfg.HISTORY_FEATURES-4:cfg.HISTORY_FEATURES+5].copy()
        zscore_features = features[:, :, cfg.HISTORY_FEATURES+5:].copy()
        
        dynamic_features = features[:, :, :cfg.HISTORY_FEATURES].copy()
        static_features = features[0, :, cfg.HISTORY_FEATURES:].copy()

        target_col_original = dynamic_features[:, :, 0]
        y_raw = target_col_original.copy()
        if scaler_y is None:
            self.scaler_y = self._initialize_scaler_y(target_col_original)
        else:
            self.scaler_y = scaler_y

        if scaler_mm is None:
            self.scaler_mm = self._initialize_scaler_mm(minmax_features)
        else:
            self.scaler_mm = scaler_mm
        
        if scaler_z is None:
            self.scaler_z = self._initialize_scaler_z(zscore_features)
        else:
            self.scaler_z = scaler_z

        if self.cfg.NORMALIZATION_TYPE != "none":
            target_col_reshaped = target_col_original.reshape(-1, 1)
            normalized_target = self.scaler_y.transform(target_col_reshaped)
            dynamic_features[:, :, 0] = normalized_target.reshape(target_col_original.shape)

        mm_norm = self.scaler_mm.transform(
            minmax_features.reshape(-1, minmax_features.shape[-1])
        ).reshape(minmax_features.shape)

        z_norm = self.scaler_z.transform(
            zscore_features.reshape(-1, zscore_features.shape[-1])
        ).reshape(zscore_features.shape)

        dynamic_features[:, :, cfg.HISTORY_FEATURES-4:] = mm_norm[:, :, :4]

        static_features[:, :5] = mm_norm[0, :, -5:]
        static_features[:, 5:] = z_norm[0, :, :]
        self.static_features = torch.tensor(static_features, dtype=torch.float)

        self.samples = create_sliding_windows(dynamic_features, y_raw, history_len, pred_len)
        
        if scaler_e is None:
            self.scaler_e = self._initialize_scaler_e()
        else:
            self.scaler_e = scaler_e
        # ============================================================
        # DID 资源加载（可选）
        # 约定：
        #  did_beta_npz_path: 保存 beta 曲线（分钟网格）+ 预先插值后的 12 步也可
        #  did_policy_npz_path: 保存每个时间点、每个节点的政策强度张量（已对齐到 features 的时间轴）
        # ============================================================

        if self.enable_did and did_policy_8am_path and did_policy_12am_path and did_beta_8am_path and did_beta_12am_path:
            self._load_did_resources(did_policy_8am_path, did_policy_12am_path, did_beta_8am_path, did_beta_12am_path)
            self.did_ready = True

    # def _fit_scaler_e_from_train_samples(self):
    #     sum_ = 0.0
    #     sumsq_ = 0.0
    #     cnt_ = 0

    #     for (start_idx, _, future) in self.samples:
    #         # y_future_raw: (L, N, 1)
    #         y_future_raw = future[:, :, :self.cfg.TARGET_FEAT_DIM].astype(np.float64)

    #         # 获取 mu 用于计算残差，忽略 components
    #         mu_future_raw, _ = self._build_mu_future(start_idx)
    #         mu_future_raw = mu_future_raw.astype(np.float64)

    #         e_raw = y_future_raw - mu_future_raw  # (L, N, 1)
    #         x = e_raw.reshape(-1)                 # flatten

    #         sum_ += x.sum()
    #         sumsq_ += (x * x).sum()
    #         cnt_ += x.size

    #     mean = sum_ / cnt_
    #     var = max(sumsq_ / cnt_ - mean * mean, 1e-12)
    #     std = np.sqrt(var)

    #     scaler = StandardScaler(with_mean=True, with_std=True)
    #     # 手动填充必要字段，使 transform / inverse_transform 可用
    #     scaler.mean_ = np.array([mean], dtype=np.float64)
    #     scaler.var_ = np.array([var], dtype=np.float64)
    #     scaler.scale_ = np.array([std], dtype=np.float64)
    #     scaler.n_features_in_ = 1

    #     return scaler
    
    def _fit_scaler_e_from_train_samples(self):
        """
        使用训练样本中的 residual e = y - mu
        拟合 MinMaxScaler，用于对 e 进行归一化 / 反归一化

        注意：
        - e 与 mu 必须处于同一尺度（通常是原始 occupancy）
        - 该 scaler 只用于 residual，不影响 y / mu 的物理含义
        """

        sum_min = np.inf
        sum_max = -np.inf
        cnt_ = 0

        all_e = []

        for (start_idx, _, future) in self.samples:
            # future: (L, N, 1)
            y_future_raw = future[:, :, :self.cfg.TARGET_FEAT_DIM]

            # mu_future: (L, N, 1)
            mu_future_raw,_ = self._build_mu_future(start_idx)

            # residual
            e_raw = y_future_raw - mu_future_raw
            x = e_raw.reshape(-1)

            all_e.append(x)

            # 安全性检查
            if not np.all(np.isfinite(x)):
                raise ValueError(
                    "[scaler_e] 检测到 NaN / Inf，请检查 mu 或 y 是否异常"
                )

            sum_min = min(sum_min, x.min())
            sum_max = max(sum_max, x.max())
            cnt_ += x.size
        
        all_e = np.concatenate(all_e)
        print(f"e min:{all_e.min()} \t e max:{all_e.max()} \t e mean:{all_e.mean()} \t e std:{all_e.std()}")
        plt.hist(all_e,1000)
        plt.savefig('./results/e_raw_hist.png')

        if cnt_ == 0:
            raise RuntimeError("[scaler_e] 未能收集到任何残差样本")

        # 防止退化为常数
        if abs(sum_max - sum_min) < 1e-8:
            sum_max = sum_min + 1e-6

        # 构造 MinMaxScaler
        scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
        scale = 2.0 / (sum_max - sum_min)
        min_ = -1.0 - sum_min * scale

        scaler.scale_ = np.array([scale], dtype=np.float64)
        scaler.min_ = np.array([min_], dtype=np.float64)

        scaler.data_min_ = np.array([sum_min], dtype=np.float64)
        scaler.data_max_ = np.array([sum_max], dtype=np.float64)
        scaler.data_range_ = np.array([sum_max - sum_min], dtype=np.float64)

        scaler.n_features_in_ = 1
        scaler.n_samples_seen_ = cnt_

        return scaler


    def _initialize_scaler_y(self, data):
        if self.cfg.NORMALIZATION_TYPE == "minmax":
            scaler = MinMaxScaler(feature_range=(-1, 1))
        elif self.cfg.NORMALIZATION_TYPE == "zscore":
            scaler = StandardScaler()
        else: return None
        scaler.fit(data.reshape(-1, 1))
        return scaler
    
    def _initialize_scaler_e(self):
        scaler=self._fit_scaler_e_from_train_samples()
        return scaler
    
    def _initialize_scaler_mm(self, data):
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(data.reshape(-1, data.shape[-1]))
        return scaler

    def _initialize_scaler_z(self, data):
        scaler = StandardScaler()
        scaler.fit(data.reshape(-1, data.shape[-1]))
        return scaler

    def _load_did_resources(
        self,
        did_policy_8am_path,
        did_policy_12am_path,
        did_beta_8am_path,
        did_beta_12am_path,
    ):
        pol8 = np.load(did_policy_8am_path, allow_pickle=True)
        pol12 = np.load(did_policy_12am_path, allow_pickle=True)

        required8 = ["D8", "PSM_IN8", "delta_p8", "expo_ctrl8", "expo_tc8"]
        required12 = ["D12", "PSM_IN12", "delta_p12", "expo_ctrl12", "expo_tc12"]

        policy = {}

        for k in required8:
            if k not in pol8:
                raise KeyError(f"[DID] did_policy_8am 缺少键: {k}")
            policy[k] = pol8[k].astype(np.float32)

        for k in required12:
            if k not in pol12:
                raise KeyError(f"[DID] did_policy_12am 缺少键: {k}")
            policy[k] = pol12[k].astype(np.float32)

        self.policy = policy

        def load_one_beta(beta_path_8am, beta_path_12am):
            bnpz_8am = np.load(beta_path_8am, allow_pickle=True)
            bnpz_12am = np.load(beta_path_12am, allow_pickle=True)

            for k in ["r_grid", "beta_own", "beta_amp", "beta_exp", "beta_tx"]:
                if k not in bnpz_8am:
                    raise KeyError(f"[DID] {beta_path_8am} 缺少键: {k}")
                if k not in bnpz_12am:
                    raise KeyError(f"[DID] {beta_path_12am} 缺少键: {k}")

            r_grid_8am = bnpz_8am["r_grid"].astype(np.float32)       # (R,)
            beta_own_8am = bnpz_8am["beta_own"].astype(np.float32)   # (R,)
            beta_amp_8am = bnpz_8am["beta_amp"].astype(np.float32)   # (R,)
            beta_exp_8am = bnpz_8am["beta_exp"].astype(np.float32)   # (R,)
            beta_tx_8am  = bnpz_8am["beta_tx"].astype(np.float32)    # (R,)

            r_grid_12am = bnpz_12am["r_grid"].astype(np.float32)       # (R,)
            beta_own_12am = bnpz_12am["beta_own"].astype(np.float32)   # (R,)
            beta_amp_12am = bnpz_12am["beta_amp"].astype(np.float32)   # (R,)
            beta_exp_12am = bnpz_12am["beta_exp"].astype(np.float32)   # (R,)
            beta_tx_12am  = bnpz_12am["beta_tx"].astype(np.float32)    # (R,)

            def build_hourly_beta_from_5min(r_grid, beta_own, beta_amp, beta_exp, beta_tx, hour_bins):
                beta_dict = {}
                for h in hour_bins:
                    left  = h * 60
                    right = (h + 1) * 60
                    mask = (r_grid >= left) & (r_grid < right)

                    if not mask.any():
                        raise ValueError(
                            f"[DID] hour={h} 对应的分钟区间 [{left},{right}) 内没有数据"
                        )

                    beta_dict[h] = np.array([
                        beta_own[mask].mean(),
                        beta_amp[mask].mean(),
                        beta_exp[mask].mean(),
                        beta_tx[mask].mean(),
                    ], dtype=np.float32)

                return beta_dict

            self.beta_8 = build_hourly_beta_from_5min(r_grid_8am, beta_own_8am, beta_amp_8am, beta_exp_8am, beta_tx_8am, hour_bins=[-3,-2, -1, 0, 1]
            )

            self.beta_12 = build_hourly_beta_from_5min(r_grid_12am, beta_own_12am, beta_amp_12am, beta_exp_12am, beta_tx_12am, hour_bins=[0, 1])
        
        load_one_beta(did_beta_8am_path, did_beta_12am_path)

    def _build_mu_future(self, start_idx):
        """
        Returns:
            mu: (Lf, N, 1) - 总均值，用于Target
            mu_comps: (Lf, N, 5) - 分量[own, amp, ec, tc, total]，用于特征
        """
        Lf = self.pred_len
        N = self.cfg.NUM_NODES
        mu = np.zeros((Lf, N, self.cfg.TARGET_FEAT_DIM), dtype=np.float32)
        mu_comps = np.zeros((Lf, N, 5), dtype=np.float32)

        if not (self.enable_did and self.did_ready):
            return mu, mu_comps

        d0 = min((start_idx + self.history_len+self.pred_len//2)//24, self.policy["D8"].shape[0]-1) 
        
        t0 = start_idx + self.history_len
        future_idx = np.arange(t0, t0 + Lf)

        hour_of_day = future_idx % 24

        rel8  = hour_of_day - 8
        rel12 = hour_of_day - 12

        D8   = self.policy["D8"][d0]
        D12  = self.policy["D12"][d0]
        PSM8  = self.policy["PSM_IN8"][d0]
        PSM12 = self.policy["PSM_IN12"][d0]
        dp8   = self.policy["delta_p8"][d0]
        dp12  = self.policy["delta_p12"][d0]

        ec8   = self.policy["expo_ctrl8"][d0]
        ec12  = self.policy["expo_ctrl12"][d0]

        tc8   = self.policy["expo_tc8"][d0]
        tc12  = self.policy["expo_tc12"][d0]

        for k in range(Lf):
            r8 = rel8[k]
            if r8 in self.beta_8:
                b = self.beta_8[r8]
                
                term_own = b[0] * D8 * PSM8
                term_amp = b[1] * D8 * dp8 * PSM8
                term_ec  = b[2] * ec8 * PSM8
                term_tc  = b[3] * tc8 * PSM8
                
                mu_comps[k, :, 0] += term_own
                mu_comps[k, :, 1] += term_amp
                mu_comps[k, :, 2] += term_ec
                mu_comps[k, :, 3] += term_tc
                mu_comps[k, :, 4] += (term_own + term_amp + term_ec + term_tc)

            r12 = rel12[k]
            if r12 in self.beta_12:
                b = self.beta_12[r12]
                
                term_own = b[0] * D12 * PSM12
                term_amp = b[1] * D12 * dp12 * PSM12
                term_ec  = b[2] * ec12 * PSM12
                term_tc  = b[3] * tc12 * PSM12
                
                mu_comps[k, :, 0] += term_own
                mu_comps[k, :, 1] += term_amp
                mu_comps[k, :, 2] += term_ec
                mu_comps[k, :, 3] += term_tc
                mu_comps[k, :, 4] += (term_own + term_amp + term_ec + term_tc)
        
        mu[:, :, 0] = mu_comps[:, :, 4]
        return mu, mu_comps

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        start_idx, history, future = self.samples[idx]

        history_c = torch.tensor(history, dtype=torch.float)

        future_x0 = torch.tensor(future[:, :, :self.cfg.TARGET_FEAT_DIM], dtype=torch.float) 

        # 计算切片索引，需排除新增的DID维度，使用原始的13维来定位
        orig_future_known_dim = self.cfg.FUTURE_KNOWN_FEAT_DIM - self.cfg.DID_FEAT_DIM
        known_start_idx = self.cfg.TARGET_FEAT_DIM + (self.cfg.DYNAMIC_FEAT_DIM - orig_future_known_dim)
        
        future_known_c = torch.tensor(
            future[:, :, known_start_idx: self.cfg.HISTORY_FEATURES],
            dtype=torch.float
        )

        mu_future_np, mu_comps_np = self._build_mu_future(start_idx)
        mu_future = torch.tensor(mu_future_np, dtype=torch.float)
        
        mu_comps = torch.tensor(mu_comps_np, dtype=torch.float)
        mu_comps = mu_comps / (self.scaler_e.scale_[0] + 1e-8) #归一化did特征

        e_raw = future_x0 - mu_future
        e_raw_np = e_raw.numpy().reshape(-1, 1)

        e_norm_np = self.scaler_e.transform(e_raw_np)

        target_e0 = torch.tensor(
            e_norm_np.reshape(e_raw.shape),
            dtype=torch.float
        )

        future_known_c = torch.cat([future_known_c, mu_comps], dim=-1)

        return history_c, self.static_features, target_e0, mu_future, future_known_c, idx, start_idx, future_x0


    def get_scaler(self):
        return self.scaler_y, self.scaler_e, self.scaler_mm, self.scaler_z