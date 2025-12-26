import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def create_sliding_windows(data, y_raw, history_len, pred_len):
    samples = []
    total_len = len(data)
    for i in range(total_len - history_len - pred_len + 1):
        history = data[i : i + history_len]
        future = data[i + history_len : i + history_len + pred_len]
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
        y_raw = target_col_original

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

    def _fit_scaler_e_from_train_samples(self):
        sum_ = 0.0
        sumsq_ = 0.0
        cnt_ = 0

        for (start_idx, _, future) in self.samples:
            # y_future_raw: (L, N, 1)
            y_future_raw = future[:, :, :self.cfg.TARGET_FEAT_DIM].astype(np.float64)

            # mu_future_raw: (L, N, 1)
            mu_future_raw = self._build_mu_future(start_idx).astype(np.float64)

            e_raw = y_future_raw - mu_future_raw  # (L, N, 1)
            x = e_raw.reshape(-1)                 # flatten

            sum_ += x.sum()
            sumsq_ += (x * x).sum()
            cnt_ += x.size

        mean = sum_ / cnt_
        var = max(sumsq_ / cnt_ - mean * mean, 1e-12)
        std = np.sqrt(var)

        scaler = StandardScaler(with_mean=True, with_std=True)
        # 手动填充必要字段，使 transform / inverse_transform 可用
        scaler.mean_ = np.array([mean], dtype=np.float64)
        scaler.var_ = np.array([var], dtype=np.float64)
        scaler.scale_ = np.array([std], dtype=np.float64)
        scaler.n_features_in_ = 1

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

    # ============================================================
    # DID: 资源加载与 beta(分钟)->beta(小时12步)
    # ============================================================
    def _load_did_resources(
        self,
        did_policy_8am_path,
        did_policy_12am_path,
        did_beta_8am_path,
        did_beta_12am_path,
    ):
        pol8 = np.load(did_policy_8am_path, allow_pickle=True)
        pol12 = np.load(did_policy_12am_path, allow_pickle=True)

        # -------------------------
        # 1) policy：合并到统一 key 空间
        # -------------------------
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

        # -------------------------
        # 2) beta：8am / 12pm 分别读取并插值到小时 12 步
        #     你的 beta 文件：r_grid + 4 条 (R,) 曲线
        # -------------------------
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
                """
                将 5 分钟级 DID beta 聚合为小时级 beta
                Parameters
                ----------
                r_grid : np.ndarray
                    shape (R,), 单位：分钟，例如 [-180, -175, ..., 120]
                beta_own / amp / exp / tx : np.ndarray
                    shape (R,)
                hour_bins : list[int]
                    例如 [-2, -1, 0, 1, 2]

                Returns
                -------
                beta_dict : dict
                    {hour: np.array([β_own, β_amp, β_exp, β_tx])}
                """
                beta_dict = {}
                for h in hour_bins:
                    # 该小时对应的分钟区间
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

            # 12 点：[0, +1, +2]
            self.beta_12 = build_hourly_beta_from_5min(r_grid_12am, beta_own_12am, beta_amp_12am, beta_exp_12am, beta_tx_12am, hour_bins=[0, 1])
        
        load_one_beta(did_beta_8am_path, did_beta_12am_path)
        # load_one_beta(did_beta_12am_path, "12am")

    # ============================================================
    # 5) 构造 μ（严格的三重 gate：事件 × PSM × 时间）
    # ============================================================
    def _build_mu_future(self, start_idx):
        """
        构造未来预测窗口的结构均值 μ(t)
        - 支持 8 点抢充（-3h ~ +2h）
        - 支持 12 点降价（0h ~ +2h）
        - beta 已预先按小时离散好，不再插值
        """
        Lf = self.pred_len
        N = self.cfg.NUM_NODES
        mu = np.zeros((Lf, N, self.cfg.TARGET_FEAT_DIM), dtype=np.float32)

        if not (self.enable_did and self.did_ready):
            return mu

        # ========== 1) 当前样本未来窗口在全局时间轴上的切片 ==========
        d0 = min((start_idx + self.history_len+self.pred_len//2)//24, self.policy["D8"].shape[0]-1) 
        
        # ============================
        # 1. 未来时间索引（小时）
        # ============================
        t0 = start_idx + self.history_len
        future_idx = np.arange(t0, t0 + Lf)

        # 每一步对应的小时（0–23）
        hour_of_day = future_idx % 24

        # 相对事件时间（小时）
        rel8  = hour_of_day - 8
        rel12 = hour_of_day - 12

        # policy slices: (Lf, N)
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

        # ============================
        # 3. 构造 mu（核心修改）
        # ============================
        for k in range(Lf):
            # ---------- 8 点事件 ----------
            r8 = rel8[k]
            if r8 in self.beta_8:
                b = self.beta_8[r8]
                mu[k, :, 0] += (
                    b[0] * D8
                    + b[1] * D8 * dp8
                    + b[2] * ec8
                    + b[3] * tc8
                ) * PSM8
            # ---------- 12 点事件 ----------
            r12 = rel12[k]
            if r12 in self.beta_12:   # ★ 只在 [0,2] 内生效
                b = self.beta_12[r12]
                mu[k, :, 0] += (
                    b[0] * D12
                    + b[1] * D12 * dp12
                    + b[2] * ec12
                    + b[3] * tc12
                ) * PSM12
        return mu

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        start_idx, history, future = self.samples[idx]

        history_c = torch.tensor(history, dtype=torch.float)

        future_x0 = torch.tensor(future[:, :, :self.cfg.TARGET_FEAT_DIM], dtype=torch.float) # 没经过归一化的occ

        known_start_idx = self.cfg.TARGET_FEAT_DIM + (self.cfg.DYNAMIC_FEAT_DIM - self.cfg.FUTURE_KNOWN_FEAT_DIM)
        future_known_c = torch.tensor(
            future[:, :, known_start_idx: self.cfg.HISTORY_FEATURES],
            dtype=torch.float
        )

        mu_future_np = self._build_mu_future(start_idx)
        mu_future = torch.tensor(mu_future_np, dtype=torch.float)

        e_raw = future_x0 - mu_future
        e_raw_np = e_raw.numpy().reshape(-1, 1)

        e_norm_np = self.scaler_e.transform(e_raw_np)

        target_e0 = torch.tensor(
            e_norm_np.reshape(e_raw.shape),
            dtype=torch.float
        )

        return history_c, self.static_features, target_e0, mu_future, future_known_c, idx, start_idx


    def get_scaler(self):
        return self.scaler_y, self.scaler_e, self.scaler_mm, self.scaler_z