import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def create_sliding_windows(data, history_len, pred_len):
    samples = []
    total_len = len(data)
    for i in range(total_len - history_len - pred_len + 1):
        history = data[i : i + history_len]
        future = data[i + history_len : i + history_len + pred_len]
        samples.append((history, future))
    return samples

class EVChargerDatasetV2(Dataset):
    def __init__(self, features, history_len, pred_len, cfg, scaler_y=None,scaler_mm=None,scaler_z=None):
        self.cfg = cfg

        minmax_features = features[:, :, cfg.HISTORY_FEATURES-4:cfg.HISTORY_FEATURES+5].copy()
        zscore_features = features[:, :, cfg.HISTORY_FEATURES+5:].copy()
        

        dynamic_features = features[:, :, :cfg.HISTORY_FEATURES].copy()
        static_features = features[0, :, cfg.HISTORY_FEATURES:].copy()

        target_col_original = dynamic_features[:, :, 0]

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
        self.samples = create_sliding_windows(dynamic_features, history_len, pred_len)

    def _initialize_scaler_y(self, data):
        if self.cfg.NORMALIZATION_TYPE == "minmax":
            scaler = MinMaxScaler(feature_range=(-1, 1))
        elif self.cfg.NORMALIZATION_TYPE == "zscore":
            scaler = StandardScaler()
        else: return None
        scaler.fit(data.reshape(-1, 1))
        return scaler
    
    def _initialize_scaler_mm(self, data):
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(data.reshape(-1, data.shape[-1]))
        return scaler

    def _initialize_scaler_z(self, data):
        scaler = StandardScaler()
        scaler.fit(data.reshape(-1, data.shape[-1]))
        return scaler

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        history, future = self.samples[idx]
        history_c = torch.tensor(history, dtype=torch.float)
        future_x0 = torch.tensor(future[:, :, :self.cfg.TARGET_FEAT_DIM], dtype=torch.float)
        known_start_idx = self.cfg.TARGET_FEAT_DIM + (self.cfg.DYNAMIC_FEAT_DIM - self.cfg.FUTURE_KNOWN_FEAT_DIM)
        future_known_c = torch.tensor(future[:, :, known_start_idx : self.cfg.HISTORY_FEATURES], dtype=torch.float)
        return history_c, self.static_features, future_x0, future_known_c, idx


    def get_scaler(self):
        return self.scaler_y, self.scaler_mm, self.scaler_z