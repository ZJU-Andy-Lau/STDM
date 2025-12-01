import torch
import numpy as np
import math
import random
import csv
import os

class CsvLogger:
    def __init__(self, log_dir, run_id, rank, headers):
        self.log_dir = log_dir
        self.run_id = run_id
        self.rank = rank
        self.headers = headers
        self.log_file_path = None

        if self.rank == 0:
            try:
                os.makedirs(self.log_dir, exist_ok=True)
                self.log_file_path = os.path.join(self.log_dir, f"{self.run_id}.csv")
                with open(self.log_file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(self.headers)
                print(f"[Rank 0] CsvLogger initialized. Logging to {self.log_file_path}")
            except Exception as e:
                print(f"[Rank 0 Logger Error] Failed to initialize logger: {e}")
                self.log_file_path = None

    def log_epoch(self, epoch_data):
        if self.rank != 0 or self.log_file_path is None:
            return
        try:
            row = [epoch_data.get(h, '') for h in self.headers]
            with open(self.log_file_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(row)
        except Exception as e:
            print(f"[Rank 0 Logger Error] Failed to log epoch data: {e}")

def calc_layer_lengths(L_in, depth, kernel_size=3, stride=2, padding=1, dilation=1):
    lengths = [L_in]
    for i in range(depth):
        L_prev = lengths[-1]
        L_out = math.floor((L_prev + 2*padding - dilation*(kernel_size - 1) - 1) / stride + 1)
        lengths.append(L_out)
    return lengths

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)