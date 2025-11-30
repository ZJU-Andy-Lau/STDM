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

def get_edge_index(adj_matrix):
    edge_indices = []
    edge_weights = []
    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            if adj_matrix[i, j] != 0:
                edge_indices.append((i, j))
                edge_weights.append(adj_matrix[i, j])

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_weights = torch.tensor(edge_weights, dtype=torch.float)
    return edge_index, edge_weights

def calc_layer_lengths(L_in, depth, kernel_size=3, stride=2, padding=1, dilation=1):
    lengths = [L_in]
    for i in range(depth):
        L_prev = lengths[-1]
        L_out = math.floor((L_prev + 2*padding - dilation*(kernel_size - 1) - 1) / stride + 1)
        lengths.append(L_out)
    return lengths

def batch_time_edge_index(edge_index, edge_weights, num_nodes, batch_size, time_steps, device):
    if time_steps <= 0: return torch.empty(2, 0, dtype=torch.long, device=device)
    num_total_graphs = batch_size * time_steps
    edge_index_list = [edge_index + i * num_nodes for i in range(num_total_graphs)]
    batched_edge_index = torch.cat(edge_index_list, dim=1).to(device)
    batched_edge_weights = edge_weights.repeat(num_total_graphs).to(device)
    return batched_edge_index, batched_edge_weights

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)