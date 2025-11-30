import torch
import numpy as np
import math
import random
import csv
import os

class CsvLogger:
    """
    一个用于 DDP 训练的模块化 CSV 日志记录器。
    它只会在 rank 0 进程上创建和写入文件。
    """
    def __init__(self, log_dir, run_id, rank, headers):
        """
        初始化记录器。

        参数:
            log_dir (str): 日志文件存放的目录 (例如: './results')
            run_id (str): 当前运行的 ID，用作文件名 (例如: '20251028_100000')
            rank (int): 当前 DDP 进程的 rank。
            headers (list[str]): CSV 文件的表头 (例如: ['epoch', 'train_loss', 'val_loss'])
        """
        self.log_dir = log_dir
        self.run_id = run_id
        self.rank = rank
        self.headers = headers
        self.log_file_path = None

        # 只有 rank 0 进程执行文件操作
        if self.rank == 0:
            try:
                # 确保日志目录存在
                os.makedirs(self.log_dir, exist_ok=True)
                
                # 定义日志文件路径
                self.log_file_path = os.path.join(self.log_dir, f"{self.run_id}.csv")
                
                # 创建文件并写入表头
                with open(self.log_file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(self.headers)
                print(f"[Rank 0] CsvLogger initialized. Logging to {self.log_file_path}")
                    
            except Exception as e:
                print(f"[Rank 0 Logger Error] Failed to initialize logger: {e}")
                self.log_file_path = None # 初始化失败，禁用日志

    def log_epoch(self, epoch_data):
        """
        记录一个 epoch 的数据。

        参数:
            epoch_data (dict): 包含要记录数据的字典。
                               键 (key) 必须与初始化时的 headers 对应。
                               例如: {'epoch': 1, 'train_loss': 0.5, ...}
        """
        # 只有 rank 0 且日志文件已成功初始化时才写入
        if self.rank != 0 or self.log_file_path is None:
            return

        try:
            # 按表头顺序准备要写入的数据行
            # 使用 .get(h, '') 来优雅地处理缺失值：
            # 如果字典中没有某个键 (例如 'avg_val_mae')，
            # 它会写入一个空字符串，而不是-
            row = [epoch_data.get(h, '') for h in self.headers]
            
            # 以追加模式 ('a') 打开文件并写入数据
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
    """
    根据Conv1d参数计算每一层的输出长度
    
    参数：
        L_in : int      # 初始序列长度
        depth : int     # 网络层数（下采样层数）
        kernel_size : int
        stride : int
        padding : int
        dilation : int
        
    返回：
        lengths : list[int]  # 每一层的输出长度（包含输入层）
    """
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
    """设置随机种子以确保可复现性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)