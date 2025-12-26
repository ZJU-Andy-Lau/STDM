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

class NanDebugger:
    """
    NaN 自动诊断工具类
    负责：
    1. 检查模型参数健康状况
    2. 诊断 Loss 计算过程中的数值异常
    3. 诊断梯度来源，定位导致 NaN 梯度的具体 Loss 项
    """
    def __init__(self, model, logger=print):
        self.model = model
        self.logger = logger

    def check_model_params(self):
        """检查模型参数是否已经损坏（用于判断是否是上一步更新导致的问题）"""
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                self.logger(f"[CRITICAL] Parameter {name} contains NaN/Inf! The model was corrupted by the previous step.")
                return False
        return True

    def check_tensor(self, tensor, name):
        """检查单个 Tensor 是否包含 NaN/Inf"""
        if torch.isnan(tensor).any():
            self.logger(f"[FAIL] {name} contains NaN")
            return False
        if torch.isinf(tensor).any():
            self.logger(f"[FAIL] {name} contains Inf")
            return False
        return True

    def diagnose_loss_detail(self, pred_x0_grouped, target_x0_expanded, weights):
        """
        详细排查 Loss 计算过程（用于定位计算图中的 NaN 来源）
        """
        self.logger("\n--- Starting Loss Calculation Diagnosis ---")
        
        # 1. 检查输入
        if not self.check_tensor(pred_x0_grouped, "pred_x0_grouped (Model Output)"): return
        if not self.check_tensor(target_x0_expanded, "target_x0_expanded (Ground Truth)"): return
        
        # 2. 检查统计量
        ensemble_var = pred_x0_grouped.var(dim=1, unbiased=True, keepdim=True)
        if not self.check_tensor(ensemble_var, "ensemble_var (raw)"): 
            self.logger(f"Min Var value: {ensemble_var.min().item()}")
            self.logger(f"Max Var value: {ensemble_var.max().item()}")
        
        # 3. 检查 NLL 具体项
        self.logger("[Diagnosis] Checking NLL Terms...")
        safe_var = torch.clamp(ensemble_var, min=1e-4)
        log_term = torch.log(safe_var)
        if not self.check_tensor(log_term, "Log Term (log(var))"):
            self.logger("Possible cause: Variance <= 0 before clamping?")

        sq_error = (pred_x0_grouped.mean(dim=1, keepdim=True) - target_x0_expanded)**2
        div_term = sq_error / (safe_var + 1e-8)
        if not self.check_tensor(div_term, "Division Term (MSE/Var)"):
            self.logger("Possible cause: Variance is too small relative to Squared Error.")

        # 4. 检查 Energy 具体项
        self.logger("[Diagnosis] Checking Energy Score Terms...")
        diff = pred_x0_grouped - target_x0_expanded
        sum_sq = diff.pow(2).sum(dim=-1)
        # 检查是否因为精度问题导致 sum_sq 出现极小负数
        if (sum_sq < 0).any():
             self.logger(f"[WARNING] Negative values found in sum_sq before sqrt: {sum_sq.min().item()}. This will cause NaN in sqrt.")
        
        sqrt_term = torch.sqrt(sum_sq + 1e-6)
        self.check_tensor(sqrt_term, "Energy Accuracy Sqrt (|x-y|)")
        
        diff_div = pred_x0_grouped.unsqueeze(2) - pred_x0_grouped.unsqueeze(1)
        sum_sq_div = diff_div.pow(2).sum(dim=-1)
        sqrt_term_div = torch.sqrt(sum_sq_div + 1e-6)
        self.check_tensor(sqrt_term_div, "Energy Diversity Sqrt (|x-x'|)")

        self.logger("--- Loss Diagnosis Complete ---\n")

    def diagnose_gradient_source(self, loss_dict):
        """
        定位坏梯度的来源
        loss_dict: {'mse': l_mse, 'energy': l_energy, 'nll': l_nll}
        注意：这需要重新进行 backward pass
        """
        self.logger("\n--- Starting Gradient Source Diagnosis ---")
        self.logger("Triggered because gradients contained NaN even though Loss value was valid.")
        
        for name, loss_tensor in loss_dict.items():
            self.logger(f"Testing gradient for component: {name}")
            # 清空之前可能存在的梯度
            self.model.zero_grad()
            try:
                # 尝试单独反向传播
                # retain_graph=True 因为我们可能对多个 loss 共享同一个 graph 进行测试
                loss_tensor.backward(retain_graph=True)
                
                # 检查梯度
                grad_is_nan = False
                for param in self.model.parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            grad_is_nan = True
                            break
                
                if grad_is_nan:
                    self.logger(f"[CULPRIT FOUND] >>> Loss component '{name}' produced NaN gradients! <<<")
                else:
                    self.logger(f"[PASS] Loss component '{name}' gradients are clean.")
                    
            except Exception as e:
                self.logger(f"[ERROR] Backward pass failed for {name}: {e}")
        
        self.logger("--- Gradient Diagnosis Complete ---\n")