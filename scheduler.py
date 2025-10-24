from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import math
from typing import List
class MultiStageOneCycleLR(_LRScheduler):
    """
    一个自定义的学习率调度器，它结合了线性预热、保持和余弦退火。
    新增了手动触发退火的功能。

    该调度器将学习率在训练过程中分为三个阶段进行调整：
    1. 线性预热：在前 `warmup_steps` 步中，学习率从0线性增加到 `base_lr`。
    2. 保持阶段：在预热之后，学习率保持为 `base_lr`。
    3. 余弦退火：在最后的 `cooldown_steps` 步中，学习率按余弦曲线从 `base_lr` 衰减到0。

    可以通过调用 `trigger_cooldown()` 方法来随时强制进入退火阶段。

    Args:
        optimizer (Optimizer): 被包装的优化器。
        total_steps (int): 训练过程的总步数。
        warmup_ratio (float): 用于线性预热的步数占总步数的比例。
        cooldown_ratio (float): 用于余弦退火的步数占总步数的比例。
        last_epoch (int, optional): 最后一个周期的索引。默认为 -1。
        verbose (bool, optional): 如果为 True，则每次更新时打印一条信息。默认为 False。
    """
    def __init__(self, optimizer: Optimizer, total_steps: int, warmup_ratio: float, cooldown_ratio: float, last_epoch: int = -1, verbose: bool = False):
        # 计算各个阶段的步数
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * warmup_ratio)
        self.cooldown_steps = int(total_steps * cooldown_ratio)
        
        # 计算保持阶段的步数
        self.constant_steps = total_steps - self.warmup_steps - self.cooldown_steps
        
        # 进行合法性检查
        if self.constant_steps < 0:
            raise ValueError("预热比例和退火比例的总和不能超过1。")

        # 计算退火阶段的起始步
        self.cooldown_start_step = self.warmup_steps + self.constant_steps

        # 新增状态：用于手动触发退火
        self.cooldown_triggered = False
        self.cooldown_trigger_step = None

        super().__init__(optimizer, last_epoch)

    def trigger_cooldown(self):
        """
        手动触发退火阶段。一旦调用，总训练步数将被更新，
        训练将在预设的 cooldown_steps 步数后结束。
        """
        if not self.cooldown_triggered:
            # 新的总步数 = 当前步数 + 预设的退火步数
            new_total_steps = self.last_epoch + self.cooldown_steps
            if self.verbose:
                print(f"INFO: Cooldown manually triggered at step {self.last_epoch}.")
                print(f"INFO: Original total_steps was {self.total_steps}. New total_steps is {new_total_steps}.")
            
            self.cooldown_triggered = True
            self.cooldown_trigger_step = self.last_epoch
            # 关键改动：更新总步数
            self.total_steps = new_total_steps

    def get_lr(self) -> List[float]:
        """
        根据当前步数计算学习率。
        这是 _LRScheduler 的核心方法，由 step() 方法在内部调用。
        """
        current_step = self.last_epoch

        # --- 优先处理手动触发的退火 ---
        if self.cooldown_triggered:
            # 由于 self.total_steps 已被更新，这里的逻辑无需改变
            manual_cooldown_duration = self.total_steps - self.cooldown_trigger_step
            # 当前在手动退火阶段的进度
            step_in_manual_cooldown = current_step - self.cooldown_trigger_step

            if manual_cooldown_duration <= 0:
                return [0.0 for _ in self.base_lrs]
            
            cooldown_progress = min(float(step_in_manual_cooldown) / float(manual_cooldown_duration), 1.0)
            cooldown_factor = 0.5 * (1.0 + math.cos(math.pi * cooldown_progress))
            
            return [base_lr * cooldown_factor for base_lr in self.base_lrs]

        # --- 默认调度逻辑 ---
        # 阶段1: 线性预热
        if self.warmup_steps > 0 and current_step < self.warmup_steps:
            warmup_factor = float(current_step + 1) / float(self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        # 阶段3: 预设的余弦退火
        elif current_step >= self.cooldown_start_step:
            step_in_cooldown = current_step - self.cooldown_start_step
            
            if self.cooldown_steps == 0:
                cooldown_progress = 1.0
            else:
                cooldown_progress = min(float(step_in_cooldown) / float(self.cooldown_steps), 1.0)
            
            cooldown_factor = 0.5 * (1.0 + math.cos(math.pi * cooldown_progress))
            return [base_lr * cooldown_factor for base_lr in self.base_lrs]

        # 阶段2: 保持阶段
        else:
            return [base_lr for base_lr in self.base_lrs]