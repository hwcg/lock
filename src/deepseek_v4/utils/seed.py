"""随机种子管理。"""
from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False, set_torch: bool = True) -> None:
    """
    全局设置随机种子。

    Args:
        seed: 种子值
        deterministic: 是否启用 cuDNN 确定性模式（牺牲性能换可复现性）
        set_torch: 是否同时设置 torch 种子
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if set_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # PyTorch 2.0+ 严格确定性
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass


def seed_worker(worker_id: int) -> None:
    """DataLoader worker 初始化函数：保证多 worker 不同种子但可复现。"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
