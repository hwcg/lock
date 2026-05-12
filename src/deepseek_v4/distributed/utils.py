"""
DDP / NCCL 通用工具。

设计：
- 与 torchrun 完全兼容（读取 RANK / LOCAL_RANK / WORLD_SIZE 环境变量）
- 在单进程模式下所有函数都安全降级（rank=0, world_size=1）
- 提供 `only_on_main` 装饰器简化"仅主进程执行"逻辑
"""
from __future__ import annotations

import functools
import os
import socket
from contextlib import contextmanager
from typing import Any, Callable, List, Optional

import torch
import torch.distributed as dist


# ============================================================
# Rank / World 信息（即便未初始化也能调用）
# ============================================================

def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def get_local_rank() -> int:
    if is_distributed():
        return int(os.environ.get("LOCAL_RANK", 0))
    return 0


def is_main_process() -> bool:
    return get_rank() == 0


# ============================================================
# 初始化 / 清理
# ============================================================

def _find_free_port() -> int:
    """寻找空闲端口（用于单机 DDP 不指定 MASTER_PORT 的兜底）。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def setup_distributed(
    backend: str = "nccl",
    timeout_seconds: int = 1800,
    init_method: Optional[str] = None,
) -> None:
    """
    DDP 初始化。

    优先从环境变量读取（torchrun 已设置）：
        RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT

    若环境变量缺失（单机单卡），不做任何事（保持 dist 未初始化状态）。
    """
    if is_distributed():
        return

    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        # 单卡：不初始化分布式
        return

    if backend == "nccl" and not torch.cuda.is_available():
        backend = "gloo"

    if init_method is None:
        # torchrun 已经设好 env://
        init_method = "env://"

    import datetime
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        timeout=datetime.timedelta(seconds=timeout_seconds),
    )

    if torch.cuda.is_available():
        torch.cuda.set_device(get_local_rank())


def cleanup_distributed() -> None:
    if is_distributed():
        dist.barrier()
        dist.destroy_process_group()


# ============================================================
# 集体通信封装
# ============================================================

def barrier() -> None:
    if is_distributed():
        dist.barrier()


def broadcast_object_list(obj_list: List[Any], src: int = 0) -> List[Any]:
    """跨 rank 广播任意 Python 对象（基于 pickle）。"""
    if not is_distributed():
        return obj_list
    obj_list = list(obj_list)
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """跨 rank 求均值。原 tensor 不变。"""
    if not is_distributed():
        return tensor
    tensor = tensor.detach().clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= get_world_size()
    return tensor


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    """跨 rank 求和。"""
    if not is_distributed():
        return tensor
    tensor = tensor.detach().clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def all_gather_tensors(tensor: torch.Tensor) -> List[torch.Tensor]:
    """跨 rank 收集所有 tensor。要求 shape 相同。"""
    if not is_distributed():
        return [tensor]
    world_size = get_world_size()
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    return gathered


# ============================================================
# 装饰器与打印
# ============================================================

def only_on_main(fn: Callable) -> Callable:
    """装饰器：仅主进程执行，其他进程返回 None。"""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if is_main_process():
            return fn(*args, **kwargs)
        return None
    return wrapper


def ddp_print(*args, **kwargs) -> None:
    """只有主进程才 print。"""
    if is_main_process():
        print(*args, **kwargs)


@contextmanager
def main_first():
    """让主进程先执行 with 块内代码，其他进程等待（用于下载缓存等）。"""
    if is_distributed():
        if not is_main_process():
            dist.barrier()
        yield
        if is_main_process():
            dist.barrier()
    else:
        yield
