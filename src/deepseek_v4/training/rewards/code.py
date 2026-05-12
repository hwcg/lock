"""
代码执行 reward（沙盒）。

警告：
- 直接执行 LLM 输出代码具有风险。本实现使用：
  * subprocess + 资源限制（CPU 时间、内存、文件系统隔离）
  * 仅 stdlib，无网络
- 生产环境强烈建议替换为容器/虚拟机沙盒（如 firejail / docker）。
"""
from __future__ import annotations

import os
import re
import resource
import signal
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from deepseek_v4.training.rewards.base import NamedReward, RewardFunction
from deepseek_v4.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# 抽取代码块
# ============================================================

_CODE_BLOCK_RE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)


def extract_python_code(text: str) -> Optional[str]:
    """从 markdown 代码块抽取 python 代码（取最后一段）。"""
    matches = _CODE_BLOCK_RE.findall(text)
    if matches:
        return matches[-1].strip()
    return None


# ============================================================
# 沙盒执行
# ============================================================

def _set_limits(cpu_seconds: int, mem_mb: int) -> None:
    """子进程内设置资源 limit。"""
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        resource.setrlimit(resource.RLIMIT_AS, (mem_mb * 1024 * 1024, mem_mb * 1024 * 1024))
    except Exception:
        pass


def run_python_sandboxed(
    code: str,
    stdin: str = "",
    timeout: float = 5.0,
    cpu_seconds: int = 5,
    mem_mb: int = 512,
) -> Tuple[bool, str, str]:
    """
    在子进程沙盒中执行 python 代码。

    Returns:
        (success, stdout, stderr)
    """
    if not code:
        return False, "", "empty code"

    with tempfile.TemporaryDirectory() as tmpdir:
        f = Path(tmpdir) / "main.py"
        f.write_text(code, encoding="utf-8")
        cmd = [sys.executable, str(f)]
        try:
            proc = subprocess.run(
                cmd,
                input=stdin,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmpdir,
                env={
                    "PATH": os.environ.get("PATH", ""),
                    "LC_ALL": "C.UTF-8",
                    # 屏蔽常见易出错的环境变量
                    "PYTHONHASHSEED": "0",
                },
                preexec_fn=(lambda: _set_limits(cpu_seconds, mem_mb))
                            if hasattr(os, "fork") else None,
            )
            return (proc.returncode == 0), proc.stdout, proc.stderr
        except subprocess.TimeoutExpired:
            return False, "", f"timeout after {timeout}s"
        except Exception as e:
            return False, "", f"sandbox error: {e}"


# ============================================================
# Reward 函数
# ============================================================

def code_python_reward(
    correct_reward: float = 1.0,
    error_reward: float = -0.5,
    no_code_reward: float = -0.5,
    timeout: float = 5.0,
) -> RewardFunction:
    """
    执行抽取出的 python 代码，能跑通就给正分。

    references 不参与（仅看执行是否成功）。
    """
    def fn(completions, references=None, prompts=None, **kwargs):
        out = []
        for c in completions:
            code = extract_python_code(c)
            if code is None:
                out.append(no_code_reward)
                continue
            ok, _, _ = run_python_sandboxed(code, timeout=timeout)
            out.append(correct_reward if ok else error_reward)
        return out
    return NamedReward(fn, name="code_python")


def code_execute_reward(
    correct_reward: float = 1.0,
    wrong_reward: float = 0.0,
    error_reward: float = -0.5,
    no_code_reward: float = -0.5,
    timeout: float = 5.0,
) -> RewardFunction:
    """
    执行代码并对比 stdout 与 references。

    references[i] 可以是：
        - str：期望的 stdout（trim 后字符串比较）
        - List[Tuple[str, str]]：[(stdin, expected_stdout), ...] 多用例
        - dict: { stdin: str, expected: str, ... }
    """
    def fn(completions, references=None, prompts=None, **kwargs):
        if references is None:
            raise ValueError("code_execute_reward requires references")
        out = []
        for c, ref in zip(completions, references):
            code = extract_python_code(c)
            if code is None:
                out.append(no_code_reward)
                continue
            # 标准化 ref 为用例列表
            cases: List[Tuple[str, str]]
            if isinstance(ref, str):
                cases = [("", ref)]
            elif isinstance(ref, dict):
                cases = [(ref.get("stdin", ""), str(ref.get("expected", "")))]
            elif isinstance(ref, list):
                cases = [(str(s[0]) if len(s) > 0 else "", str(s[1])) for s in ref]
            else:
                out.append(error_reward)
                continue

            ok_count = 0
            err = False
            for stdin, expected in cases:
                success, stdout, stderr = run_python_sandboxed(
                    code, stdin=stdin, timeout=timeout,
                )
                if not success:
                    err = True
                    break
                if stdout.strip() == expected.strip():
                    ok_count += 1
            if err:
                out.append(error_reward)
            else:
                # 全对 = correct_reward；部分对 = 线性插值
                if ok_count == len(cases):
                    out.append(correct_reward)
                elif ok_count == 0:
                    out.append(wrong_reward)
                else:
                    frac = ok_count / len(cases)
                    out.append(wrong_reward + (correct_reward - wrong_reward) * frac)
        return out
    return NamedReward(fn, name="code_execute")
