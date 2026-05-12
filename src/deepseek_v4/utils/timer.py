"""计时与进度工具。"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, Optional


def format_time(seconds: float) -> str:
    """1234 → '20m 34s'，3661 → '1h 01m 01s'。"""
    if seconds < 0:
        return "-" + format_time(-seconds)
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60):02d}s"
    if seconds < 86400:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h}h {m:02d}m {s:02d}s"
    d = int(seconds // 86400)
    h = int((seconds % 86400) // 3600)
    return f"{d}d {h:02d}h"


class Timer:
    """简单上下文计时器。

    示例：
        with Timer() as t:
            do_something()
        print(t.elapsed)   # → 秒数
    """
    def __init__(self, name: Optional[str] = None):
        self.name = name
        self.start: Optional[float] = None
        self.elapsed: float = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start

    def __repr__(self):
        return f"Timer({self.name}={format_time(self.elapsed)})"


class Stopwatch:
    """累计多段时间。

    示例：
        sw = Stopwatch()
        with sw.track("forward"):
            ...
        with sw.track("backward"):
            ...
        sw.report()
    """
    def __init__(self):
        self.records: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}

    @contextmanager
    def track(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self.records[name] = self.records.get(name, 0.0) + dt
            self.counts[name] = self.counts.get(name, 0) + 1

    def reset(self) -> None:
        self.records.clear()
        self.counts.clear()

    def report(self) -> Dict[str, Dict[str, float]]:
        return {
            name: {
                "total":  total,
                "count":  self.counts[name],
                "mean":   total / self.counts[name],
            }
            for name, total in self.records.items()
        }

    def __str__(self):
        report = self.report()
        lines = []
        for name, info in sorted(report.items(), key=lambda kv: -kv[1]["total"]):
            lines.append(
                f"  {name:20s} total={format_time(info['total'])}  "
                f"avg={format_time(info['mean'])}  n={info['count']}"
            )
        return "\n".join(lines)
