"""
评测基础抽象。

设计原则：
- 一切围绕 EvalSample → EvalResult 流水线
- Task 负责数据加载 / few-shot 构造 / scoring
- Evaluator 负责调用模型 + 聚合
- 模型层用 InferenceEngine 抽象（local / vllm / openai-api 任选）
"""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

# ============================================================
# 数据结构
# ============================================================

@dataclass
class EvalSample:
    """单条评测样本。"""
    id: str
    prompt: str
    reference: Any                         # 标准答案（字符串、字典、test cases 等）
    metadata: Dict[str, Any] = field(default_factory=dict)
    choices: Optional[List[str]] = None    # 选择题选项（"A. ...", "B. ..."）

    def __post_init__(self):
        # 强制 id 为字符串
        self.id = str(self.id)


@dataclass
class EvalResult:
    """单条评测结果。"""
    id: str
    prompt: str
    completion: str
    reference: Any
    pred: Any                              # 抽取出的预测值
    score: float                           # 0 ~ 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "prompt": self.prompt[:500],   # 避免太长
            "completion": self.completion[:1000],
            "reference": self.reference if isinstance(self.reference, (str, int, float, list, dict)) else str(self.reference),
            "pred": self.pred if isinstance(self.pred, (str, int, float, list, dict)) else str(self.pred),
            "score": float(self.score),
            "metadata": self.metadata,
        }


@dataclass
class TaskResult:
    """一个 task 的聚合结果。"""
    task: str
    num_samples: int
    accuracy: float
    subject_scores: Dict[str, float] = field(default_factory=dict)
    extra_metrics: Dict[str, float] = field(default_factory=dict)
    samples: List[EvalResult] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "num_samples": self.num_samples,
            "accuracy": self.accuracy,
            "subject_scores": self.subject_scores,
            "extra_metrics": self.extra_metrics,
        }


# ============================================================
# Evaluator 抽象
# ============================================================

class Evaluator:
    """
    通用 evaluator —— task-agnostic。

    与 Task 的解耦让我们能 mix-and-match 任意 task 和任意 engine。
    """

    def __init__(self, engine, batch_size: int = 8, show_progress: bool = True):
        self.engine = engine
        self.batch_size = batch_size
        self.show_progress = show_progress

    def evaluate_task(self, task, max_samples: Optional[int] = None) -> TaskResult:
        """跑一个完整 task。"""
        samples = task.get_samples()
        if max_samples is not None:
            samples = samples[:max_samples]

        # 由 task 决定如何生成（含 prompt 构造、generation kwargs）
        prompts = [task.build_prompt(s) for s in samples]
        generation_kwargs = task.generation_kwargs()

        # 批量推理
        completions: List[str] = []
        iterator = range(0, len(prompts), self.batch_size)
        if self.show_progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc=f"[Eval-{task.name}]", unit="batch")
        for start in iterator:
            batch = prompts[start:start + self.batch_size]
            outputs = self.engine.generate(batch, **generation_kwargs)
            completions.extend(outputs)

        # Task 负责 scoring
        results = [
            task.score_sample(s, c) for s, c in zip(samples, completions)
        ]
        return task.aggregate(results)


# ============================================================
# 聚合 / 报告
# ============================================================

def aggregate_results(results: List[EvalResult]) -> Dict[str, float]:
    """计算 accuracy / std。"""
    if not results:
        return {"accuracy": 0.0, "n": 0}
    scores = [r.score for r in results]
    return {
        "accuracy": sum(scores) / len(scores),
        "n": len(scores),
        "std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
    }


def format_markdown_report(task_results: List[TaskResult]) -> str:
    """
    把多个 TaskResult 渲染为可视化 Markdown 报告：

    | Task     | N    | Accuracy |
    |----------|------|----------|
    | C-Eval   | 1346 | 45.32%   |
    | ...
    """
    lines: List[str] = []
    lines.append("# Evaluation Report\n")

    # 总览表
    lines.append("## Overview\n")
    lines.append("| Task | N | Accuracy | Extra Metrics |")
    lines.append("|------|---|----------|---------------|")
    for tr in task_results:
        extra = ", ".join(f"{k}={v:.4f}" for k, v in tr.extra_metrics.items())
        lines.append(
            f"| {tr.task} | {tr.num_samples} | {tr.accuracy * 100:.2f}% | {extra or '-'} |"
        )
    lines.append("")

    # 子任务（学科）分数
    for tr in task_results:
        if not tr.subject_scores:
            continue
        lines.append(f"## {tr.task} — Subject breakdown\n")
        lines.append("| Subject | Accuracy |")
        lines.append("|---------|----------|")
        for sub, acc in sorted(tr.subject_scores.items(), key=lambda x: -x[1]):
            lines.append(f"| {sub} | {acc * 100:.2f}% |")
        lines.append("")

    # macro / micro 平均
    if task_results:
        macro = sum(tr.accuracy for tr in task_results) / len(task_results)
        total_n = sum(tr.num_samples for tr in task_results)
        if total_n > 0:
            micro = sum(tr.accuracy * tr.num_samples for tr in task_results) / total_n
        else:
            micro = 0.0
        lines.append("## Average\n")
        lines.append(f"- **Macro avg**: {macro * 100:.2f}%")
        lines.append(f"- **Micro avg**: {micro * 100:.2f}%")

    return "\n".join(lines)


def save_results_jsonl(results: List[EvalResult], path: Union[str, Path]) -> None:
    """逐条结果保存为 jsonl，便于事后分析。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r.as_dict(), ensure_ascii=False) + "\n")
