"""
EvalTask 抽象 + 注册表。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type

from deepseek_v4.evaluation.base import EvalResult, EvalSample, TaskResult


class EvalTask:
    """
    所有评测任务的基类。

    子类需要实现：
        name:               任务名
        load_samples()      返回 List[EvalSample]
        build_prompt(s)     构造 prompt 字符串
        score_sample(s, c)  对单条结果打分 → EvalResult
        generation_kwargs() 推理超参（max_new_tokens 等）
    """
    name: str = "base"
    metric: str = "accuracy"        # 主指标名

    def __init__(
        self,
        data_dir: Optional[str] = None,
        n_shots: int = 0,
        split: str = "test",
        language: str = "auto",        # zh | en | auto
        seed: int = 42,
        few_shot_seed: int = 0,
        **kwargs,
    ):
        self.data_dir = data_dir
        self.n_shots = n_shots
        self.split = split
        self.language = language
        self.seed = seed
        self.few_shot_seed = few_shot_seed
        self._samples: Optional[List[EvalSample]] = None
        self._fewshot: List[EvalSample] = []

    # ----- 子类必须实现 -----

    def load_samples(self) -> List[EvalSample]:
        raise NotImplementedError

    def build_prompt(self, sample: EvalSample) -> str:
        raise NotImplementedError

    def score_sample(self, sample: EvalSample, completion: str) -> EvalResult:
        raise NotImplementedError

    def generation_kwargs(self) -> Dict[str, Any]:
        return {"max_new_tokens": 512, "temperature": 0.0, "top_p": 1.0}

    # ----- 通用工具 -----

    def get_samples(self) -> List[EvalSample]:
        if self._samples is None:
            self._samples = self.load_samples()
        return self._samples

    def aggregate(self, results: List[EvalResult]) -> TaskResult:
        if not results:
            return TaskResult(task=self.name, num_samples=0, accuracy=0.0)
        acc = sum(r.score for r in results) / len(results)

        # 按 subject 聚合（如果 metadata 里有）
        subject_results: Dict[str, List[float]] = {}
        for r in results:
            sub = r.metadata.get("subject")
            if sub:
                subject_results.setdefault(sub, []).append(r.score)
        subject_scores = {k: sum(v) / len(v) for k, v in subject_results.items()}

        return TaskResult(
            task=self.name,
            num_samples=len(results),
            accuracy=acc,
            subject_scores=subject_scores,
            samples=results,
        )


# ============================================================
# Task 注册表
# ============================================================

class TaskRegistry:
    """全局任务注册表。"""
    _registry: Dict[str, Type[EvalTask]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(klass):
            cls._registry[name] = klass
            klass.name = name
            return klass
        return decorator

    @classmethod
    def build(cls, name: str, **kwargs) -> EvalTask:
        if name not in cls._registry:
            raise KeyError(f"Unknown task: {name}. Available: {list(cls._registry)}")
        return cls._registry[name](**kwargs)

    @classmethod
    def names(cls) -> List[str]:
        return sorted(cls._registry.keys())


TASKS = TaskRegistry
