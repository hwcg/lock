"""
DeepSeek-V4 评测子包。

提供：
- Evaluator 抽象 + 三种评测协议（multiple_choice / open_ended / code）
- 五大评测集 loader：C-Eval / C-MMLU / OpenBookQA / GSM8K / HumanEval
- 结果聚合 + Markdown 报告
- CLI：`ds4-evaluate --task ceval --model_path ... --shots 5`
- 推理引擎抽象：本地 HF / vLLM / OpenAI 兼容 API
"""
from deepseek_v4.evaluation.base import (
    Evaluator, EvalSample, EvalResult, TaskResult,
    aggregate_results, format_markdown_report,
)
from deepseek_v4.evaluation.engine import (
    InferenceEngine, LocalEngine, OpenAIEngine, VLLMEngine, build_engine,
)
from deepseek_v4.evaluation.tasks.base_task import EvalTask, TaskRegistry, TASKS
from deepseek_v4.evaluation.tasks.ceval import CEvalTask
from deepseek_v4.evaluation.tasks.cmmlu import CMMLUTask
from deepseek_v4.evaluation.tasks.openbookqa import OpenBookQATask
from deepseek_v4.evaluation.tasks.gsm8k import GSM8KTask
from deepseek_v4.evaluation.tasks.humaneval import HumanEvalTask
from deepseek_v4.evaluation.evaluator import (
    run_evaluation, EvaluationConfig,
)

__all__ = [
    "Evaluator", "EvalSample", "EvalResult", "TaskResult",
    "aggregate_results", "format_markdown_report",
    "InferenceEngine", "LocalEngine", "OpenAIEngine", "VLLMEngine", "build_engine",
    "EvalTask", "TaskRegistry", "TASKS",
    "CEvalTask", "CMMLUTask", "OpenBookQATask", "GSM8KTask", "HumanEvalTask",
    "run_evaluation", "EvaluationConfig",
]
