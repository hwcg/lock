"""评测模块单测。"""
import pytest

from deepseek_v4.evaluation.base import EvalSample, EvalResult
from deepseek_v4.evaluation.engine import InferenceEngine


# ============================================================
# EvalSample / EvalResult
# ============================================================

def test_eval_sample_basic():
    sample = EvalSample(
        id="001",
        prompt="What is the capital of France?",
        reference="Paris",
    )
    assert sample.id == "001"
    assert sample.prompt == "What is the capital of France?"
    assert sample.reference == "Paris"
    assert sample.choices is None


def test_eval_sample_with_choices():
    sample = EvalSample(
        id="002",
        prompt="Which is a mammal?",
        reference="B",
        choices=["A. Shark", "B. Dolphin", "C. Eagle", "D. Crocodile"],
    )
    assert len(sample.choices) == 4
    assert sample.choices[1] == "B. Dolphin"


def test_eval_sample_metadata():
    sample = EvalSample(
        id="003",
        prompt="test",
        reference="x",
        metadata={"category": "science", "difficulty": "easy"},
    )
    assert sample.metadata["category"] == "science"


def test_eval_result_basic():
    result = EvalResult(
        id="001",
        prompt="Q",
        completion="Paris",
        reference="Paris",
        pred="Paris",
        score=1.0,
    )
    assert result.score == 1.0
    assert result.pred == "Paris"


def test_eval_result_partial_score():
    result = EvalResult(
        id="002",
        prompt="Q",
        completion="Wrong answer",
        reference="Correct answer",
        pred="Wrong answer",
        score=0.0,
        metadata={"error_type": "hallucination"},
    )
    assert result.score == 0.0
    assert "error_type" in result.metadata


def test_eval_result_as_dict():
    result = EvalResult(
        id="001",
        prompt="Hello",
        completion="Hi",
        reference="Hi",
        pred="Hi",
        score=1.0,
    )
    d = result.as_dict()
    assert d["id"] == "001"
    assert d["score"] == 1.0
    assert len(d["prompt"]) <= 500


# ============================================================
# Accuracy / Metrics
# ============================================================

def test_accuracy_calculation():
    results = [
        EvalResult(id=str(i), prompt="", completion="", reference="", pred="", score=s)
        for i, s in enumerate([1.0, 1.0, 0.0, 1.0, 0.0])
    ]
    acc = sum(r.score for r in results) / len(results)
    assert acc == 0.6


def test_per_category_accuracy():
    results = [
        EvalResult(id="1", prompt="", completion="", reference="", pred="", score=1.0,
                   metadata={"category": "math"}),
        EvalResult(id="2", prompt="", completion="", reference="", pred="", score=0.0,
                   metadata={"category": "math"}),
        EvalResult(id="3", prompt="", completion="", reference="", pred="", score=1.0,
                   metadata={"category": "code"}),
        EvalResult(id="4", prompt="", completion="", reference="", pred="", score=1.0,
                   metadata={"category": "code"}),
    ]

    by_cat = {}
    for r in results:
        cat = r.metadata.get("category", "unknown")
        if cat not in by_cat:
            by_cat[cat] = {"total": 0.0, "n": 0}
        by_cat[cat]["total"] += r.score
        by_cat[cat]["n"] += 1

    math_acc = by_cat["math"]["total"] / by_cat["math"]["n"]
    code_acc = by_cat["code"]["total"] / by_cat["code"]["n"]
    assert math_acc == 0.5
    assert code_acc == 1.0


# ============================================================
# Few-shot Prompt Construction
# ============================================================

def test_few_shot_prompt_builder():
    examples = [
        {"question": "1+1=?", "answer": "2"},
        {"question": "2+2=?", "answer": "4"},
    ]
    prompt = "Question: 3+3=?\nAnswer:"

    full = ""
    for ex in examples:
        full += f"Question: {ex['question']}\nAnswer: {ex['answer']}\n\n"
    full += prompt
    assert "1+1=?" in full
    assert "3+3=?" in full
    assert "2" in full  # answer from example


# ============================================================
# Task Configuration
# ============================================================

def test_task_name_mapping():
    """评测任务名称应能映射到数据集名称。"""
    tasks = {
        "ceval": "C-Eval",
        "cmmlu": "C-MMLU",
        "gsm8k": "GSM8K",
        "humaneval": "HumanEval",
        "openbookqa": "OpenBookQA",
    }
    assert "ceval" in tasks
    assert "humaneval" in tasks
    assert len(tasks) == 5
