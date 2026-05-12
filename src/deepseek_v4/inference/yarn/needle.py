"""
Needle-in-a-Haystack 长文本压力测试。

构造 prompt：
    haystack 文本（语料）
    在指定位置（depth_ratio in [0, 1]）插入 needle：
        "The magic number for {ID} is {VALUE}."
    末尾问：
        "What is the magic number for {ID}?"

评测模型能否在指定深度找回 needle 的 value。
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from deepseek_v4.utils.io import safe_save_json
from deepseek_v4.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# 默认 haystack
# ============================================================

DEFAULT_HAYSTACK_PARAGRAPHS = [
    "The most important thing in life is to keep learning.",
    "When I was young, I thought success meant being famous.",
    "Curiosity is the most reliable guide to interesting problems.",
    "Most people are surprisingly bad at predicting what will make them happy.",
    "If you want to build something great, start by building something small that works.",
    "The biggest mistake young programmers make is to use abstractions before they understand them.",
    "Reading widely is one of the best investments of time.",
    "Procrastination, in moderate doses, can be a sign that you are working on hard problems.",
    "Good writing is rewriting; the first draft is just a starting point.",
    "Real understanding only comes when you can explain something simply.",
    "Habits, more than goals, determine the trajectory of your life.",
    "The most dangerous lies are the ones we tell ourselves.",
    "Time is the only truly non-renewable resource.",
    "Specialization is for insects; humans should remain generalists when they can.",
    "Compound interest applies to knowledge too.",
]


@dataclass
class NeedleConfig:
    """Needle 测试配置。"""
    context_lengths: List[int] = field(default_factory=lambda: [
        4_000, 8_000, 16_000, 32_000, 65_000, 128_000,
    ])
    depths: List[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])
    n_repeats: int = 3
    needle_template: str = "The magic number for {key} is {value}."
    question_template: str = "What is the magic number for {key}?"
    haystack_paragraphs: Optional[List[str]] = None
    max_new_tokens: int = 32
    temperature: float = 0.0
    seed: int = 42


@dataclass
class NeedleResult:
    """单次测试结果。"""
    context_length: int
    depth: float
    needle_key: str
    needle_value: str
    prompt_preview: str
    completion: str
    success: bool


# ============================================================
# 构造 prompt
# ============================================================

def _approx_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text))


def _make_haystack(
    tokenizer,
    target_tokens: int,
    paragraphs: List[str],
    rng: random.Random,
) -> List[str]:
    """构造接近 target_tokens 长度的 haystack（句子列表）。"""
    sentences: List[str] = []
    total = 0
    safety = target_tokens * 3
    while total < target_tokens and len(sentences) < safety:
        s = rng.choice(paragraphs)
        sentences.append(s)
        total += _approx_tokens(tokenizer, s)
    return sentences


def _insert_needle(
    sentences: List[str],
    needle: str,
    depth_ratio: float,
) -> List[str]:
    """按 depth_ratio 把 needle 插入 sentences。"""
    if not sentences:
        return [needle]
    idx = int(round(depth_ratio * len(sentences)))
    idx = max(0, min(idx, len(sentences)))
    return sentences[:idx] + [needle] + sentences[idx:]


def build_needle_prompt(
    tokenizer,
    context_length: int,
    depth: float,
    needle_key: str,
    needle_value: str,
    cfg: NeedleConfig,
    rng: random.Random,
) -> str:
    """构造完整 prompt。"""
    paras = cfg.haystack_paragraphs or DEFAULT_HAYSTACK_PARAGRAPHS
    sentences = _make_haystack(tokenizer, context_length, paras, rng)

    needle_sent = cfg.needle_template.format(key=needle_key, value=needle_value)
    sentences = _insert_needle(sentences, needle_sent, depth)

    context = " ".join(sentences)
    question = cfg.question_template.format(key=needle_key)

    return (
        "Read the following text carefully and answer the question at the end.\n\n"
        f"{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


# ============================================================
# 运行测试
# ============================================================

def _check_success(completion: str, expected_value: str) -> bool:
    """检查 completion 中是否包含 expected_value。"""
    return expected_value.strip().lower() in completion.strip().lower()


def run_needle_test(
    engine,
    tokenizer,
    config: Optional[NeedleConfig] = None,
    output_dir: Optional[str] = None,
) -> List[NeedleResult]:
    """
    跑 Needle-in-a-Haystack 测试。

    engine 必须有 generate(prompts, max_new_tokens, temperature, ...) 接口。
    """
    cfg = config or NeedleConfig()
    rng = random.Random(cfg.seed)

    results: List[NeedleResult] = []
    total = len(cfg.context_lengths) * len(cfg.depths) * cfg.n_repeats
    logger.info(f"[Needle] starting: {total} trials "
                f"({len(cfg.context_lengths)} lengths x {len(cfg.depths)} depths x {cfg.n_repeats} repeats)")

    idx = 0
    for ctx_len in cfg.context_lengths:
        for depth in cfg.depths:
            for r in range(cfg.n_repeats):
                key = f"K{rng.randint(10000, 99999)}"
                value = str(rng.randint(100, 9999))

                prompt = build_needle_prompt(
                    tokenizer, ctx_len, depth, key, value, cfg, rng,
                )
                try:
                    completion = engine.generate(
                        [prompt],
                        max_new_tokens=cfg.max_new_tokens,
                        temperature=cfg.temperature,
                    )[0]
                except Exception as e:
                    completion = f"[ENGINE_ERROR] {e}"

                success = _check_success(completion, value)
                results.append(NeedleResult(
                    context_length=ctx_len, depth=depth,
                    needle_key=key, needle_value=value,
                    prompt_preview=prompt[:200],
                    completion=completion[:200],
                    success=success,
                ))
                idx += 1
                logger.info(
                    f"  [{idx}/{total}] ctx={ctx_len:>7d}  depth={depth:.2f}  "
                    f"value={value}  ->  {'OK' if success else 'FAIL'}"
                )

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        safe_save_json(out / "needle_results.json", [
            {**r.__dict__} for r in results
        ])
        agg: Dict[Tuple[int, float], List[float]] = {}
        for r in results:
            agg.setdefault((r.context_length, r.depth), []).append(1.0 if r.success else 0.0)
        rows = []
        for (ctx_len, depth), vals in sorted(agg.items()):
            rows.append({
                "context_length": ctx_len,
                "depth": depth,
                "success_rate": sum(vals) / len(vals),
                "n": len(vals),
            })
        safe_save_json(out / "needle_summary.json", rows)

        depths_sorted = sorted(cfg.depths)
        ctxs_sorted = sorted(cfg.context_lengths)
        lines = ["# Needle-in-a-Haystack Results\n",
                 "| Context \\ Depth | " + " | ".join(f"{d:.2f}" for d in depths_sorted) + " |",
                 "|---|" + "---|" * len(depths_sorted)]
        for cl in ctxs_sorted:
            row = [f"{cl:,}"]
            for d in depths_sorted:
                vals = agg.get((cl, d), [])
                if not vals:
                    row.append("-")
                else:
                    sr = sum(vals) / len(vals)
                    row.append(f"{sr*100:.0f}%")
            lines.append("| " + " | ".join(row) + " |")
        (out / "needle_report.md").write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"[Needle] report saved to {out}")

    return results
