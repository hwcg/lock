"""
工具注册表与内置工具。

每个 Tool 包含：
- name / description / parameters schema
- 一个可调用 fn(**kwargs) -> str
- 内置安全限制（超时、内存）
"""
from __future__ import annotations

import json
import math
import re
import statistics
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from deepseek_v4.training.tool_use.schema import ToolSchema
from deepseek_v4.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# Tool 抽象
# ============================================================

@dataclass
class Tool:
    """一个可被 LLM 调用的工具。"""
    schema: ToolSchema
    fn: Callable[..., Any]
    timeout: float = 5.0

    def __call__(self, **kwargs) -> str:
        """统一返回 string（便于注入到对话）。"""
        try:
            result = self.fn(**kwargs)
            if isinstance(result, str):
                return result
            try:
                return json.dumps(result, ensure_ascii=False)
            except Exception:
                return str(result)
        except Exception as e:
            return f"[Tool {self.schema.name} error] {type(e).__name__}: {e}"


# ============================================================
# 注册表
# ============================================================

class ToolRegistry:
    """工具集合，支持 dict 接口与 OpenAI tools schema 序列化。"""

    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self.tools[tool.schema.name] = tool

    def __contains__(self, name: str) -> bool:
        return name in self.tools

    def __getitem__(self, name: str) -> Tool:
        return self.tools[name]

    def get(self, name: str) -> Optional[Tool]:
        return self.tools.get(name)

    @property
    def schemas(self) -> Dict[str, ToolSchema]:
        return {n: t.schema for n, t in self.tools.items()}

    def to_openai_tools(self) -> List[Dict[str, Any]]:
        return [
            {"type": "function", "function": t.schema.to_dict()}
            for t in self.tools.values()
        ]

    def execute(self, call: Dict[str, Any]) -> str:
        """执行单个调用，返回结果文本。"""
        name = call.get("name")
        args = call.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                return f"[ToolRegistry error] arguments 无法解析为 JSON: {args[:100]}"
        if name not in self.tools:
            return f"[ToolRegistry error] 未知工具: {name}"
        return self.tools[name](**args)


# ============================================================
# 内置工具实现
# ============================================================

# ---- 1. Calculator ----

_CALC_SAFE_NAMES = {
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
    "pow": pow, "len": len,
    "pi": math.pi, "e": math.e,
    "sqrt": math.sqrt, "log": math.log, "log2": math.log2, "log10": math.log10,
    "exp": math.exp, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan, "atan2": math.atan2,
    "floor": math.floor, "ceil": math.ceil,
    "factorial": math.factorial, "gcd": math.gcd,
}


def _calculator(expression: str) -> Any:
    """安全的数学表达式求值。"""
    # 简单 sanitize
    if not isinstance(expression, str):
        raise ValueError("expression 必须为 string")
    if re.search(r"(__|import|exec|eval|open|input|os|sys|subprocess)", expression):
        raise ValueError("非法标识符")
    code = compile(expression, "<calc>", "eval")
    for name in code.co_names:
        if name not in _CALC_SAFE_NAMES:
            raise ValueError(f"未知标识符: {name}")
    return eval(code, {"__builtins__": {}}, _CALC_SAFE_NAMES)


# ---- 2. Python sandbox ----

def _python_exec(code: str, stdin: str = "") -> str:
    """在沙盒中执行 python，返回 stdout（限制 5s / 256MB）。"""
    from deepseek_v4.training.rewards.code import run_python_sandboxed
    success, stdout, stderr = run_python_sandboxed(code, stdin=stdin, timeout=5.0)
    if success:
        return stdout.strip() or "(no output)"
    return f"[error] {stderr.strip()}"


# ---- 3. Date/Time ----

def _now(timezone_name: str = "UTC") -> str:
    """返回当前时间。"""
    if timezone_name.upper() == "UTC":
        return datetime.now(timezone.utc).isoformat(timespec="seconds")
    # 简化：仅支持 UTC（生产中可用 pytz / zoneinfo）
    return datetime.now().isoformat(timespec="seconds") + f"  [TZ={timezone_name}]"


# ---- 4. String search ----

def _search(text: str, query: str, max_matches: int = 5) -> str:
    """在 text 中搜索 query，返回前后各 50 字符的上下文。"""
    text = str(text)
    query = str(query)
    if not query:
        return "(empty query)"
    matches: List[str] = []
    for m in re.finditer(re.escape(query), text):
        s = max(m.start() - 50, 0)
        e = min(m.end() + 50, len(text))
        ctx = text[s:e].replace("\n", " ")
        matches.append(f"... {ctx} ...")
        if len(matches) >= max_matches:
            break
    if not matches:
        return f"No matches for {query!r}"
    return "\n".join(matches)


# ---- 5. Stats ----

def _stats(values: List[float]) -> Dict[str, float]:
    """统计：mean/median/stdev/min/max。"""
    values = [float(v) for v in values]
    if not values:
        return {"error": "empty"}
    return {
        "mean":   statistics.mean(values),
        "median": statistics.median(values),
        "stdev":  statistics.stdev(values) if len(values) >= 2 else 0.0,
        "min":    min(values),
        "max":    max(values),
        "n":      len(values),
    }


# ---- 6. Echo（debug） ----

def _echo(text: str) -> str:
    return text


# ============================================================
# 注册所有内置工具
# ============================================================

BUILTIN_TOOLS: Dict[str, Tool] = {}


def _build_tool(name, desc, params, fn, **kwargs) -> Tool:
    return Tool(schema=ToolSchema(name=name, description=desc, parameters=params), fn=fn, **kwargs)


def register_builtin_tools(registry: Optional[ToolRegistry] = None) -> ToolRegistry:
    """构造并返回内置工具注册表。"""
    if registry is None:
        registry = ToolRegistry()

    registry.register(_build_tool(
        name="calculator",
        desc="Evaluate a math expression. Supports +-*/, sqrt, log, sin, factorial, pi, e, etc.",
        params={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Python math expression to evaluate"}
            },
            "required": ["expression"],
        },
        fn=_calculator,
    ))

    registry.register(_build_tool(
        name="python",
        desc="Execute Python code in a sandboxed environment with 5s/256MB limit, returns stdout.",
        params={
            "type": "object",
            "properties": {
                "code":  {"type": "string", "description": "Python code to execute"},
                "stdin": {"type": "string", "description": "stdin content (optional)"},
            },
            "required": ["code"],
        },
        fn=_python_exec,
    ))

    registry.register(_build_tool(
        name="current_time",
        desc="Get current date and time.",
        params={
            "type": "object",
            "properties": {
                "timezone": {"type": "string", "description": "IANA timezone name", "default": "UTC"},
            },
            "required": [],
        },
        fn=_now,
    ))

    registry.register(_build_tool(
        name="search_text",
        desc="Search for occurrences of `query` substring within `text`, returns context windows.",
        params={
            "type": "object",
            "properties": {
                "text":  {"type": "string"},
                "query": {"type": "string"},
                "max_matches": {"type": "integer", "default": 5},
            },
            "required": ["text", "query"],
        },
        fn=_search,
    ))

    registry.register(_build_tool(
        name="statistics",
        desc="Compute basic statistics on a list of numbers.",
        params={
            "type": "object",
            "properties": {
                "values": {"type": "array", "items": {"type": "number"}},
            },
            "required": ["values"],
        },
        fn=_stats,
    ))

    registry.register(_build_tool(
        name="echo",
        desc="Echo input text back (debug).",
        params={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        fn=_echo,
    ))

    return registry


# 初始化
BUILTIN_TOOLS = register_builtin_tools().tools
