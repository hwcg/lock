"""
工具 Schema 校验。

解析 DSML 调用：
    <｜DSML｜tool_calls>
    <｜DSML｜invoke name="calc">
    <｜DSML｜parameter name="expr" string="true">1+1</｜DSML｜parameter>
    </｜DSML｜invoke>
    </｜DSML｜tool_calls>
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union


# ============================================================
# Tool Schema
# ============================================================

@dataclass
class ToolSchema:
    """单个工具的 OpenAI 兼容 schema。"""
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)   # JSON Schema

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ToolSchema":
        # OpenAI 格式: {"type": "function", "function": {...}}
        if "function" in d:
            d = d["function"]
        return cls(
            name=d["name"],
            description=d.get("description", ""),
            parameters=d.get("parameters", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


# ============================================================
# DSML 解析（轻量版，独立于 encoding.parse_tool_calls）
# ============================================================

DSML_TOKEN = "｜DSML｜"
_INVOKE_OPEN_RE = re.compile(
    rf'<{re.escape(DSML_TOKEN)}invoke\s+name="([^"]+)">'
)
_PARAM_RE = re.compile(
    rf'<{re.escape(DSML_TOKEN)}parameter\s+name="([^"]+)"\s+string="(true|false)">(.*?)</{re.escape(DSML_TOKEN)}parameter>',
    re.DOTALL,
)
_INVOKE_BLOCK_RE = re.compile(
    rf'<{re.escape(DSML_TOKEN)}invoke[^>]*>(.*?)</{re.escape(DSML_TOKEN)}invoke>',
    re.DOTALL,
)


def parse_dsml_tool_calls(text: str) -> List[Dict[str, Any]]:
    """
    解析文本中所有 DSML invoke 块。

    Returns:
        [{"name": str, "arguments": dict}, ...]
    """
    calls: List[Dict[str, Any]] = []
    for block in _INVOKE_BLOCK_RE.finditer(text):
        body = block.group(1)
        # 找 invoke name
        m_name = _INVOKE_OPEN_RE.search(text, block.start(), block.start() + len(block.group(0)))
        if not m_name:
            continue
        name = m_name.group(1)
        args: Dict[str, Any] = {}
        for p in _PARAM_RE.finditer(body):
            key = p.group(1)
            is_str = p.group(2) == "true"
            val_raw = p.group(3)
            if is_str:
                args[key] = val_raw
            else:
                try:
                    args[key] = json.loads(val_raw)
                except Exception:
                    args[key] = val_raw
        calls.append({"name": name, "arguments": args})
    return calls


# ============================================================
# 校验
# ============================================================

@dataclass
class ValidationResult:
    """单个调用的校验结果。"""
    ok: bool
    errors: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.ok


def _check_type(value: Any, type_spec: str) -> bool:
    """JSON Schema 简化类型检查。"""
    if type_spec == "string":
        return isinstance(value, str)
    if type_spec == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_spec == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_spec == "boolean":
        return isinstance(value, bool)
    if type_spec == "array":
        return isinstance(value, list)
    if type_spec == "object":
        return isinstance(value, dict)
    if type_spec == "null":
        return value is None
    return True   # 未知类型不校验


def validate_tool_call(
    call: Dict[str, Any],
    schemas: Dict[str, ToolSchema],
    strict_extra_keys: bool = True,
) -> ValidationResult:
    """
    校验一个调用是否符合任一 schema。

    检查：
    1. tool name 在 schemas 中
    2. 所有 required 参数都存在
    3. 类型匹配
    4. 严格模式下不允许多余字段
    """
    errors: List[str] = []
    name = call.get("name")
    args = call.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            errors.append(f"arguments 不是合法 JSON")
            args = {}

    if name not in schemas:
        errors.append(f"未知工具: {name}")
        return ValidationResult(ok=False, errors=errors)

    schema = schemas[name]
    props = schema.parameters.get("properties", {}) or {}
    required = schema.parameters.get("required", []) or []

    # required 检查
    for k in required:
        if k not in args:
            errors.append(f"缺少必需参数: {k}")

    # 类型 + 多余字段
    for k, v in args.items():
        if k not in props:
            if strict_extra_keys:
                errors.append(f"未知参数: {k}")
            continue
        type_spec = props[k].get("type")
        if type_spec is not None:
            if isinstance(type_spec, list):
                # union 类型
                if not any(_check_type(v, t) for t in type_spec):
                    errors.append(f"参数 {k} 类型不符: 期望 {type_spec}, 实际 {type(v).__name__}")
            else:
                if not _check_type(v, type_spec):
                    errors.append(f"参数 {k} 类型不符: 期望 {type_spec}, 实际 {type(v).__name__}")
        # enum
        if "enum" in props[k] and v not in props[k]["enum"]:
            errors.append(f"参数 {k} 不在 enum {props[k]['enum']} 中")

    return ValidationResult(ok=(len(errors) == 0), errors=errors)


def validate_dsml_text(
    text: str,
    schemas: Dict[str, ToolSchema],
) -> Tuple[bool, List[Dict[str, Any]], List[str]]:
    """对一段文本中所有调用做校验。返回 (全部 ok, calls, error 列表)。"""
    calls = parse_dsml_tool_calls(text)
    all_errors: List[str] = []
    all_ok = True
    for c in calls:
        r = validate_tool_call(c, schemas)
        if not r.ok:
            all_ok = False
            all_errors.extend([f"[{c.get('name')}] {e}" for e in r.errors])
    return all_ok, calls, all_errors
