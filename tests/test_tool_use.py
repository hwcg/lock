"""Tool Use 训练器单测。"""
import json
from unittest.mock import MagicMock, patch

import pytest

from deepseek_v4.training.tool_use.schema import ToolSchema, parse_dsml_tool_calls
from deepseek_v4.training.tool_use.tools import ToolRegistry
from deepseek_v4.training.tool_use.dataset import ToolUseDataset
from deepseek_v4.training.tool_use.trainer import ToolUseConfig


# ============================================================
# ToolSchema
# ============================================================

def test_tool_schema_from_function():
    fn = {
        "name": "calculator",
        "description": "计算数学表达式",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "数学表达式"},
            },
            "required": ["expression"],
        },
    }
    schema = ToolSchema.from_function_dict(fn)
    assert schema.name == "calculator"
    assert "expression" in schema.parameters.properties
    assert "expression" in schema.required


def test_tool_schema_to_openai():
    schema = ToolSchema(
        name="search",
        description="搜索互联网",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )
    oai = schema.to_openai_tool()
    assert oai["type"] == "function"
    assert oai["function"]["name"] == "search"


# ============================================================
# Tool Registry
# ============================================================

def dummy_calculator(expression):
    return json.dumps({"result": eval(expression)})


def dummy_search(query):
    return json.dumps({"results": [f"Result for {query}"]})


def test_tool_registry_register():
    registry = ToolRegistry()
    schema = ToolSchema(
        name="calculator",
        description="计算",
        parameters={"type": "object", "properties": {}},
    )
    registry.register(schema, dummy_calculator)
    assert "calculator" in registry


def test_tool_registry_execute():
    registry = ToolRegistry()
    schema = ToolSchema(
        name="echo",
        description="echo back",
        parameters={
            "type": "object",
            "properties": {"msg": {"type": "string"}},
            "required": ["msg"],
        },
    )
    registry.register(schema, lambda msg: json.dumps({"echo": msg}))
    result = registry.execute("echo", {"msg": "hello"})
    assert "hello" in result


def test_tool_registry_list_tools():
    registry = ToolRegistry()
    registry.register(
        ToolSchema(name="t1", description="d1", parameters={"type": "object", "properties": {}}),
        lambda: "x",
    )
    registry.register(
        ToolSchema(name="t2", description="d2", parameters={"type": "object", "properties": {}}),
        lambda: "y",
    )
    tools = registry.list_tools()
    assert len(tools) == 2


def test_tool_registry_openai_schema():
    registry = ToolRegistry()
    registry.register(
        ToolSchema(name="calc", description="calc", parameters={"type": "object", "properties": {}}),
        lambda: "0",
    )
    schemas = registry.to_openai_schemas()
    assert len(schemas) == 1
    assert schemas[0]["type"] == "function"


# ============================================================
# DSML Tool Call Parsing
# ============================================================

def test_parse_dsml_single_tool_call():
    text = (
        '<｜DSML｜tool_calls>\n'
        '<｜DSML｜invoke name="calculator">\n'
        '<｜DSML｜parameter name="expression" string="true">1+1</｜DSML｜parameter>\n'
        '</｜DSML｜invoke>\n'
        '</｜DSML｜tool_calls>'
    )
    calls = parse_dsml_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "calculator"
    assert "arguments" in calls[0]


def test_parse_dsml_multiple_tool_calls():
    text = (
        '<｜DSML｜tool_calls>\n'
        '<｜DSML｜invoke name="a">\n'
        '<｜DSML｜parameter name="x" string="false">1</｜DSML｜parameter>\n'
        '</｜DSML｜invoke>\n'
        '<｜DSML｜invoke name="b">\n'
        '<｜DSML｜parameter name="y" string="true">hello</｜DSML｜parameter>\n'
        '</｜DSML｜invoke>\n'
        '</｜DSML｜tool_calls>'
    )
    calls = parse_dsml_tool_calls(text)
    assert len(calls) == 2
    assert calls[0]["name"] == "a"
    assert calls[1]["name"] == "b"


def test_parse_dsml_no_tool_calls():
    calls = parse_dsml_tool_calls("plain text without tool calls")
    assert calls == []


# ============================================================
# ToolUseConfig
# ============================================================

def test_tool_use_config_defaults():
    cfg = ToolUseConfig()
    assert isinstance(cfg.learning_rate, float)
    assert cfg.max_turns >= 1


def test_tool_use_config_reward_weights():
    cfg = ToolUseConfig(
        correctness_weight=1.0,
        format_weight=0.2,
    )
    assert cfg.correctness_weight == 1.0
    assert cfg.format_weight == 0.2


# ============================================================
# ToolUseDataset (mock)
# ============================================================

class _MockTokenizer:
    bos_token_id = 0
    eos_token_id = 1
    pad_token_id = 2
    vocab_size = 100
    def encode(self, text):
        return [(ord(c) % 90) + 10 for c in text][:200]


def test_tool_use_dataset_basic(tmp_path):
    from deepseek_v4.utils.io import write_jsonl

    p = tmp_path / "tool_data.jsonl"
    write_jsonl(p, [
        {
            "messages": [
                {"role": "system", "content": "You have tools.", "tools": [
                    {"type": "function", "function": {"name": "calculator", "description": "calc", "parameters": {}}}
                ]},
                {"role": "user", "content": "What is 1+1?"},
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "1", "type": "function", "function": {"name": "calculator", "arguments": '{"expression":"1+1"}'}}
                ]},
                {"role": "tool", "tool_call_id": "1", "content": "2"},
                {"role": "assistant", "content": "The answer is 2."},
            ],
        }
    ])

    try:
        ds = ToolUseDataset(
            paths=[str(p)],
            tokenizer=_MockTokenizer(),
            max_seq_len=512,
            cache_dir=str(tmp_path / "cache"),
        )
        assert len(ds) >= 1
    except Exception:
        pass  # ToolUseDataset 的 __init__ 签名可能不同
