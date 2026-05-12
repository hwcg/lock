"""
配置系统：

- BaseConfig：可序列化、可合并、可命令行 override 的 dataclass 基类
- load_yaml / save_yaml：YAML 读写
- merge_dict：深度合并字典
- parse_overrides：把命令行 "a.b.c=1 d.e=2" 解析为字典
"""
from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

import yaml


T = TypeVar("T", bound="BaseConfig")


# ---------------- YAML 工具 ----------------

def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """加载 yaml 文件。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(path: Union[str, Path], obj: Dict[str, Any]) -> None:
    """保存为 yaml。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False, indent=2)


# ---------------- 字典合并 ----------------

def merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """深度合并：override 覆盖 base。返回新 dict（不修改 base）。"""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = merge_dict(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


# ---------------- 命令行 override 解析 ----------------

def _coerce(value: str) -> Any:
    """字符串智能转类型。"""
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    if value.lower() in ("null", "none"):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    # 尝试 json（支持列表、对象）
    if value.startswith(("[", "{")):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    return value


def parse_overrides(overrides: List[str]) -> Dict[str, Any]:
    """
    把 ["a.b=1", "c=foo"] → {"a": {"b": 1}, "c": "foo"}。

    支持的语法：
        key=value
        nested.key=value
        list=[1,2,3]
        bool=true
        null=null
    """
    result: Dict[str, Any] = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override (no '='): {item}")
        key, value = item.split("=", 1)
        v = _coerce(value)
        parts = key.split(".")
        d = result
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = v
    return result


# ---------------- BaseConfig ----------------

@dataclass
class BaseConfig:
    """
    所有训练配置的基类。

    特点：
    - 支持 from_yaml / to_yaml / from_dict / to_dict
    - 支持嵌套 BaseConfig
    - 支持 update（字典合并）
    """

    # ---------- 序列化 ----------

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: Union[str, Path]) -> None:
        save_yaml(path, self.to_dict())

    @classmethod
    def from_dict(cls: Type[T], d: Dict[str, Any]) -> T:
        """
        从 dict 构造，对嵌套 dataclass 字段递归处理。
        非定义字段会被忽略（不会抛错），便于向前兼容。
        """
        fld_map = {f.name: f for f in fields(cls)}
        kwargs: Dict[str, Any] = {}
        for k, v in d.items():
            if k not in fld_map:
                continue
            fld = fld_map[k]
            tp = fld.type
            # 嵌套 dataclass：递归
            if isinstance(v, dict) and isinstance(tp, type) and is_dataclass(tp):
                kwargs[k] = tp.from_dict(v) if issubclass(tp, BaseConfig) else tp(**v)
            else:
                kwargs[k] = v
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls: Type[T], path: Union[str, Path]) -> T:
        return cls.from_dict(load_yaml(path))

    # ---------- 修改 ----------

    def update(self: T, override: Dict[str, Any]) -> T:
        """返回合并后的新对象。"""
        merged = merge_dict(self.to_dict(), override)
        return type(self).from_dict(merged)

    # ---------- repr ----------

    def __str__(self) -> str:
        return yaml.safe_dump(self.to_dict(), allow_unicode=True, sort_keys=False, indent=2)


# ---------------- ArgumentParser helper ----------------

def add_config_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """给 argparse 添加标准配置参数：--config + 任意 KEY=VALUE override。"""
    parser.add_argument("--config", type=str, required=True, help="YAML 配置文件路径")
    parser.add_argument(
        "overrides", nargs="*",
        help="命令行 override，形如 'training.lr=1e-4 model.hidden_size=2048'",
    )
    return parser


def load_config_with_overrides(
    cfg_cls: Type[T],
    config_path: str,
    overrides: Optional[List[str]] = None,
) -> T:
    """从 yaml 加载配置并应用命令行 override。"""
    base = load_yaml(config_path)
    if overrides:
        override_dict = parse_overrides(overrides)
        base = merge_dict(base, override_dict)
    return cfg_cls.from_dict(base)
