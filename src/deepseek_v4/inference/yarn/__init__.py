"""
YaRN 长文本外推工具。

模块：
    config.py        YarnConfig + 三种 scaling 方法的工厂
    apply.py         给已加载的模型动态注入 YaRN（不重训）
    needle.py        Needle-in-a-Haystack 压力测试
"""
from deepseek_v4.inference.yarn.config import (
    YarnConfig, RopeScalingMethod, build_rope_scaling,
)
from deepseek_v4.inference.yarn.apply import (
    apply_yarn_to_config, apply_yarn_to_model, recompute_inv_freq_buffers,
)
from deepseek_v4.inference.yarn.needle import (
    NeedleConfig, run_needle_test, NeedleResult,
)

__all__ = [
    "YarnConfig", "RopeScalingMethod", "build_rope_scaling",
    "apply_yarn_to_config", "apply_yarn_to_model", "recompute_inv_freq_buffers",
    "NeedleConfig", "run_needle_test", "NeedleResult",
]
