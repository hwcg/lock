"""
LoRA 子包：从 0 实现的 LoRA / DoRA 适配。

主要导出：
    LoRAConfig          配置
    LoRALinear          线性层 LoRA 适配
    LoRAExpertProj      MoE expert 3D 权重适配
    LoRAGroupedLinear   V4 GroupedLinear 适配
    apply_lora          全模型注入
    merge_and_unload    合并 LoRA 权重并恢复原模型
    get_lora_state_dict 仅 LoRA 参数 state_dict
    set_lora_state_dict 仅加载 LoRA 参数
    save_lora           保存 adapter
    load_lora           加载 adapter
    print_trainable_parameters
"""
from deepseek_v4.training.lora.config import LoRAConfig
from deepseek_v4.training.lora.layers import (
    LoRALinear, LoRAExpertProj, LoRAGroupedLinear,
)
from deepseek_v4.training.lora.apply import (
    apply_lora,
    merge_and_unload,
    get_lora_state_dict,
    set_lora_state_dict,
    save_lora,
    load_lora,
    print_trainable_parameters,
    set_lora_trainable,
)

__all__ = [
    "LoRAConfig",
    "LoRALinear", "LoRAExpertProj", "LoRAGroupedLinear",
    "apply_lora", "merge_and_unload",
    "get_lora_state_dict", "set_lora_state_dict",
    "save_lora", "load_lora",
    "print_trainable_parameters", "set_lora_trainable",
]
