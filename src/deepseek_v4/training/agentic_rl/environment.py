"""
工具调用环境。

环境 = (initial prompt, tool registry, terminate condition)

模型生成 → 解析 tool_call → 执行 → 把结果作为 <tool_result> 反馈给模型 → 重复
直到：
- 模型不再调用工具（直接回答）
- 达到 max_turns
- 触发 stop signal（如显式 final_answer 调用）
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from deepseek_v4.tokenizer.encoding import encode_messages
from deepseek_v4.training.tool_use.schema import parse_dsml_tool_calls
from deepseek_v4.training.tool_use.tools import ToolRegistry


# ============================================================
# 数据结构
# ============================================================

@dataclass
class EnvironmentStep:
    """单轮信息。"""
    turn: int
    assistant_text: str               # 模型完整生成（含 tool_call）
    tool_calls: List[Dict[str, Any]]  # 解析出的调用
    tool_results: List[str]           # 调用结果（与 tool_calls 同序）
    done: bool                        # 该步是否结束 trajectory


@dataclass
class AgenticTrajectory:
    """
    一条 multi-turn trajectory。

    final_text: 最后一轮 assistant 生成的最终回答（可能为空）
    messages: 完整对话历史（OpenAI 格式）—— 用于 RL 训练时按 turn-token 标 mask

    sequences / response_mask / old_logp / ref_logp 在 RL trainer 阶段才计算
    """
    initial_messages: List[Dict[str, Any]]
    steps: List[EnvironmentStep] = field(default_factory=list)
    final_text: str = ""
    messages: List[Dict[str, Any]] = field(default_factory=list)
    success: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# 环境
# ============================================================

class ToolEnvironment:
    """
    最小工具调用环境。

    用法：
        env = ToolEnvironment(tool_registry, max_turns=5)
        traj = env.run(model, tokenizer, initial_messages, generation_config)
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        max_turns: int = 5,
        final_answer_tool: Optional[str] = None,    # 设了之后调用该工具即终止
    ):
        self.tools = tool_registry
        self.max_turns = max_turns
        self.final_answer_tool = final_answer_tool

    # ---------- 单步 ----------

    def step(self, assistant_text: str) -> EnvironmentStep:
        """
        给定一轮 assistant 输出，执行其中所有 tool_call，返回 step。
        """
        calls = parse_dsml_tool_calls(assistant_text)
        results: List[str] = []
        done = False

        if not calls:
            # 无工具调用 → trajectory 结束（最终回答）
            return EnvironmentStep(
                turn=0, assistant_text=assistant_text,
                tool_calls=[], tool_results=[],
                done=True,
            )

        for call in calls:
            if self.final_answer_tool and call["name"] == self.final_answer_tool:
                # 显式 final_answer：取 message 字段作为最终回答
                done = True
                results.append("(final answer recorded)")
                continue
            result = self.tools.execute(call)
            results.append(result)

        return EnvironmentStep(
            turn=0, assistant_text=assistant_text,
            tool_calls=calls, tool_results=results,
            done=done,
        )

    # ---------- Roll out 一整条 trajectory ----------

    @torch.no_grad()
    def run(
        self,
        model,
        tokenizer,
        initial_messages: List[Dict[str, Any]],
        generation_config,
        device: Optional[torch.device] = None,
    ) -> AgenticTrajectory:
        """
        从 initial_messages 出发，与 model 交互直到结束。
        """
        from deepseek_v4.inference.generation import generate

        if device is None:
            device = next(model.parameters()).device

        traj = AgenticTrajectory(
            initial_messages=list(initial_messages),
            messages=list(initial_messages),
        )

        for turn in range(self.max_turns):
            # 编码当前对话
            text = encode_messages(
                traj.messages,
                thinking_mode="chat",
                drop_thinking=True,
                add_default_bos_token=True,
            )
            input_ids = torch.tensor([tokenizer.encode(text)], device=device, dtype=torch.long)

            # 生成
            out = generate(
                model, input_ids,
                attention_mask=torch.ones_like(input_ids),
                config=generation_config,
            )
            response = out["responses"][0].tolist()
            assistant_text = tokenizer.decode(response, skip_special_tokens=False)
            # 去掉末尾 eos
            assistant_text = assistant_text.split(tokenizer.eos_token)[0]

            # 一步交互
            step = self.step(assistant_text)
            step.turn = turn
            traj.steps.append(step)

            # 把 assistant 消息加入历史
            assistant_msg: Dict[str, Any] = {
                "role": "assistant",
                "content": "",
            }
            if step.tool_calls:
                # OpenAI 格式 tool_calls
                assistant_msg["tool_calls"] = [
                    {
                        "id": f"call_{turn}_{i}",
                        "type": "function",
                        "function": {
                            "name": c["name"],
                            "arguments": json.dumps(c["arguments"], ensure_ascii=False),
                        },
                    }
                    for i, c in enumerate(step.tool_calls)
                ]
                # 把工具调用前的文字（如有）放 content
                # 这里简化处理
                # encode_messages 会用 tool_calls 自动渲染
            else:
                # 没有 tool call：作为最终回答
                assistant_msg["content"] = assistant_text
                traj.final_text = assistant_text

            traj.messages.append(assistant_msg)

            if step.done:
                break

            # 否则把 tool 结果作为 tool 消息加入
            for i, (call, result) in enumerate(zip(step.tool_calls, step.tool_results)):
                traj.messages.append({
                    "role": "tool",
                    "tool_call_id": f"call_{turn}_{i}",
                    "content": result,
                })

        return traj
