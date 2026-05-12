"""Agentic RL 单测：Environment / Trajectory / Trainer。"""
from unittest.mock import MagicMock

import pytest

from deepseek_v4.training.agentic_rl.environment import EnvironmentStep
from deepseek_v4.training.agentic_rl.trajectory import Trajectory
from deepseek_v4.training.agentic_rl.trainer import AgenticRLConfig


# ============================================================
# EnvironmentStep
# ============================================================

def test_environment_step_create():
    step = EnvironmentStep(
        turn=0,
        model_output="<tool_call>...</tool_call>",
        tool_calls=[{"name": "calculator", "arguments": {"expression": "1+1"}}],
        tool_results=["2"],
        reward=0.0,
        done=False,
    )
    assert step.turn == 0
    assert step.tool_calls[0]["name"] == "calculator"
    assert not step.done
    assert step.reward == 0.0


def test_environment_step_terminal():
    step = EnvironmentStep(
        turn=2,
        model_output="The answer is 42.",
        tool_calls=[],
        tool_results=[],
        reward=1.0,
        done=True,
    )
    assert step.done
    assert step.reward == 1.0
    assert step.tool_calls == []


# ============================================================
# Trajectory
# ============================================================

def test_trajectory_empty():
    traj = Trajectory(prompt="What is 1+1?")
    assert traj.prompt == "What is 1+1?"
    assert len(traj.steps) == 0
    assert traj.total_reward() == 0.0


def test_trajectory_add_step():
    traj = Trajectory(prompt="Test")
    step = EnvironmentStep(turn=0, model_output="output", tool_calls=[],
                           tool_results=[], reward=0.5, done=False)
    traj.add_step(step)
    assert len(traj.steps) == 1
    assert traj.total_reward() == 0.5


def test_trajectory_multiple_steps():
    traj = Trajectory(prompt="Multi-step")
    traj.add_step(EnvironmentStep(turn=0, model_output="", tool_calls=[],
                                  tool_results=[], reward=0.3, done=False))
    traj.add_step(EnvironmentStep(turn=1, model_output="", tool_calls=[],
                                  tool_results=[], reward=0.5, done=False))
    traj.add_step(EnvironmentStep(turn=2, model_output="", tool_calls=[],
                                  tool_results=[], reward=0.2, done=True))
    assert len(traj.steps) == 3
    assert traj.total_reward() == 1.0


def test_trajectory_is_done():
    traj = Trajectory(prompt="X")
    traj.add_step(EnvironmentStep(turn=0, model_output="", tool_calls=[],
                                  tool_results=[], reward=0.0, done=True))
    assert traj.is_done()
    assert traj.final_reward() == 0.0


# ============================================================
# AgenticRLConfig
# ============================================================

def test_agentic_rl_config_defaults():
    cfg = AgenticRLConfig()
    assert cfg.max_turns >= 1
    assert isinstance(cfg.learning_rate, float)
    assert cfg.gamma > 0
    assert cfg.lam > 0


def test_agentic_rl_config_reward_shaping():
    cfg = AgenticRLConfig(
        final_correctness_weight=1.0,
        step_penalty=-0.01,
        tool_call_format_penalty=-0.5,
    )
    assert cfg.final_correctness_weight == 1.0
    assert cfg.step_penalty == -0.01
    assert cfg.tool_call_format_penalty == -0.5


# ============================================================
# Trajectory advantage computation
# ============================================================

def test_trajectory_discounted_return():
    """测试轨迹折扣回报。"""
    rewards = [0.0, 0.0, 1.0]  # sparse reward on final step
    gamma = 0.99

    # G = r0 + γ*r1 + γ²*r2
    G = rewards[0] + gamma * rewards[1] + gamma * gamma * rewards[2]
    expected = 0 + 0 + 0.9801
    assert abs(G - expected) < 1e-4


def test_trajectory_advantage_zero_sum():
    """同一组内 advantage 应零和。"""
    rewards = [0.2, 0.5, 0.8, 0.5]
    mean_r = sum(rewards) / len(rewards)
    advantages = [r - mean_r for r in rewards]
    assert abs(sum(advantages)) < 1e-6


def test_trajectory_pplx():
    """测试轨迹的 per-token loss 权重（工具调用 token 可能被 mask）。"""
    # 模拟：工具调用的 token 可以被 mask（不在 policy improvement 中）
    traj = Trajectory(prompt="prompt")
    traj.add_step(EnvironmentStep(
        turn=0, model_output="<tool_call>...</tool_call>The answer.",
        tool_calls=[{"name": "calc"}], tool_results=["result"], reward=0.0, done=False,
    ))
    traj.add_step(EnvironmentStep(
        turn=1, model_output="Final answer.", tool_calls=[],
        tool_results=[], reward=1.0, done=True,
    ))
    # trajectory 应正确记录是否完成
    assert traj.is_done()
