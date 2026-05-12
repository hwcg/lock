"""
端到端 Pipeline 集成测试：
1. Tokenizer 训练   (fixture)
2. 模型构造 + 一步前向 + 一步反向
3. 一轮 Pretrain   (10 steps)
4. 一轮 SFT        (10 steps)
5. 一轮 DPO        (5 steps)
6. 评测            (用本地 engine 跑一个 mock 任务)
7. 服务端          (mock engine 端到端走通 /v1/chat/completions)
8. 格式转换        (HF export 后能再次加载)

设计原则：
- 跑得动 CPU（≤ 1 分钟，模型极小）
- 每个测试只验证 "走得通"，不验证收敛性（已在 unit test 覆盖）
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from deepseek_v4.modeling.model import DeepseekV4Config, DeepseekV4ForCausalLM


# ============================================================
# 1. Tokenizer + 编码 解码
# ============================================================

@pytest.mark.integration
def test_e2e_tokenizer_chat_template(tiny_tokenizer):
    tok, _ = tiny_tokenizer
    text = tok.apply_chat_template(
        [{"role": "user", "content": "hi"}],
        thinking_mode="chat", add_generation_prompt=True,
    )
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    assert "hi" in decoded


# ============================================================
# 2. 模型一次前向 + 反向
# ============================================================

@pytest.mark.integration
def test_e2e_model_forward_backward(tiny_model_dir, tiny_tokenizer):
    tok, _ = tiny_tokenizer
    with open(Path(tiny_model_dir) / "config.json") as f:
        cfg = DeepseekV4Config.from_dict(json.load(f))
    model = DeepseekV4ForCausalLM(cfg)
    from safetensors.torch import load_file
    model.load_state_dict(load_file(str(Path(tiny_model_dir) / "model.safetensors")), strict=False)

    input_ids = torch.tensor([[tok.bos_token_id, 5, 7, 9, tok.eos_token_id]])
    labels = input_ids.clone()
    out = model(input_ids=input_ids, labels=labels)
    assert "loss" in out
    out["loss"].backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters() if p.requires_grad)
    assert has_grad


# ============================================================
# 3. Pretrain 10 steps
# ============================================================

@pytest.mark.integration
def test_e2e_pretrain(tmp_workspace, tiny_tokenizer, tiny_pretrain_data, tiny_model_config_path):
    """跑通 PretrainTrainer 10 步。"""
    from deepseek_v4.training.pretrain import PretrainConfig, PretrainTrainer

    tok, tok_path = tiny_tokenizer
    out_dir = tmp_workspace / "pt_run"
    cfg = PretrainConfig(
        output_dir=str(out_dir),
        run_name="pt_smoke",
        train_data_paths=[tiny_pretrain_data],
        max_seq_len=64,
        use_packed_dataset=False,
        cache_dir=str(tmp_workspace / "cache_pt"),
        model_config_path=tiny_model_config_path,
        tokenizer_path=tok_path,
        seed=0,
        max_steps=10, log_steps=2, save_steps=10, eval_steps=0,
        micro_batch_size=2, gradient_accumulation_steps=1,
        learning_rate=1e-4, warmup_steps=2,
        precision="fp32", gradient_checkpointing=False,
        use_aux_loss=False, z_loss_weight=0.0,
        estimate_mfu=False, print_param_count=False,
        logger_backends=["jsonl"],
        keep_last=1, keep_best=0,
        resume_from_checkpoint=None,
    )
    with open(tiny_model_config_path) as f:
        model_cfg_dict = json.load(f)
    model_cfg = DeepseekV4Config.from_dict(model_cfg_dict)
    model_cfg.pad_token_id = tok.pad_token_id
    model = DeepseekV4ForCausalLM(model_cfg)
    model.init_weights()
    trainer = PretrainTrainer(config=cfg, model=model, tokenizer=tok)
    trainer.train()
    # 验证 ckpt 写出
    ckpts = list(out_dir.glob("checkpoint-*"))
    assert len(ckpts) >= 1


# ============================================================
# 4. SFT 10 steps
# ============================================================

@pytest.mark.integration
def test_e2e_sft(tmp_workspace, tiny_tokenizer, tiny_sft_data, tiny_model_config_path):
    from deepseek_v4.training.sft import SFTConfig, SFTTrainer

    tok, tok_path = tiny_tokenizer
    out_dir = tmp_workspace / "sft_run"
    cfg = SFTConfig(
        output_dir=str(out_dir),
        run_name="sft_smoke",
        train_data_paths=[tiny_sft_data],
        max_seq_len=64,
        cache_dir=str(tmp_workspace / "cache_sft"),
        model_config_path=tiny_model_config_path,
        tokenizer_path=tok_path,
        max_steps=10, log_steps=2, save_steps=10, eval_steps=0,
        micro_batch_size=2, gradient_accumulation_steps=1,
        learning_rate=1e-4, warmup_steps=2,
        precision="fp32", gradient_checkpointing=False,
        use_aux_loss=False, neftune_alpha=0.0,
        logger_backends=["jsonl"],
        keep_last=1, keep_best=0,
        resume_from_checkpoint=None,
    )
    with open(tiny_model_config_path) as f:
        model_cfg = DeepseekV4Config.from_dict(json.load(f))
    model_cfg.pad_token_id = tok.pad_token_id
    model = DeepseekV4ForCausalLM(model_cfg)
    model.init_weights()
    trainer = SFTTrainer(config=cfg, model=model, tokenizer=tok)
    trainer.train()
    assert any(out_dir.glob("checkpoint-*"))


# ============================================================
# 5. DPO 5 steps
# ============================================================

@pytest.mark.integration
def test_e2e_dpo(tmp_workspace, tiny_tokenizer, tiny_dpo_data, tiny_model_config_path):
    from deepseek_v4.training.dpo import DPOConfig, DPOTrainer

    tok, tok_path = tiny_tokenizer
    out_dir = tmp_workspace / "dpo_run"
    cfg = DPOConfig(
        output_dir=str(out_dir),
        run_name="dpo_smoke",
        train_data_paths=[tiny_dpo_data],
        max_prompt_len=32, max_seq_len=64,
        cache_dir=str(tmp_workspace / "cache_dpo"),
        model_config_path=tiny_model_config_path,
        tokenizer_path=tok_path,
        init_from_checkpoint="",   # 不加载，使用现 init
        max_steps=5, log_steps=1, save_steps=5, eval_steps=0,
        micro_batch_size=1, gradient_accumulation_steps=1,
        learning_rate=1e-5, warmup_steps=1,
        precision="fp32", gradient_checkpointing=False,
        beta=0.1, dpo_variant="dpo",
        logger_backends=["jsonl"],
        keep_last=1, keep_best=0,
        resume_from_checkpoint=None,
    )
    with open(tiny_model_config_path) as f:
        model_cfg = DeepseekV4Config.from_dict(json.load(f))
    model_cfg.pad_token_id = tok.pad_token_id
    policy = DeepseekV4ForCausalLM(model_cfg)
    policy.init_weights()
    trainer = DPOTrainer(config=cfg, model=policy, tokenizer=tok)
    trainer.train()
    assert any(out_dir.glob("checkpoint-*"))


# ============================================================
# 6. 评测：注册 mock task 验证 evaluator/engine
# ============================================================

@pytest.mark.integration
def test_e2e_evaluate_mock_task(tmp_workspace, tiny_model_dir, tiny_tokenizer):
    """注册一个不依赖 datasets 的 mock task，验证完整评测链路。"""
    from deepseek_v4.evaluation.base import EvalSample, EvalResult
    from deepseek_v4.evaluation.tasks.base_task import EvalTask, TASKS
    from deepseek_v4.evaluation.engine import LocalEngine
    from deepseek_v4.evaluation.evaluator import EvaluationConfig, run_evaluation

    @TASKS.register("mock_task")
    class _MockTask(EvalTask):
        name = "mock_task"
        def load_samples(self):
            return [
                EvalSample(id="1", prompt="hello", reference="x"),
                EvalSample(id="2", prompt="world", reference="y"),
            ]
        def build_prompt(self, s):
            return s.prompt
        def score_sample(self, s, completion):
            return EvalResult(
                id=s.id, prompt=s.prompt, completion=completion,
                reference=s.reference, pred=completion[:1] if completion else "",
                score=1.0,
            )
        def generation_kwargs(self):
            return {"max_new_tokens": 4, "temperature": 0.0}

    _, tok_path = tiny_tokenizer
    out_dir = tmp_workspace / "eval_run"
    cfg = EvaluationConfig(
        tasks=["mock_task"],
        n_shots=0,
        output_dir=str(out_dir),
        engine_backend="local",
        engine_kwargs={
            "model_path": tiny_model_dir,
            "tokenizer_path": tok_path,
            "device": "cpu",
            "dtype": torch.float32,
            "max_seq_len": 64,
        },
        batch_size=2,
        save_per_sample=False,
        show_progress=False,
    )
    results = run_evaluation(cfg)
    assert len(results) == 1
    assert results[0].num_samples == 2
    assert (out_dir / "report.md").exists()


# ============================================================
# 7. 服务端：仅协议层（不加载真模型，使用 MockEngine）
# ============================================================

@pytest.mark.integration
def test_e2e_server_with_mock(tmp_workspace):
    """复用 unit test 中的 MockEngine 验证服务端协议链路。"""
    from fastapi.testclient import TestClient

    from deepseek_v4.inference.server.app import build_app
    from deepseek_v4.inference.server.config import ServerConfig
    from tests.test_server_e2e import MockEngine   # 来自 Part 9 的单测文件

    cfg = ServerConfig(model_path="x", tokenizer_path="x", model_name="mock")
    app = build_app(cfg, engine=MockEngine())
    client = TestClient(app)
    r = client.post(
        "/v1/chat/completions",
        json={"model": "mock", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    assert "Hello" in r.json()["choices"][0]["message"]["content"]


# ============================================================
# 8. 格式转换 + 重新加载
# ============================================================

@pytest.mark.integration
def test_e2e_export_hf_and_reload(
    tmp_workspace, tiny_model_dir, tiny_model_config_path, tiny_tokenizer,
):
    from deepseek_v4.inference.convert.to_hf import export_to_hf
    from deepseek_v4.inference.convert.safetensors_utils import load_sharded_safetensors

    _, tok_path = tiny_tokenizer
    out = tmp_workspace / "exported_hf"
    export_to_hf(
        state_dict_dir=tiny_model_dir,
        model_config_path=tiny_model_config_path,
        tokenizer_dir=tok_path,
        output_dir=out,
        max_shard_size="100KB",   # 强制多分片
    )
    # 验证关键产物
    for fn in ["config.json", "tokenizer_config.json", "vocab.json", "merges.txt",
               "modeling_deepseek_v4.py", "configuration_deepseek_v4.py",
               "tokenization_deepseek_v4.py", "model.safetensors.index.json",
               "generation_config.json", "README.md"]:
        assert (out / fn).exists(), f"missing {fn}"
    # 重新加载权重
    sd = load_sharded_safetensors(out)
    assert len(sd) > 0
