SHELL  := /bin/bash
PYTHON ?= python
N_GPUS ?= 1

# ============================================================
# 安装
# ============================================================
.PHONY: install install-train install-serve install-eval install-all install-dev

install:
	pip install -e ".[serve,eval]"

install-train:
	pip install -e ".[train,dev]"

install-serve:
	pip install -e ".[serve]"

install-eval:
	pip install -e ".[eval]"

install-all:
	pip install -e ".[all]"

install-dev: install-all
	pre-commit install

# ============================================================
# 数据准备
# ============================================================
.PHONY: data-pretrain data-sft data-dpo data-tool data-thinking
data-pretrain:
	$(PYTHON) scripts/prepare_data.py --stage pretrain --output_dir data
data-sft:
	$(PYTHON) scripts/prepare_data.py --stage sft --output_dir data
data-dpo:
	$(PYTHON) scripts/prepare_data.py --stage dpo --output_dir data
data-tool:
	$(PYTHON) scripts/prepare_data.py --stage tool --output_dir data
data-thinking:
	$(PYTHON) scripts/prepare_data.py --stage thinking --output_dir data

# ============================================================
# Tokenizer
# ============================================================
.PHONY: tokenizer
tokenizer:
	$(PYTHON) scripts/train_tokenizer.py \
		--config configs/tokenizer/train_config.yaml \
		--output_dir checkpoints/tokenizer

# ============================================================
# 训练阶段
# ============================================================
.PHONY: pretrain pretrain-multi pretrain-deepspeed
pretrain:
	bash scripts/launch.sh pretrain configs/training/pretrain.yaml
pretrain-multi:
	N_GPUS=$(N_GPUS) bash scripts/launch.sh pretrain configs/training/pretrain.yaml
pretrain-deepspeed:
	N_GPUS=$(N_GPUS) bash scripts/launch.sh pretrain configs/training/pretrain.yaml \
		use_deepspeed=true deepspeed_config=configs/deepspeed/zero2.json

.PHONY: sft sft-multi
sft:
	bash scripts/launch.sh sft configs/training/sft.yaml
sft-multi:
	N_GPUS=$(N_GPUS) bash scripts/launch.sh sft configs/training/sft.yaml

.PHONY: lora lora-merge
lora:
	N_GPUS=$(N_GPUS) bash scripts/launch.sh lora configs/training/lora.yaml
lora-merge:
	$(PYTHON) scripts/merge_lora.py \
		--base_model checkpoints/sft/checkpoint-final \
		--adapter   checkpoints/lora-sft/checkpoint-final \
		--output    checkpoints/lora-merged \
		--tokenizer_path checkpoints/tokenizer

.PHONY: rm dpo ppo grpo cispo
rm:
	N_GPUS=$(N_GPUS) bash scripts/launch.sh train_reward_model configs/training/reward_model.yaml
dpo:
	N_GPUS=$(N_GPUS) bash scripts/launch.sh dpo configs/training/dpo.yaml
ppo:
	N_GPUS=$(N_GPUS) bash scripts/launch.sh ppo configs/training/ppo.yaml
grpo:
	N_GPUS=$(N_GPUS) bash scripts/launch.sh grpo configs/training/grpo.yaml
cispo:
	N_GPUS=$(N_GPUS) bash scripts/launch.sh cispo configs/training/cispo.yaml

.PHONY: tool-use agentic-rl distill adaptive-thinking
tool-use:
	N_GPUS=$(N_GPUS) bash scripts/launch.sh tool_use configs/training/tool_use.yaml
agentic-rl:
	N_GPUS=$(N_GPUS) bash scripts/launch.sh agentic_rl configs/training/agentic_rl.yaml
distill:
	N_GPUS=$(N_GPUS) bash scripts/launch.sh distill configs/training/distill.yaml
adaptive-thinking:
	N_GPUS=$(N_GPUS) bash scripts/launch.sh adaptive_thinking configs/training/adaptive_thinking.yaml

# ============================================================
# 评测
# ============================================================
.PHONY: eval eval-ceval eval-cmmlu eval-gsm8k eval-humaneval eval-all
eval: eval-all
eval-ceval:
	$(PYTHON) scripts/evaluate.py --task ceval --backend local \
		--model_path checkpoints/sft/checkpoint-final \
		--tokenizer_path checkpoints/tokenizer --shots 5
eval-cmmlu:
	$(PYTHON) scripts/evaluate.py --task cmmlu --backend local \
		--model_path checkpoints/sft/checkpoint-final \
		--tokenizer_path checkpoints/tokenizer --shots 5
eval-gsm8k:
	$(PYTHON) scripts/evaluate.py --task gsm8k --backend local \
		--model_path checkpoints/sft/checkpoint-final \
		--tokenizer_path checkpoints/tokenizer --shots 8
eval-humaneval:
	$(PYTHON) scripts/evaluate.py --task humaneval --backend local \
		--model_path checkpoints/sft/checkpoint-final \
		--tokenizer_path checkpoints/tokenizer --shots 0
eval-all:
	$(PYTHON) scripts/evaluate.py \
		--task ceval --task cmmlu --task openbookqa --task gsm8k --task humaneval \
		--backend local \
		--model_path checkpoints/sft/checkpoint-final \
		--tokenizer_path checkpoints/tokenizer \
		--shots 5 --output_dir eval_results

# ============================================================
# 长文本（YaRN）
# ============================================================
.PHONY: needle needle-long
needle:
	$(PYTHON) scripts/run_needle.py \
		--model_path checkpoints/sft/checkpoint-final \
		--tokenizer_path checkpoints/tokenizer \
		--max_seq_len 65000 \
		--context_lengths 4000,16000,32000,65000 \
		--depths 0.0,0.25,0.5,0.75,1.0 \
		--n_repeats 3 \
		--output_dir needle_results/short
needle-long:
	$(PYTHON) scripts/run_needle.py \
		--model_path checkpoints/sft/checkpoint-final \
		--tokenizer_path checkpoints/tokenizer \
		--yarn_factor 16 \
		--max_seq_len 1048576 \
		--context_lengths 65000,131000,262000,524000,1048000 \
		--depths 0.0,0.5,1.0 \
		--n_repeats 2 \
		--output_dir needle_results/long

# ============================================================
# 服务
# ============================================================
.PHONY: serve serve-dev serve-yarn
serve:
	$(PYTHON) scripts/serve.py --config configs/inference/server.yaml
serve-dev:
	$(PYTHON) scripts/serve.py --config configs/inference/server.yaml \
		log_level=DEBUG log_request_payload=true
serve-yarn:
	$(PYTHON) scripts/serve.py --config configs/inference/server.yaml \
		yarn_factor=16 max_model_len=131072

# ============================================================
# 格式转换
# ============================================================
CKPT ?= checkpoints/sft/checkpoint-final
MCFG ?= configs/model/mini_2b.json
TOK  ?= checkpoints/tokenizer

.PHONY: export-hf export-vllm export-gguf export-ollama
export-hf:
	$(PYTHON) scripts/convert_model.py to-hf \
		--state_dict_dir $(CKPT) --model_config $(MCFG) \
		--tokenizer_dir $(TOK) --output_dir exports/hf
export-vllm:
	$(PYTHON) scripts/convert_model.py to-vllm \
		--state_dict_dir $(CKPT) --model_config $(MCFG) \
		--tokenizer_dir $(TOK) --output_dir exports/vllm
export-gguf:
	$(PYTHON) scripts/convert_model.py to-gguf \
		--state_dict_dir $(CKPT) --model_config $(MCFG) \
		--tokenizer_dir $(TOK) --output_dir exports/gguf \
		--quantization q4_K_M \
		--llama_cpp_dir $${LLAMA_CPP_DIR:-$(HOME)/llama.cpp}
export-ollama: export-gguf
	$(PYTHON) scripts/convert_model.py to-ollama \
		--gguf_path exports/gguf/model-q4_K_M.gguf \
		--output_dir exports/ollama \
		--model_name deepseek-v4-mini

# ============================================================
# Docker
# ============================================================
.PHONY: docker-build-train docker-build-serve docker-up docker-down
docker-build-train:
	docker build -f docker/Dockerfile.train -t deepseek-v4-mini:train .
docker-build-serve:
	docker build -f docker/Dockerfile.serve -t deepseek-v4-mini:serve .
docker-up:
	docker compose -f docker/docker-compose.yml up -d
docker-down:
	docker compose -f docker/docker-compose.yml down

# ============================================================
# 测试 / Lint / CI
# ============================================================
.PHONY: test test-fast test-integration lint format type-check ci

test:
	pytest tests/ -v --ignore=tests/integration

test-fast:
	pytest tests/ -v --ignore=tests/integration -m "not slow"

test-integration:
	pytest tests/integration -v -m integration

lint:
	ruff check src tests
	black --check src tests
	isort --check-only src tests

format:
	ruff check --fix src tests
	black src tests
	isort src tests

type-check:
	mypy src --ignore-missing-imports || true

ci: lint test test-integration
	@echo "✅ CI passed"

# ============================================================
# 清理
# ============================================================
.PHONY: clean clean-cache clean-checkpoints clean-all
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name '.pytest_cache' -exec rm -rf {} +
	find . -type d -name '.ruff_cache' -exec rm -rf {} +
	find . -type d -name '.mypy_cache' -exec rm -rf {} +
	find . -type d -name '*.egg-info' -exec rm -rf {} +
	rm -rf build/ dist/

clean-cache:
	rm -rf cache/

clean-checkpoints:
	rm -rf checkpoints/

clean-all: clean clean-cache clean-checkpoints
	rm -rf data/ logs/ eval_results/ needle_results/ exports/

# ============================================================
# 全流程（demo）
# ============================================================
.PHONY: full-pipeline
full-pipeline: tokenizer data-pretrain pretrain data-sft sft data-dpo rm dpo eval-all
	@echo "✅ Full pipeline complete!"

# ============================================================
# 帮助
# ============================================================
.PHONY: help
help:
	@echo "DeepSeek-V4 Mini Makefile targets:"
	@echo ""
	@echo "  Setup:"
	@echo "    install          - 基础安装"
	@echo "    install-all      - 全量安装（含 dev）"
	@echo "    install-dev      - 开发环境（含 pre-commit hook）"
	@echo ""
	@echo "  Data:"
	@echo "    data-pretrain    - 准备预训练数据"
	@echo "    data-sft         - 准备 SFT 数据"
	@echo "    data-dpo         - 准备 DPO 数据"
	@echo ""
	@echo "  Training (单卡 / 多卡设置 N_GPUS=8):"
	@echo "    tokenizer        - 训练 tokenizer"
	@echo "    pretrain         - 预训练"
	@echo "    sft              - SFT"
	@echo "    lora             - LoRA-SFT"
	@echo "    rm / dpo / ppo / grpo / cispo / tool-use / agentic-rl / distill / adaptive-thinking"
	@echo ""
	@echo "  Evaluation:"
	@echo "    eval-all         - 跑全部评测集"
	@echo "    eval-{ceval,cmmlu,gsm8k,humaneval}"
	@echo "    needle / needle-long  - Needle in a Haystack"
	@echo ""
	@echo "  Inference:"
	@echo "    serve            - 启动 OpenAI 兼容服务"
	@echo "    serve-yarn       - 启用 YaRN 长上下文"
	@echo "    export-{hf,vllm,gguf,ollama}"
	@echo ""
	@echo "  Docker:"
	@echo "    docker-build-train / docker-build-serve / docker-up"
	@echo ""
	@echo "  Dev:"
	@echo "    test / test-fast / test-integration"
	@echo "    lint / format / type-check / ci"
	@echo ""
	@echo "  全流程 demo: make full-pipeline"

.DEFAULT_GOAL := help
