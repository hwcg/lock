# 贡献指南

## 开发环境

```bash
git clone <repo>
cd deepseek-v4-mini
pip install -e ".[all]"
pre-commit install
```

## 提 PR 前检查

```
make lint          # ruff + black --check
make format        # 自动格式化
make test          # 单测
make test-fast     # 跳过 slow
```

## 代码规范

- 遵循 [PEP 8](https://peps.python.org/pep-0008/) + Black（line-length 110）
- 公共 API 必须有 docstring
- 新功能必须配套单测
- 类名 `PascalCase`，函数 `snake_case`，常量 `UPPER_SNAKE`

## 提交信息

```
<type>: <subject>

<body (optional)>

<footer (optional)>
type` ∈ `feat / fix / docs / style / refactor / test / chore / perf
```

例：

```
feat: add CISPO trainer with IS-weight clipping

Implements MiniMax M1 style CISPO with stop_grad importance weights.
Tests added in tests/test_grpo_cispo.py.
```

## 分支策略

- `main` — 受保护分支，只接 PR
- `feature/<topic>` — 开发新功能
- `fix/<issue>` — 修 bug
- `release/v0.x` — 发版分支

## 报 bug

请提供：

1. 复现命令
2. 完整错误栈
3. 环境信息：`python --version` / `pip freeze` / GPU 型号 / CUDA 版本
4. 最小复现代码
