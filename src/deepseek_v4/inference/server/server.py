"""
uvicorn 启动入口。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from deepseek_v4.inference.server.app import build_app
from deepseek_v4.inference.server.config import ServerConfig
from deepseek_v4.utils.config import parse_overrides
from deepseek_v4.utils.logger import get_logger

logger = get_logger("server")


def cli():
    parser = argparse.ArgumentParser("DeepSeek-V4 Mini OpenAI-compatible server")
    parser.add_argument("--config", required=False, default=None, help="YAML 配置文件路径")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--yarn_factor", type=float, default=None)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--workers", type=int, default=1, help="uvicorn workers (推荐 1，模型常驻进程)")
    parser.add_argument("overrides", nargs="*", help="key=value overrides")
    args = parser.parse_args()

    if args.config:
        cfg = ServerConfig.from_yaml(args.config)
    else:
        cfg = ServerConfig()

    if args.host: cfg.host = args.host
    if args.port: cfg.port = args.port
    if args.model_path: cfg.model_path = args.model_path
    if args.tokenizer_path: cfg.tokenizer_path = args.tokenizer_path
    if args.model_name: cfg.model_name = args.model_name
    if args.dtype: cfg.dtype = args.dtype
    if args.max_model_len: cfg.max_model_len = args.max_model_len
    if args.yarn_factor is not None: cfg.yarn_factor = args.yarn_factor
    if args.api_key: cfg.api_key = args.api_key

    if args.overrides:
        overrides = parse_overrides(args.overrides)
        cfg = cfg.update(overrides)

    logger.info(f"[server] starting on {cfg.host}:{cfg.port}, model={cfg.model_path}")

    import uvicorn
    app = build_app(cfg)
    uvicorn.run(
        app, host=cfg.host, port=cfg.port,
        log_level=cfg.log_level.lower(),
        workers=args.workers,
    )


if __name__ == "__main__":
    cli()
