"""
FastAPI 应用工厂。
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from deepseek_v4.inference.server.config import ServerConfig
from deepseek_v4.inference.server.engine import ServerEngine
from deepseek_v4.inference.server.routes import build_router
from deepseek_v4.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


def build_app(config: ServerConfig, engine: Optional[ServerEngine] = None) -> FastAPI:
    setup_logging(level=config.log_level)
    engine = engine or ServerEngine(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            engine.start()
            yield
        finally:
            engine.stop()

    app = FastAPI(
        title="DeepSeek-V4-Mini Server",
        version="0.1.0",
        description="OpenAI-compatible server for DeepSeek-V4-Mini.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    router = build_router(engine=engine, config=config)
    app.include_router(router)

    @app.get("/")
    async def root():
        return {"name": config.model_name, "status": "ok"}

    return app


def create_app_from_config(config_path: str) -> FastAPI:
    cfg = ServerConfig.from_yaml(config_path)
    return build_app(cfg)
