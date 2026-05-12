#!/usr/bin/env python
"""服务端启动入口。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from deepseek_v4.inference.server.server import cli

if __name__ == "__main__":
    cli()
