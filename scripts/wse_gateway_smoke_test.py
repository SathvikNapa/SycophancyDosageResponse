"""
wse_gateway_smoke_test.py
--------------------------
Minimal connectivity check for the JHU WSE AI Gateway routing added to
src/sycophancy/generator.py (see config.USE_WSE_GATEWAY). Exercises the
actual project code path (ResponseGenerator + config.MODELS), not a
standalone request, so a pass here means the real pipeline works through
the gateway, not just that the gateway itself is reachable.

Requires in .env:
    USE_WSE_GATEWAY=true
    WSE_GATEWAY_KEY=jhu_live_sk_...   (from a WSE gateway project you created
                                        at gateway.engineering.jhu.edu)
and that the models below are enabled on both the project and the key
(gateway UI: Project -> Models, and Project -> Keys -> that key's allowlist).

Usage:
    uv run python scripts/wse_gateway_smoke_test.py
"""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv

load_dotenv()

from sycophancy import config  # noqa: E402  (import after load_dotenv so env vars are read fresh)
from sycophancy.generator import ResponseGenerator  # noqa: E402

MODELS_TO_TEST = ["GPT5_4Nano", "ClaudeHaiku"]


async def check_one(gen: ResponseGenerator, model_key: str) -> None:
    messages = [{"role": "user", "content": "Reply with exactly: OK"}]
    try:
        reply = await gen.acomplete(messages, model_key, timeout_s=30)
        status = "OK" if reply and "OK" in reply.upper() else "UNEXPECTED REPLY"
        print(f"  {model_key:15s} [{config.MODELS[model_key]}] -> {status}: {reply!r}")
    except Exception as e:
        print(f"  {model_key:15s} [{config.MODELS[model_key]}] -> FAILED: {type(e).__name__}: {e}")


async def main() -> None:
    print(f"USE_WSE_GATEWAY = {config.USE_WSE_GATEWAY}")
    print(f"WSE_GATEWAY_BASE = {config.WSE_GATEWAY_BASE}")
    print(f"WSE_GATEWAY_KEY set = {bool(config.WSE_GATEWAY_KEY)}")
    if not config.USE_WSE_GATEWAY:
        print("\nUSE_WSE_GATEWAY is not enabled in .env -- nothing to test.")
        print("Set USE_WSE_GATEWAY=true and WSE_GATEWAY_KEY=... to run this.")
        return
    if not config.WSE_GATEWAY_KEY:
        print("\nWSE_GATEWAY_KEY is not set in .env -- nothing to test.")
        return

    print()
    gen = ResponseGenerator(temperature=0)
    for model_key in MODELS_TO_TEST:
        await check_one(gen, model_key)


if __name__ == "__main__":
    asyncio.run(main())
