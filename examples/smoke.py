"""Smoke test — 端到端验证 nova 节点与 LLM 缓存命中。

用法：cd NovaCode && python examples/smoke.py [prompt]

节点至少需要 config.yaml 在 .novacode/config.yaml 或 ~/.novacode/config.yaml。
若第二个请求 cache_read > 0 则说明缓存策略生效。
"""

import asyncio
import sys
from pathlib import Path

from novacode.agent import Agent
from novacode.config import ConfigError, load
from novacode.conversation import Conversation
from novacode.llm import new_provider
from novacode.permission import Mode, Outcome
from novacode.permission.engine import new_engine
from novacode.tool import new_default_registry


async def main() -> None:
    import os

    config_paths = [
        os.path.join(os.getcwd(), ".novacode", "config.yaml"),
        os.path.join(os.path.expanduser("~"), ".novacode", "config.yaml"),
    ]
    cfg = None
    for path in config_paths:
        try:
            cfg = load(path)
        except (ConfigError, FileNotFoundError):
            continue
    if cfg is None or not cfg.providers:
        print("❌ 未找到有效配置。请创建 .novacode/config.yaml")
        sys.exit(1)

    provider_cfg = cfg.providers[0]
    print(f"📍 Provider: {provider_cfg.name} [{provider_cfg.model}]")
    print(f"📍 Protocol: {provider_cfg.protocol}")

    provider = new_provider(provider_cfg)
    registry = new_default_registry()

    # 构造权限引擎（BYPASS 模式跳过 Ask）
    cwd = str(Path.cwd().resolve())
    engine, _ = new_engine(cwd)

    prompt_text = sys.argv[1] if len(sys.argv) > 1 else "列出当前目录的文件"

    for round_idx in range(1, 3):
        conv = Conversation()
        conv.add_user(prompt_text)

        agent = Agent(provider, registry, "dev", engine)
        total_in = 0
        total_out = 0
        total_cache_write = 0
        total_cache_read = 0

        async for ev in agent.run(conv, Mode.BYPASS, asyncio.Event()):
            if ev.usage is not None:
                total_in += ev.usage.input
                total_out += ev.usage.output
                total_cache_write += ev.usage.cache_write
                total_cache_read += ev.usage.cache_read
            if ev.approval is not None:
                # BYPASS 下不应出现 Ask，防御性处理
                if not ev.approval.respond.done():
                    ev.approval.respond.set_result(Outcome.DENY_ONCE)
                continue

        print(
            f"  Round {round_idx}: "
            f"input={total_in}, output={total_out}, "
            f"cache_write={total_cache_write}, cache_read={total_cache_read}"
        )

        msgs = conv.messages()
        for m in msgs:
            if m.role == "assistant" and m.content.strip():
                print(f"  Response: {m.content[:200]}...")
                break

    if round_idx >= 2 and total_cache_read > 0:
        print("✅ 缓存命中！稳定前缀被复用。")
    else:
        print("⚠️  缓存未命中或仅运行单轮（正常——首轮需创建缓存）。")


if __name__ == "__main__":
    asyncio.run(main())
