"""Smoke test — 端到端验证 nova 节点与 LLM 缓存命中。

用法：cd NovaCode && python examples/smoke.py [prompt]

节点至少需要 config.yaml 在 .novacode/config.yaml 或 ~/.novacode/config.yaml。
若第二个请求 cache_read > 0 则说明缓存策略生效。
"""

import asyncio
import sys

from novacode.agent import Agent, Mode
from novacode.config import ConfigError, load
from novacode.conversation import Conversation
from novacode.llm import new_provider
from novacode.tool import new_default_registry


async def main() -> None:
    # 1. 加载配置
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

    # 2. 两轮请求，观察缓存命中
    prompt_text = sys.argv[1] if len(sys.argv) > 1 else "列出当前目录的文件"

    for round_idx in range(1, 3):
        conv = Conversation()
        conv.add_user(prompt_text)

        agent = Agent(provider, registry, "dev")
        total_in = 0
        total_out = 0
        total_cache_write = 0
        total_cache_read = 0

        async for ev in agent.run(conv, Mode.NORMAL, asyncio.Event()):
            if ev.usage is not None:
                total_in += ev.usage.input
                total_out += ev.usage.output
                total_cache_write += ev.usage.cache_write
                total_cache_read += ev.usage.cache_read

        print(
            f"  Round {round_idx}: "
            f"input={total_in}, output={total_out}, "
            f"cache_write={total_cache_write}, cache_read={total_cache_read}"
        )

        # 打印模型回复
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
