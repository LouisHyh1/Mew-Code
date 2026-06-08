"""Configuration data types and YAML loader."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml


class ConfigError(Exception):
    pass


@dataclass
class ProviderConfig:
    name: str
    protocol: Literal["anthropic", "openai"]
    api_key: str
    model: str
    base_url: str | None = None
    thinking: bool = False


@dataclass
class Config:
    providers: list[ProviderConfig] = field(default_factory=list)


def load(path: str) -> Config:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"config file not found: {path}")

    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ConfigError(f"YAML parse error in {path}: {e}") from e

    if raw is None or "providers" not in raw:
        raise ConfigError(f"Config must contain a 'providers' key with at least one entry")

    providers_raw = raw["providers"]
    if not isinstance(providers_raw, list) or len(providers_raw) == 0:
        raise ConfigError("'providers' must be a non-empty list")

    providers: list[ProviderConfig] = []
    for i, entry in enumerate(providers_raw):
        prefix = f"providers[{i}]"
        if not isinstance(entry, dict):
            raise ConfigError(f"{prefix}: must be a mapping")
        _validate_provider(entry, prefix)
        providers.append(ProviderConfig(
            name=entry["name"],
            protocol=entry["protocol"],
            api_key=entry["api_key"],
            model=entry["model"],
            base_url=entry.get("base_url"),
            thinking=entry.get("thinking", False),
        ))

    return Config(providers=providers)


def _validate_provider(entry: dict, prefix: str) -> None:
    for field in ("name", "protocol", "api_key", "model"):
        if field not in entry or entry[field] is None:
            raise ConfigError(f"{prefix}.{field} cannot be empty")
    if entry["protocol"] not in ("anthropic", "openai"):
        raise ConfigError(
            f"{prefix}.protocol must be 'anthropic' or 'openai', got '{entry['protocol']}'"
        )
