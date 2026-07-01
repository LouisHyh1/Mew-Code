"""NovaCode CLI entry — config loading and TUI startup."""

import os
import sys
from pathlib import Path

from novacode import __version__


def main() -> None:
    if "--version" in sys.argv:
        print(__version__)
        return

    cwd = os.getcwd()
    config_paths = [
        os.path.join(cwd, ".novacode", "config.yaml"),
        os.path.join(os.path.expanduser("~"), ".novacode", "config.yaml"),
    ]

    cfg = None
    err = None
    for path in config_paths:
        try:
            from novacode.config import ConfigError, load

            cfg = load(path)
            break
        except (ConfigError, FileNotFoundError) as e:
            err = e
            continue

    if cfg is None:
        if err:
            print(f"Config error: {err}", file=sys.stderr)
        else:
            searched = "\n  - ".join(config_paths)
            print(f"No config file found. Searched:\n  - {searched}", file=sys.stderr)
        sys.exit(1)

    from novacode.permission.engine import new_engine
    from novacode.tool import new_default_registry
    from novacode.tui.app import NovaCodeApp
    from novacode.tui.driver import NoAltScreenDriver

    # 构造权限引擎
    root = str(Path.cwd().resolve())
    engine, engine_err = new_engine(root)
    if engine_err is not None:
        print(f"权限引擎降级: {engine_err}", file=sys.stderr)

    registry = new_default_registry()
    app = NovaCodeApp(
        cfg.providers,
        registry,
        __version__,
        driver_class=NoAltScreenDriver,
        engine=engine,
    )
    app.run()


if __name__ == "__main__":
    main()
