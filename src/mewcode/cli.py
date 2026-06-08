"""MewCode CLI entry — config loading and TUI startup."""

import os
import sys

from mewcode.config import ConfigError, load


def main() -> None:
    if "--version" in sys.argv:
        from mewcode import __version__
        print(__version__)
        return

    cwd = os.getcwd()
    config_paths = [
        os.path.join(cwd, ".mewcode", "config.yaml"),
        os.path.join(os.path.expanduser("~"), ".mewcode", "config.yaml"),
    ]

    cfg = None
    err = None
    for path in config_paths:
        try:
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

    from mewcode.tui.app import MewCodeApp

    app = MewCodeApp(cfg.providers)
    app.run()


if __name__ == "__main__":
    main()
