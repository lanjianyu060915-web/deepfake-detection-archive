from __future__ import annotations

import sys


PERSISTENT_SWITCH = "--persistent-worker"


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == PERSISTENT_SWITCH:
        del sys.argv[1]
        from persistent_worker import main as persistent_main

        return int(persistent_main())

    from identity_search.cli import main as cli_main

    return int(cli_main())


if __name__ == "__main__":
    raise SystemExit(main())
