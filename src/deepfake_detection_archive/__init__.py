"""Reproducible offline AIGC audio and video detection."""


def main() -> int:
    from .cli import main as cli_main  # noqa: PLC0415 -- keep stdlib CLI lightweight

    return cli_main()
