"""Module entrypoint for :mod:`~fmp_docs_compiler`.

Purpose:
    Support running the CLI with ``python -m fmp_docs_compiler``.

Design:
    The module simply imports and invokes the Typer application.

Attributes:
    None.

Examples:
    ::
        >>> callable(main)
        True
"""

from .cli import app


def main() -> None:
    """Run the package CLI.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.

    Examples:
        ::
            >>> callable(main)
            True
    """
    app()


if __name__ == "__main__":
    main()
