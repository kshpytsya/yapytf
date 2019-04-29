import textwrap as _textwrap
import typing as _typing


def dedent(s: str) -> str:
    return _textwrap.dedent(s).lstrip()


def format(
    template_str: str,
    *args: _typing.Any,
    **kw: _typing.Any
) -> str:
    return dedent(template_str).format(*args, **kw)
