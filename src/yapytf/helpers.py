import textwrap
from typing import Any


def format(template_str: str, *args: Any, **kw: Any) -> str:
    return textwrap.dedent(template_str).lstrip().format(*args, **kw)
