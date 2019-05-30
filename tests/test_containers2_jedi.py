import sys
import textwrap
from typing import Set

import jedi
import pytest

import yapytf._containers2 as cnt


prefix_code = textwrap.dedent("""
    import yapytf
    import yapytf._containers2 as cnt

    """)


# new in Python 3.7, not yet added to typeshed
STR_IGNORE = {"isascii"}


@pytest.mark.parametrize(
    'code,ref,ignore',
    [
        (
            """
            x = dict()
            x.
            """,
            dir(dict),
            set()
        ),
        (
            """
            x = ""
            x.
            """,
            dir(str),
            STR_IGNORE
        ),
        (
            """
            from typing import Mapping
            x: Mapping[int, str]
            x[0].
            """,
            dir(str),
            STR_IGNORE
        ),
        (
            """
            cnt.
            """,
            dir(cnt),
            set()
        ),
        (
            """
            class R(cnt.Record):
                f1 = cnt.ConstRecordField[str]()

            r = R()
            r.
            """,
            ["f1"],
            set()
        ),
        (
            """
            class R(cnt.Record):
                f1 = cnt.ConstRecordField[str]()

            r = R()
            r.f1.
            """,
            dir(str),
            STR_IGNORE
        ),
        (
            """
            class R(cnt.Record):
                f1 = cnt.ConstRecordField[cnt.PlainDict[int, str]]()

            r = R()
            r.f1[0].
            """,
            dir(str),
            STR_IGNORE
        ),
        (
            """
            class R(cnt.Record):
                f1 = cnt.ConstRecordField[cnt.PlainDict[int, str]]()

            r = R()
            with r as x:
                x.
            """,
            ["f1"],
            set()
        ),
        (
            """
            class R(cnt.Record):
                f1 = cnt.ConstRecordField[cnt.PlainDict[int, str]]()

            r = R()
            r.f1.
            """,
            dir(dict),
            {"copy", "fromkeys"}
        ),
    ]
)
def test(code: str, ref: str, ignore: Set[str]) -> None:
    code = prefix_code + textwrap.dedent(code)
    lines = code.splitlines()
    script = jedi.Script(
        source=code,
        line=len(lines),
        column=len(lines[-1]),
        path="source.py",
        # sys_path=sys.path
    )

    def nounders(l):
        return (i for i in l if not i.startswith("_"))

    completions = set(nounders(i.name for i in script.completions())) - ignore
    expected_completions = set(nounders(ref)) - ignore
    assert completions == expected_completions
