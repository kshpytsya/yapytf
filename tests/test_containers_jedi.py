import textwrap

import jedi
import pytest

import yapytf._containers as cnt


prefix_code = textwrap.dedent("""
    import yapytf
    import yapytf._containers as cnt

    """)


@pytest.mark.parametrize(
    'code,ref',
    [
        (
            """
            x = cnt.DirectMutableListAccessor[int]([])
            x.
            """,
            dir(cnt.DirectMutableListAccessor[int]([]))
        ),
        (
            """
            class O(cnt.DirectMutableObjectAccessor):
                a: list
                b: str

            x = O({})
            x.
            """,
            ["a", "b"]
        ),
        (
            """
            class O(cnt.DirectMutableObjectAccessor):
                a: list
                b: str

            x = O({})
            x.a.
            """,
            dir(list)
        ),
        (
            """
            class O(cnt.DirectMutableObjectAccessor["O"]):
                a: list

            x = O({})
            with x as y:
                y.a.
            """,
            dir(list)
        ),
    ]
)
def test(code: str, ref: str) -> None:
    code = prefix_code + textwrap.dedent(code)
    lines = code.splitlines()
    script = jedi.Script(
        source=code,
        line=len(lines),
        column=len(lines[-1]),
        path="source.py",
    )

    def nounders(l):
        return (i for i in l if not i.startswith("_"))

    completions = set(nounders(i.name for i in script.completions()))
    expected_completions = set(nounders(ref))
    assert completions == expected_completions
