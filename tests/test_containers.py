import copy

import pytest

import yapytf._containers as cnt


@pytest.fixture
def cmp(inp, out):
    if out is Ellipsis:
        out = copy.deepcopy(inp)

    def f(actual_out):
        assert actual_out == out

    return f


@pytest.mark.parametrize(
    'inp,out',
    [
        ({}, {}),
        ({}, ...)
    ]
)
def test_null(inp, cmp):
    cmp(inp)


# TODO
# test "generic was not parametrized" assertion


def test_DirectMutableDictAccessor_subclass():
    class D(cnt.DirectMutableDictAccessor[str, "cnt.DirectMutableDictAccessor[int, int]"]):
        _auto_create = True

    d = {}
    x = D(d)
    x["a"][1] = 2

    assert d == {"a": {1: 2}}


@pytest.mark.parametrize(
    'inp,out,tp,value',
    [
        ([], [1], int, 1),
        ([], [""], str, ""),
        ([], [[]], list, []),
    ]
)
def test_direct_append(inp, cmp, tp, value):
    x = cnt.DirectMutableListAccessor[tp](inp)
    x.append(value)
    cmp(inp)


@pytest.mark.parametrize(
    'inp,tp,value',
    [
        ([], str, 1),
        ([], int, ""),
    ]
)
def test_direct_append_wrong_type(inp, tp, value):
    with pytest.raises(TypeError):
        x = cnt.DirectMutableListAccessor[tp](inp)
        x.append(value)


@pytest.mark.parametrize(
    'inp,tkey,key,tvalue,value',
    [
        ({}, str, 1, str, ""),
        ({}, int, "", str, ""),
        ({}, int, 1, str, 1),
        ({}, int, 1, int, ""),
        ({}, "int", 1, int, ""),
    ]
)
def test_direct_set_wrong_type(inp, tkey, key, tvalue, value):
    with pytest.raises(TypeError):
        x = cnt.DirectMutableListAccessor[tkey, tvalue](inp)
        x[key] = value


def test_mutable_namespace():
    class O1(cnt.DirectMutableObjectAccessor):
        a: cnt.DictItemMutableListAccessor[int]

    class O2(cnt.DirectMutableObjectAccessor):
        b: cnt.DictItemMutableDictAccessor[int, str]

    class Ns(cnt.MutableNamespace):
        _paths = {
            'v': ('x', 'y')
        }
        u: O1
        v: O2

    d = {}
    ns = Ns(d)
    ns.u.a.append(1)
    ns.v.b[1] = "Z"
    assert d == {"a": [1], "x": {"y": {"b": {1: "Z"}}}}
