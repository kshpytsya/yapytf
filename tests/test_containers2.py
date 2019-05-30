import copy
import typing as tp

import pytest

import yapytf._containers2 as cnt


@pytest.yield_fixture
def cmp(inp, out):
    if out is Ellipsis:
        out = copy.deepcopy(inp)

    yield

    assert inp == out


@pytest.mark.parametrize(
    'inp,out',
    [
        ({}, {}),
        ({}, ...)
    ]
)
def test_nop(cmp):
    pass


@pytest.mark.parametrize(
    'bad_container',
    [
        [],
        set(),
        1,
        ""
    ]
)
def test_Record_bad_container(bad_container):
    with pytest.raises(TypeError):
        cnt.Record(bad_container)


@pytest.mark.parametrize(
    'inp,out',
    [
        ({"f1": 1, "f2_": "2", "f3": "a", "f4": {"f4a": {"f4b": 4}}}, ...)
    ]
)
def test_ConstRecordField(inp, cmp):
    class Rec2(cnt.Record):
        f1 = cnt.ConstRecordField[str]()

    with pytest.raises(RuntimeError):
        class BadRecord(cnt.Record):
            # missing generic arg
            f1 = cnt.ConstRecordField()

    class IntField(cnt.ConstRecordField[int]):
        pass

    class Rec1(cnt.Record):
        f1 = IntField()

    class Rec(cnt.Record):
        f1 = IntField()
        f2 = cnt.ConstRecordField[str](name="f2_")
        f3 = IntField()
        f4 = IntField(path=["f4", "f4a"], name="f4b")
        f5 = IntField()

    r = Rec(inp)

    assert r.f1 == 1
    assert r.f2 == "2"
    assert r.f4 == 4

    with pytest.raises(NotImplementedError):
        r.f1 = 1

    with pytest.raises(NotImplementedError):
        del r.f1

    with pytest.raises(TypeError):
        r.f3

    with pytest.raises(AttributeError):
        r.f5

    r2 = Rec(inp, path=["nonexistent"])

    with pytest.raises(KeyError):
        r2.f1


@pytest.mark.parametrize(
    'inp,out',
    [
        (
            {"f1": 1, "f3": []},
            {"f1": 2, "f2": "a", "x": {"f1": 3}})
    ]
)
def test_PlainRecordField(inp, cmp):
    class Rec(cnt.Record):
        f1 = cnt.PlainRecordField[int]()
        f2 = cnt.PlainRecordField[str]()
        f3 = cnt.PlainRecordField[str]()

    r = Rec(inp)

    assert r.f1 == 1
    with pytest.raises(AttributeError):
        r.f2

    r2 = Rec(inp)

    r.f1 = 2
    r.f2 = "a"
    del r.f3

    assert r2.f1 == 2
    assert r2.f2 == "a"
    with pytest.raises(AttributeError):
        r2.f3

    r3 = Rec(inp, path=["x"])
    r3.f1 = 3

    with pytest.raises(TypeError):
        r3.f1 = ""


@pytest.mark.parametrize(
    'inp,out',
    [
        (
            {
                "int2str": {1: "A", 2: 3}
            },
            ...
        )
    ]
)
def test_ConstRecordField_ConstDict(inp, cmp):
    class R(cnt.Record):
        int2str = cnt.ConstRecordField[cnt.ConstDict[int, str]]()

    r = R(inp)

    assert len(r.int2str) == 2
    assert set(r.int2str) == {1, 2}

    assert r.int2str[1] == "A"

    with pytest.raises(TypeError):
        r.int2str[2]

    with pytest.raises(KeyError):
        r.int2str[3]

    with pytest.raises(TypeError):
        r.int2str["x"]


@pytest.mark.parametrize(
    'inp,out',
    [
        (
            {
                "int2str": {1: "A", "2": 3}
            },
            ...
        )
    ]
)
def test_ConstDict_bad_key(inp, cmp):
    class R(cnt.Record):
        int2str = cnt.ConstRecordField[cnt.ConstDict[int, str]]()

    r = R(inp)

    with pytest.raises(TypeError):
        list(r.int2str)


@pytest.mark.parametrize(
    'inp,out',
    [
        (
            {
                "str2rec":
                {
                    "a":
                    {
                        "s": "S",
                        "i": 1,
                    }
                }
            },
            ...
        )
    ]
)
def test_ConstDict_Record(inp, cmp):
    class R2(cnt.Record):
        s = cnt.ConstRecordField[str]()
        i = cnt.ConstRecordField[int]()

    class R(cnt.Record):
        str2rec = cnt.ConstRecordField[cnt.ConstDict[str, R2]]()

    r = R(inp)
    assert len(r.str2rec) == 1
    list(r.str2rec)

    assert r.str2rec["a"].s == "S"
    assert r.str2rec["a"].i == 1

    with pytest.raises(NotImplementedError):
        del r.str2rec

    with pytest.raises(TypeError):
        del r.str2rec["a"]

    with pytest.raises(NotImplementedError):
        del r.str2rec["a"].s

    with pytest.raises(NotImplementedError):
        del r.str2rec["a"].i



@pytest.mark.parametrize(
    'inp,out',
    [
        (
            {
                "f1":
                {
                    "a":
                    {
                        1: b"A",
                        2: b"B"
                    }
                },
                "f2":
                {
                    "u": {10: {b"v": 1.5}}
                }
            },
            ...
        )
    ]
)
def test_ConstDict_nested_plain(inp, cmp):
    class R(cnt.Record):
        f1 = cnt.ConstRecordField[cnt.ConstDict[str, cnt.ConstDict[int, bytes]]]()
        f2 = cnt.ConstRecordField[cnt.ConstDict[str, cnt.ConstDict[int, cnt.ConstDict[bytes, float]]]]()

    r = R(inp)
    assert len(r.f1) == 1
    list(r.f1)

    a = r.f1["a"]
    assert len(a) == 2
    assert set(a) == {1, 2}
    assert a[1] == b"A"
    assert a[2] == b"B"

    assert len(r.f2) == 1
    assert r.f2["u"][10][b"v"] == 1.5


@pytest.mark.parametrize(
    'inp,out',
    [
        (
            {
                "f1": [1, 2, 3],
                "f2": ["a", "b"],
                "f3": ["a", 1, "b", 2],
            },
            ...
        )
    ]
)
def test_ConstDict_nested_plain(inp, cmp):
    class R(cnt.Record):
        f1 = cnt.ConstRecordField[cnt.ConstList[int]]()
        f2 = cnt.ConstRecordField[cnt.ConstList[str]]()
        f3 = cnt.ConstRecordField[cnt.ConstList[str]]()

    r = R(inp)
    assert len(r.f1) == 3
    assert len(r.f2) == 2
    assert len(r.f3) == 4

    assert list(r.f1) == [1, 2, 3]
    assert list(r.f1[1:]) == [2, 3]
    assert list(r.f2) == ["a", "b"]
    assert list(r.f3[0::2]) == ["a", "b"]

    with pytest.raises(TypeError):
        r.f3[:]


@pytest.mark.parametrize(
    'inp,out',
    [
        (
            {
                "a":
                {
                    "b":
                    {
                        "f1_":
                        {
                            "a": [1],
                            "ab": [1, 2],
                            "abc": [1, 2, 3]
                        }

                    }
                }
            },
            ...
        )
    ]
)
def test_ConstDict_ConstList(inp, cmp):
    class R(cnt.Record):
        f1 = cnt.ConstRecordField[cnt.ConstDict[str, cnt.ConstList[int]]](name="f1_", path=["a", "b"])

    r = R(inp)
    assert len(r.f1) == 3

    assert set(r.f1) == {"a", "ab", "abc"}
    assert list(r.f1["a"]) == [1]
    assert list(r.f1["ab"]) == [1, 2]
    assert list(r.f1["abc"][::2]) == [1, 3]


@pytest.mark.parametrize(
    'inp,out',
    [
        (
            {
                "fa":
                {
                    "f1": 1
                }
            },
            {
                "fa":
                {
                    "f1": 1
                },
                "fc":
                {
                    "f1": 2
                }
            }
        )
    ]
)
def test_RecordField_RecordField(inp, cmp):
    class R2(cnt.Record):
        f1 = cnt.PlainRecordField[int]()

    class R(cnt.Record):
        fa = cnt.PlainRecordField[R2]()
        fb = cnt.PlainRecordField[R2]()
        fc = cnt.PlainRecordField[R2]()

    r = R(inp)

    assert r.fa.f1 == 1
    r.fb

    with pytest.raises(AttributeError):
        r.fb.f1

    r.fc.f1 = 2


@pytest.mark.parametrize(
    'inp,out',
    [
        (
            {
                "f1":
                {
                    "a": 1,
                    "b": 2
                },
                "f1bad":
                {
                    1: "a",
                    "2": "b"
                }
            },
            {
                "f1":
                {
                    "a": 1,
                    "c": 3
                },
                "f2":
                {
                    1: "a"
                }
            }
        )
    ]
)
def test_RecordField_PlainDict(inp, cmp):
    class R(cnt.Record):
        f1 = cnt.PlainRecordField[cnt.PlainDict[str, int]]()
        f1bad = cnt.PlainRecordField[cnt.PlainDict[str, int]]()
        f2 = cnt.PlainRecordField[cnt.PlainDict[int, str]]()

    r = R(inp)

    assert len(r.f1) == 2
    assert sorted(r.f1.items()) == [("a", 1), ("b", 2)]
    del r.f1["b"]

    with pytest.raises(TypeError):
        list(r.f1bad)

    with pytest.raises(TypeError):
        r.f1bad["2"]

    del r.f1bad

    with pytest.raises(KeyError):
        del r.f1["b"]

    r.f1["c"] = 3

    with pytest.raises(AttributeError):
        len(r.f2)

    with pytest.raises(AttributeError):
        list(r.f2)

    r.f2[1] = "a"


@pytest.mark.parametrize(
    'inp,out',
    [
        (
            {
                "f1": [1, 11]
            },
            {
                "f1": [10, 2, 3]
            }
        )
    ]
)
def test_List(inp, cmp):
    class R(cnt.Record):
        f1 = cnt.RecordField[cnt.PlainList[int], tp.Iterable[int]]()

    r = R(inp)
    assert len(r.f1) == 2

    assert set(r.f1) == {1, 11}

    del r.f1[1]
    r.f1[0] = 10
    r.f1.append(2)
    r.f1.append(3)


@pytest.mark.parametrize(
    'inp,out',
    [
        (
            {
                "f1":
                {
                    "a": [1],
                    "ab": [1, 2],
                    "abc": [1, 2, 3]
                }
            },
            {
                "f1":
                {
                    "a": [10],
                    "ab": [2, 1],
                    "abc": [1, 3],
                    "x1": [1, 2, 3]
                }
            }
        )
    ]
)
def test_Dict_List(inp, cmp):
    class R(cnt.Record):
        f1 = cnt.PlainRecordField[cnt.Dict[str, cnt.PlainList[int], tp.Iterable[int]]]()

    r = R(inp)
    assert len(r.f1) == 3

    assert set(r.f1) == {"a", "ab", "abc"}
    assert list(r.f1["a"]) == [1]
    assert list(r.f1["ab"]) == [1, 2]
    assert list(r.f1["abc"][::2]) == [1, 3]

    r.f1["a"][0] = 10
    r.f1["ab"] = [2, 1]
    r.f1["x1"] = r.f1["abc"]
    del r.f1["abc"][1]


@pytest.mark.parametrize(
    'inp,out',
    [
        (
            {
                "f1":
                [
                    {"a": 2}
                ]
            },
            {
                "f1":
                [
                ],
                "f2":
                [
                    {"a": 20, "b": 30},
                    {"c": 40},
                    {"a": 2}
                ]
            }
        )
    ]
)
def test_List_Dict(inp, cmp):
    D1 = cnt.PlainDict[str, int]
    D1x = tp.Mapping[str, int]
    T1 = cnt.RecordField[cnt.List[D1, D1x], tp.Iterable[tp.Union[D1, D1x]]]

    class R(cnt.Record):
        f1 = T1()
        f2 = T1()

    r = R(inp)

    assert len(r.f1) == 1
    assert r.f1[0] == {"a": 2}

    with pytest.raises(AttributeError):
        len(r.f2)

    r.f2.extend(r.f1)
    r.f2.append(r.f1[0])
    r.f1.clear()
    r.f2.insert(1, {"c": 4})
    r.f2[1]["c"] = 40
    with r.f2[0] as x:
        x["b"] = 30
        x["a"] *= 10

    with pytest.raises(AttributeError):
        len(x)


# def test_DirectMutableDictAccessor_subclass():
#     class D(cnt.DirectMutableDictAccessor[str, "cnt.DirectMutableDictAccessor[int, int]"]):
#         _auto_create = True

#     d = {}
#     x = D(d)
#     x["a"][1] = 2

#     assert d == {"a": {1: 2}}


# @pytest.mark.parametrize(
#     'inp,out,tp,value',
#     [
#         ([], [1], int, 1),
#         ([], [""], str, ""),
#         ([], [[]], list, []),
#     ]
# )
# def test_direct_append(inp, cmp, tp, value):
#     x = cnt.DirectMutableListAccessor[tp](inp)
#     x.append(value)
#     cmp(inp)


# @pytest.mark.parametrize(
#     'inp,tp,value',
#     [
#         ([], str, 1),
#         ([], int, ""),
#     ]
# )
# def test_direct_append_wrong_type(inp, tp, value):
#     with pytest.raises(TypeError):
#         x = cnt.DirectMutableListAccessor[tp](inp)
#         x.append(value)


# @pytest.mark.parametrize(
#     'inp,tkey,key,tvalue,value',
#     [
#         ({}, str, 1, str, ""),
#         ({}, int, "", str, ""),
#         ({}, int, 1, str, 1),
#         ({}, int, 1, int, ""),
#         ({}, "int", 1, int, ""),
#     ]
# )
# def test_direct_set_wrong_type(inp, tkey, key, tvalue, value):
#     with pytest.raises(TypeError):
#         x = cnt.DirectMutableListAccessor[tkey, tvalue](inp)
#         x[key] = value


# def test_mutable_namespace():
#     class O1(cnt.DirectMutableObjectAccessor):
#         a: cnt.DictItemMutableListAccessor[int]

#     class O2(cnt.DirectMutableObjectAccessor):
#         b: cnt.DictItemMutableDictAccessor[int, str]

#     class Ns(cnt.MutableNamespace):
#         _paths = {
#             'v': ('x', 'y')
#         }
#         u: O1
#         v: O2

#     d = {}
#     ns = Ns(d)
#     ns.u.a.append(1)
#     ns.v.b[1] = "Z"
#     assert d == {"a": [1], "x": {"y": {"b": {1: "Z"}}}}
