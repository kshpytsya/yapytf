import json
import typing as _tp

from . import JsonType
from . import _containers as _cnt


def dict_path_rw(
    d: JsonType,
    path: _tp.Iterable[str]
) -> _tp.Dict[str, _tp.Any]:
    for i in path:
        d = d.setdefault(i, {})

    return d


def dict_path_ro(
    d: JsonType,
    path: _tp.Iterable[str]
) -> _tp.Dict[str, _tp.Any]:
    for i in path:
        d = d.get(i, {})

    return d


class Locals(_tp.MutableMapping[str, _tp.Any]):
    def __init__(self, data: _tp.Dict[str, _tp.Any]) -> None:
        self._data = dict_path_rw(data, ["tf", "locals"])

    def __delitem__(self, key: str) -> None:
        self._data.pop(key)

    def __getitem__(self, key: str) -> _tp.Any:
        return self._data[key]

    def __iter__(self) -> _tp.Iterator[str]:
        return self._data.__iter__()

    def __len__(self) -> int:
        return len(self._data)

    def __setitem__(self, key: str, value: _tp.Any) -> None:
        # note: this does not preclude further data manipulations from
        # making the value non-json-serializable, yet this is
        # a test to catch most straighforward mistakes
        json.dumps(value)
        self._data[key] = value


@_tp.runtime_checkable
class Extras(_tp.Protocol):
    # regretfully, typing is not yet powerful enough to express
    # requirement that the number of *args must match the number
    # of args in Callable signature. Also, we want a true "json serializable"
    # type instead of Any

    def func(self, func: _tp.Callable[..., _tp.Any], *args: _tp.Any) -> str:
        ...


class _DictWithDefault(_cnt.PlainDict[_cnt.Tkey, _cnt.Tval]):
    @property
    def default(self) -> _cnt.Tval:
        return self[_tp.cast(_cnt.Tkey, "")]

    @default.setter
    def default(self, val: _cnt.Tval) -> None:
        self[_tp.cast(_cnt.Tkey, "")] = val

    @default.deleter
    def default(self) -> None:
        del self[_tp.cast(_cnt.Tkey, "")]


class _ConstDictWithX(_cnt.ConstDict[_cnt.Tkey, _cnt.Tval]):
    @property
    def x(self) -> _cnt.Tval:
        return self[_tp.cast(_cnt.Tkey, None)]


class _SolitaryZerothAsXMixin(_tp.Sequence[_cnt.Tval]):
    @property
    def x(self) -> _cnt.Tval:
        assert len(self) == 1
        return self[0]
