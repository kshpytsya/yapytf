import json
from typing import Any, Dict, Iterable, Iterator, MutableMapping

from . import JsonType


def dict_path_rw(
    d: JsonType,
    path: Iterable[str]
) -> Dict[str, Any]:
    for i in path:
        d = d.setdefault(i, {})

    return d


def dict_path_ro(
    d: JsonType,
    path: Iterable[str]
) -> Dict[str, Any]:
    for i in path:
        d = d.get(i, {})

    return d


class Locals(MutableMapping[str, Any]):
    def __init__(self, data: Dict[str, Any]) -> None:
        self._data = dict_path_rw(data, ["tf", "locals"])

    def __delitem__(self, key: str) -> None:
        self._data.pop(key)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return self._data.__iter__()

    def __len__(self) -> int:
        return len(self._data)

    def __setitem__(self, key: str, value: Any) -> None:
        # note: this does not preclude further data manipulations from
        # making the value non-json-serializable, yet this is
        # a test to catch most straighforward mistakes
        json.dumps(value)
        self._data[key] = value
