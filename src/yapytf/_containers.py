import abc
import copy
import sys
from typing import (Any, Collection, Dict, Generic, Iterable, Iterator,
                    Mapping, MutableMapping, MutableSequence, NoReturn,
                    Optional, Sequence, Tuple, Type, TypeVar, Union, cast,
                    get_type_hints, overload, _eval_type)

import typeguard
import typing_inspect


class _IsAssignedAbc:
    @abc.abstractmethod
    def _isassigned(self) -> bool:
        raise NotImplementedError


class _Unassigned(_IsAssignedAbc):
    def _isassigned(self) -> bool:
        return False


_unassigned = _Unassigned()


def isassigned(v: Any) -> bool:
    return not(isinstance(v, _IsAssignedAbc)) or v._isassigned()


T = TypeVar("T")
Tkey = TypeVar("Tkey")

# note: "_slots" workaround is used to circumvent incompatibility of plain "__slots__"
# with multiple-inheritance


class _InjectGenericTypes:
    __slots__ = ()
    _slots: Tuple[str, ...] = ("_types_cache", "_types_origins_cache", "_types_args_cache")

    _inject_for_class: Type["_InjectGenericTypes"]
    _types_count: int
    _types_cache: Optional[Tuple[Type[Any], ...]]
    _types_origins_cache: Optional[Tuple[Type[Any], ...]]
    _types_args_cache: Optional[Tuple[Type[Any], ...]]

    def __init__(self) -> None:
        # note: due to peculiarity of typing, this cannot be initialized in ctor
        self._types_cache = None
        self._types_origins_cache = None
        self._types_args_cache = None

    @property
    def _types(self) -> Tuple[Type[Any], ...]:
        if self._types_cache is None:
            types = typing_inspect.get_args(typing_inspect.get_generic_type(self))

            if not types:
                types = typing_inspect.get_args(typing_inspect.get_generic_bases(self)[0])

            globalns = sys.modules[self.__class__.__module__].__dict__
            types = tuple(_eval_type(i, globalns, None) for i in types)

            assert len(types) == self._types_count, "generic was not properly parameterized"

            self._types_cache = types

        return self._types_cache

    @property
    def _types_origins(self) -> Tuple[Type[Any], ...]:
        if self._types_origins_cache is None:
            self._types_origins_cache = tuple(typing_inspect.get_origin(i) for i in self._types)

        return self._types_origins_cache

    @property
    def _types_args(self) -> Tuple[Type[Any], ...]:
        if self._types_args_cache is None:
            self._types_args_cache = tuple(typing_inspect.get_args(i) for i in self._types)

        return self._types_args_cache


class _Accessor:
    __slots__ = ()
    _slots: Tuple[str, ...] = ()

    @abc.abstractmethod
    def _reset_accessor(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def _get_container_for_read(self) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def _get_container_for_write(self) -> Any:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def _get_container_type() -> Type[Collection[Any]]:
        raise NotImplementedError

    def _validate_container_type(self, data: Any) -> None:
        if not isinstance(data, self._get_container_type()):
            raise TypeError(f"expected {self._get_container_type()!r}, got {data!r}")


class _ContainerAccessor(_InjectGenericTypes, _Accessor):
    __slots__ = ()
    _slots = _InjectGenericTypes._slots + _Accessor._slots + ("_types_is_accessor_cache", )

    _types_is_accessor_cache: Optional[Tuple[bool, ...]]

    def __init__(self) -> None:
        _InjectGenericTypes.__init__(self)
        _Accessor.__init__(self)
        self._types_is_accessor_cache = None

    def __repr__(self) -> str:
        return f"{typing_inspect.get_generic_type(self)}({self._get_container_for_read()!r})"

    @property
    def _types_is_accessor(self) -> Tuple[bool, ...]:
        if self._types_is_accessor_cache is None:
            self._types_is_accessor_cache = tuple(
                i is not None and issubclass(i, _ContainerAccessor)
                for i in self._types_origins
            )

        return self._types_is_accessor_cache

    def _validate_and_copy_value(self, i: int, v: Any) -> Any:
        is_accessor = self._types_is_accessor[i]

        def raise_bad_value_type() -> NoReturn:
            raise TypeError(f"expected {self._types[i]!r}, got {v!r}")

        if is_accessor:
            if not(
                isinstance(v, self._types_origins[i])
                and v._types == self._types_args[i]
            ):
                raise_bad_value_type()

            v = copy.deepcopy(v._get_container_for_read())
        else:
            if not isinstance(v, self._types[i]):
                raise_bad_value_type()

        return v


class _LenMixin(_ContainerAccessor):
    def __len__(self) -> int:
        return len(self._get_container_for_read())


class _DictAccessor(_LenMixin, Mapping[Tkey, T], _ContainerAccessor):
    __slots__ = ()
    _slots = _ContainerAccessor._slots + ("__orig_class__", )
    _types_count = 2

    @staticmethod
    def _get_container_type() -> Type[Collection[Any]]:
        return dict

    def _validate_key(self, k: Tkey) -> None:
        typeguard.check_type("key", k, self._types[0])

    def __getitem__(self, k: Tkey) -> T:
        self._validate_key(k)

        tp = self._types_origins[1] or self._types[1]
        if issubclass(tp, _DictItemAccessorMixin):
            return cast(T, self._types[1](self._get_container_for_read(), k))

        return cast(T, self._types[1](self._get_container_for_read()[k]))

    def __iter__(self) -> Iterator[Tkey]:
        return cast(Iterator[Tkey], self._get_container_for_read().__iter__())


class _MutableDictAccessor(_DictAccessor[Tkey, T], MutableMapping[Tkey, T]):
    __slots__ = ()
    _auto_create: bool = False

    def __getitem__(self, k: Tkey) -> T:
        self._validate_key(k)

        tp = self._types_origins[1] or self._types[1]
        if issubclass(tp, _DictItemAccessorMixin):
            return cast(T, self._types[1](self._get_container_for_write(), k))

        if self._auto_create:
            return cast(T, self._types[1](self._get_container_for_write().setdefault(k, {})))
        else:
            return cast(T, self._types[1](self._get_container_for_read()[k]))

    def __setitem__(self, k: Tkey, v: T) -> None:
        self._validate_key(k)
        self._get_container_for_write()[k] = cast(T, self._validate_and_copy_value(1, v))

    def __delitem__(self, k: Tkey) -> None:
        self._validate_key(k)
        self._get_container_for_write().__delitem__(k)


class _ListAccessor(_LenMixin, Sequence[T], _ContainerAccessor):
    __slots__ = ()
    _slots = _ContainerAccessor._slots + ("__orig_class__", )
    _types_count = 1

    @staticmethod
    def _get_container_type() -> Type[Collection[Any]]:
        return list

    @overload
    def __getitem__(self, i: int) -> T:
        ...

    @overload  # noqa: F811
    def __getitem__(self, i: slice) -> Sequence[T]:
        ...

    def __getitem__(self, i: Union[int, slice]) -> Union[T, Sequence[T]]:  # noqa: F811
        def validate(idx: int, v: Any) -> T:
            if self._types_is_accessor[0]:
                return cast(T, self._types[0](v))
            else:
                typeguard.check_type(f"[{idx}]", v, self._types[0])
                return cast(T, v)

        if isinstance(i, slice):
            return [
                validate((i.start or 0) + j, v)
                for j, v in enumerate(self._get_container_for_read()[i])
            ]
        else:
            return validate(i, self._get_container_for_read()[i])


class _MutableListAccessor(_ListAccessor[T], MutableSequence[T]):
    __slots__ = ()

    @overload
    def __setitem__(self, i: int, v: T) -> None:
        ...

    @overload  # noqa: F811
    def __setitem__(self, i: slice, v: Iterable[T]) -> None:
        ...

    def __setitem__(self, i: Union[int, slice], v: Union[T, Iterable[T]]) -> None:  # noqa: F811
        if isinstance(i, slice):
            self._get_container_for_write()[i] = (
                self._validate_and_copy_value(0, j)
                for j in cast(Iterable[T], v)
            )
        else:
            self._get_container_for_write()[i] = self._validate_and_copy_value(0, v)

    @overload
    def __delitem__(self, i: int) -> None:
        ...

    @overload  # noqa: F811
    def __delitem__(self, i: slice) -> None:
        ...

    def __delitem__(self, i: Union[int, slice]) -> None:  # noqa: F811
        self._get_container_for_write().__delitem__(i)

    def insert(self, i: int, v: T) -> None:
        self._get_container_for_write().insert(i, self._validate_and_copy_value(0, v))

    def append(self, v: T) -> None:
        # note: a separate "append" is needed as default one does "len" before
        # self._get_container_for_write() is called, causing dict item autocreation to fail

        self._get_container_for_write().append(self._validate_and_copy_value(0, v))

    def insert_new(self, i: int) -> T:
        pass
        # TODO
        # self._get_container_for_write().insert(i, self._validate_and_copy_value(0, v))


class _DirectAccessorMixin(_Accessor):
    __slots__ = ()
    _slots: Tuple[str, ...] = ("_container", )
    _container: Any
    _path: Tuple[str, ...] = ()

    def __init__(self, data: Any) -> None:
        _Accessor.__init__(self)
        self._validate_container_type(data)
        self._container = data

    def _reset_accessor(self) -> None:
        self._container = None

    def _get_container_for_read(self) -> Any:
        result = self._container

        for component in self._path:
            result = result.get(component, {})

        return result

    def _get_container_for_write(self) -> Any:
        result = self._container

        for component in self._path:
            result = result.setdefault(component, {})

        return result


class _DictItemAccessorMixin(_Accessor, _IsAssignedAbc):
    __slots__ = ()
    _slots: Tuple[str, ...] = ("_parent", "_key")
    _parent: Dict[str, Any]
    _key: str

    def __init__(self, parent: Dict[str, Any], key: str) -> None:
        _Accessor.__init__(self)

        if not isinstance(parent, dict):
            raise TypeError(f"parent: expected dict, got {parent!r}")

        if not isinstance(key, str):
            raise TypeError(f"key: expected str, got {key!r}")

        self._parent = parent
        self._key = key

    def _isassigned(self) -> bool:
        return self._key in self._parent

    def _reset_accessor(self) -> None:
        self._parent = None  # type: ignore

    def _get_container_for_read(self) -> Any:
        result = self._parent[self._key]
        self._validate_container_type(result)
        return result

    def _get_container_for_write(self) -> Any:
        if self._isassigned():
            return self._get_container_for_read()

        result = self._parent[self._key] = self._get_container_type()()
        return result


class DirectDictAccessor(_DirectAccessorMixin, _DictAccessor[Tkey,T]):
    __slots__ = _DictAccessor._slots + _DirectAccessorMixin._slots

    def __init__(self, data: Dict[Tkey, Any]) -> None:
        _DirectAccessorMixin.__init__(self, data)
        _DictAccessor.__init__(self)


class DirectMutableDictAccessor(_DirectAccessorMixin, _MutableDictAccessor[Tkey, T]):
    __slots__ = _MutableDictAccessor._slots + _DirectAccessorMixin._slots

    def __init__(self, data: Dict[Tkey, Any]) -> None:
        _DirectAccessorMixin.__init__(self, data)
        _MutableDictAccessor.__init__(self)


class DictItemDictAccessor(_DictItemAccessorMixin, _DictAccessor[Tkey, T]):
    __slots__ = _DictAccessor._slots + _DictItemAccessorMixin._slots

    def __init__(self, parent: Dict[str, Any], key: str) -> None:
        _DictItemAccessorMixin.__init__(self, parent, key)
        _DictAccessor.__init__(self)


class DictItemMutableDictAccessor(_DictItemAccessorMixin, _MutableDictAccessor[Tkey, T]):
    __slots__ = _MutableDictAccessor._slots + _DictItemAccessorMixin._slots

    def __init__(self, parent: Dict[str, Any], key: str) -> None:
        _DictItemAccessorMixin.__init__(self, parent, key)
        _MutableDictAccessor.__init__(self)


class DirectListAccessor(_DirectAccessorMixin, _ListAccessor[T]):
    __slots__ = _ListAccessor._slots + _DirectAccessorMixin._slots

    def __init__(self, data: Dict[str, Any]) -> None:
        _DirectAccessorMixin.__init__(self, data)
        _ListAccessor.__init__(self)


class DirectMutableListAccessor(_DirectAccessorMixin, _MutableListAccessor[T]):
    __slots__ = _MutableListAccessor._slots + _DirectAccessorMixin._slots

    def __init__(self, data: Dict[str, Any]) -> None:
        _DirectAccessorMixin.__init__(self, data)
        _MutableListAccessor.__init__(self)


class DictItemListAccessor(_DictItemAccessorMixin, _ListAccessor[T]):
    __slots__ = _ListAccessor._slots + _DictItemAccessorMixin._slots

    def __init__(self, parent: Dict[str, Any], key: str) -> None:
        _DictItemAccessorMixin.__init__(self, parent, key)
        _ListAccessor.__init__(self)


class DictItemMutableListAccessor(_DictItemAccessorMixin, _MutableListAccessor[T]):
    __slots__ = _MutableListAccessor._slots + _DictItemAccessorMixin._slots

    def __init__(self, parent: Dict[str, Any], key: str) -> None:
        _DictItemAccessorMixin.__init__(self, parent, key)
        _MutableListAccessor.__init__(self)


# since jedi seemingly has troubles with "f(self: T) -> T", CRTP idiom
# borrowed from C++ is used instead

class _ObjectAccessorBase(_Accessor, Generic[T]):
    __slots__ = ()
    _slots = _Accessor._slots + ("_context_copy", )

    _renames: Dict[str, str] = {}
    _context_copy: Optional[T]

    def __init__(self) -> None:
        _Accessor.__init__(self)
        self._context_copy = None

    @staticmethod
    def _get_container_type() -> Type[Collection[Any]]:
        return dict

    def _get_ann_and_key(self, name: str) -> Tuple[Any, str]:
        ann = self.__annotations__.get(name)
        if not ann:
            raise AttributeError(name)

        key = self._renames.get(name, name)

        return ann, key

    def __enter__(self) -> T:
        assert self._context_copy is None
        self._context_copy = cast(T, copy.copy(self))
        return self._context_copy

    def __exit__(
        self,
        *args: Any,
    ) -> None:
        assert self._context_copy is not None
        cast(_Accessor, self._context_copy)._reset_accessor()
        self._context_copy = None


class _ObjectAccessor(_ObjectAccessorBase[T]):
    __slots__ = ()

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            raise NotImplementedError

    def __delattr__(self, name: str) -> None:
        if name.startswith("_"):
            super().__delattr__(name)
        else:
            raise NotImplementedError

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)

        ann, key = self._get_ann_and_key(name)
        result = self._get_container_for_read().get(key, _unassigned)

        if result is not _unassigned and issubclass(ann, _DirectAccessorMixin):
            return ann(result)

        typeguard.check_type(name, result, ann)
        return result


class _MutableObjectAccessor(_ObjectAccessorBase[T]):
    __slots__ = ()

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            ann, key = self._get_ann_and_key(name)
            self._get_container_for_write()[key] = value  # TODO validate and copy

    def __delattr__(self, name: str) -> None:
        if name.startswith("_"):
            super().__delattr__(name)
        else:
            _, key = self._get_ann_and_key(name)
            self._get_container_for_write().pop(key, None)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)

        ann, key = self._get_ann_and_key(name)

        if issubclass(typing_inspect.get_origin(ann) or ann, _DictItemAccessorMixin):
            return ann(self._get_container_for_write(), key)

        result = self._get_container_for_read().get(key, _unassigned)

        if result is not _unassigned and issubclass(ann, _DirectAccessorMixin):
            return ann(result)

        typeguard.check_type(name, result, ann)
        return result


class DirectObjectAccessor(_DirectAccessorMixin, _ObjectAccessor[T]):
    __slots__ = _ObjectAccessor._slots + _DirectAccessorMixin._slots

    def __init__(self, data: Dict[str, Any]) -> None:
        _DirectAccessorMixin.__init__(self, data)
        _ObjectAccessor.__init__(self)


class DirectMutableObjectAccessor(_DirectAccessorMixin, _MutableObjectAccessor[T]):
    __slots__ = _MutableObjectAccessor._slots + _DirectAccessorMixin._slots

    def __init__(self, data: Dict[str, Any]) -> None:
        _DirectAccessorMixin.__init__(self, data)
        _MutableObjectAccessor.__init__(self)


class DictItemObjectAccessor(_DictItemAccessorMixin, _ObjectAccessor[T]):
    __slots__ = _ObjectAccessor._slots + _DictItemAccessorMixin._slots

    def __init__(self, parent: Dict[str, Any], key: str) -> None:
        _DictItemAccessorMixin.__init__(self, parent, key)
        _ObjectAccessor.__init__(self)


class DictItemMutableObjectAccessor(_DictItemAccessorMixin, _MutableObjectAccessor[T]):
    __slots__ = _MutableObjectAccessor._slots + _DictItemAccessorMixin._slots

    def __init__(self, parent: Dict[str, Any], key: str) -> None:
        _DictItemAccessorMixin.__init__(self, parent, key)
        _MutableObjectAccessor.__init__(self)


class _SlotsForAnnotationsMeta(type):
    def __new__(cls, name: str, bases: Tuple[type, ...], d: Dict[str, Any]) -> type:
        d["__slots__"] = (
            d.get("__slots__", ())
            + tuple(i for i in d.get("__annotations__", ()) if not i.startswith("_"))
        )
        return super().__new__(cls, name, bases, d)


class _NamespaceBase(metaclass=_SlotsForAnnotationsMeta):
    _paths: Dict[str, Tuple[str, ...]]

    def __init__(self, data: Any) -> None:
        paths = getattr(self, "_paths", {})
        typeguard.check_type("_paths", paths, Namespace.__annotations__["_paths"])

        for name, ann in get_type_hints(self.__class__).items():
            if name.startswith("_"):
                continue

            this_data = data

            for component in paths.get(name, ()):
                this_data = self._path_step(component, this_data)

            super().__setattr__(name, ann(this_data))

    def __setattr__(self, name: str, value: Any) -> None:
        raise NotImplementedError

    def __delattr__(self, name: str) -> None:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def _path_step(component: str, data: Any) -> Any:
        raise NotImplementedError


class Namespace(_NamespaceBase):
    @staticmethod
    def _path_step(component: str, data: Any) -> Any:
        return data.get(component, {})


class MutableNamespace(_NamespaceBase):
    @staticmethod
    def _path_step(component: str, data: Any) -> Any:
        return data.setdefault(component, {})
