import abc as _abc
import copy as _copy
import dataclasses as _dataclasses
import sys as _sys
import typing as _tp
import typeguard as _typeguard
import typing_inspect as _typing_inspect


_CONCRETE_CONTAINERS: _tp.Dict[_tp.Type[_tp.Any], _abc.ABCMeta] = {
    _tp.Dict: dict,
    _tp.List: list,
}


def _get_generic_types(
    instance: _tp.Any,
    count: _tp.Union[int, _tp.Set[int]],
    *,
    hint: str = ""
) -> _tp.Tuple[_tp.Type[_tp.Any], ...]:
    types = _typing_inspect.get_args(_typing_inspect.get_generic_type(instance))

    if not types:
        types = _typing_inspect.get_args(_typing_inspect.get_generic_bases(instance)[0])

    globalns = _sys.modules[instance.__class__.__module__].__dict__

    _eval_type = _tp._eval_type  # type: ignore
    types = tuple(_eval_type(i, globalns, None) for i in types)

    if isinstance(count, int):
        count = {count}

    if count != {-1} and len(types) not in count or any(_typing_inspect.is_typevar(i) for i in types):
        raise TypeError(f"{instance.__class__.__name__} generic was not properly parameterized{hint}: {types}")

    return types


class _Never:
    pass


_Key = _tp.Optional[_tp.Union[str, int]]
_DictAny = _tp.Dict[_tp.Any, _tp.Any]
_ListAny = _tp.List[_tp.Any]


@_dataclasses.dataclass
class _PathComponent:
    name: _tp.Any
    is_attr: bool


def _traverse_dict(d: _DictAny, path: _tp.Iterable[_PathComponent]) -> _tp.Any:
    result = d

    for component in path:
        try:
            result = result[component.name]
        except KeyError as e:
            if component.is_attr:
                raise AttributeError(e)
            else:
                raise

    return result


T = _tp.TypeVar("T")


class _Getter(_abc.ABC, _tp.Generic[T]):
    @_abc.abstractmethod
    def get(self) -> T:
        raise NotImplementedError

    @_abc.abstractmethod
    def get_autocreate(self) -> T:
        raise NotImplementedError


class _DirectGetter(_Getter[T]):
    __slots__ = ("_val")

    def __init__(self, val: T) -> None:
        self._val = val

    def get(self) -> T:
        return self._val

    def get_autocreate(self) -> T:
        return self._val


class _NestedDictGetter(_Getter[T]):
    __slots__ = ("_dict", "_path", "_type", "_default")

    def __init__(
        self,
        src: _tp.Union[_DictAny, "_Getter[_tp.Any]"],
        path: _tp.Iterable[_PathComponent],
        default: _tp.Type[T]
    ) -> None:
        path = list(path)

        if isinstance(src, _NestedDictGetter):
            path = src._path + path
            src = src._dict
        elif isinstance(src, _Getter):
            src = src.get_autocreate()

        assert isinstance(src, dict)

        self._dict: _DictAny = src
        self._path: _tp.List[_PathComponent] = path
        self._type = default
        concrete = _tp.cast(_tp.Type[T], _typing_inspect.get_origin(default)) or default
        self._default = _CONCRETE_CONTAINERS.get(concrete, concrete)

    def get(self) -> T:
        result = _traverse_dict(self._dict, self._path)
        _typeguard.check_type("", result, self._type)
        return _tp.cast(T, result)

    def get_autocreate(self) -> T:
        result = self._dict

        if self._path:
            for component in self._path[:-1]:
                result = result.setdefault(component.name, {})

            result = result.setdefault(self._path[-1].name, self._default())

        _typeguard.check_type("", result, self._default)
        return _tp.cast(T, result)


Tself = _tp.TypeVar("Tself", bound="_ContextManagerMixin")


class _ContextManagerMixin:
    def __init__(self) -> None:
        self._context_copy: _tp.Optional["_ContextManagerMixin"] = None

    def _reset_getter(self) -> None:
        raise NotImplementedError

    def __enter__(self: Tself) -> Tself:
        assert self._context_copy is None
        self._context_copy = _copy.copy(self)
        return self._context_copy

    def __exit__(
        self,
        *args: _tp.Any,
    ) -> None:
        assert self._context_copy is not None
        self._context_copy._reset_getter()
        self._context_copy = None


class Record(_ContextManagerMixin):
    __slots__ = ("_getter")

    _getter: _Getter[_DictAny]

    def __init__(
        self,
        container: _tp.Union[_DictAny, _Getter[_DictAny]],
        # *,
        # path: _tp.Iterable[_tp.Any] = []
    ) -> None:
        _ContextManagerMixin.__init__(self)
        _typeguard.check_argument_types()
        path: _tp.Iterable[_tp.Any] = []

        if isinstance(container, _Getter):
            assert not path
            self._getter = container
        else:
            _typeguard.check_type("container", container, dict)
            self._getter = _NestedDictGetter(container, (_PathComponent(i, False) for i in path), _DictAny)

    def _reset_getter(self) -> None:
        self._getter = None  # type: ignore


Tsetter = _tp.TypeVar("Tsetter")
Tval = _tp.TypeVar("Tval")
Tkey = _tp.TypeVar("Tkey", bound=_Key)


def _generic_to_tuple(
    type_: _tp.Type[_tp.Any]
) -> _tp.Union[_tp.Type[_tp.Any], _tp.Tuple[_tp.Type[_tp.Any], ...]]:
    if _typing_inspect.is_generic_type(type_):
        class_ = _typing_inspect.get_origin(type_)
        types = _typing_inspect.get_args(type_)
        return (class_, *types)

    return type_


_TConstCtor = _tp.Callable[[_tp.Any], _tp.Any]


def _make_const_ctor(type_: _tp.Type[_tp.Any]) -> _TConstCtor:
    type2 = _generic_to_tuple(type_)

    if isinstance(type2, tuple):
        class_ = type2[0]
        types = type2[1:]

        if class_ is None:

            if getattr(type_, "_is_protocol", False):

                def ctor(raw: _tp.Any) -> _tp.Any:
                    _typeguard.check_type("value", raw, type_)
                    return raw

            else:

                def ctor(raw: _tp.Any) -> _tp.Any:
                    return type_(raw)

        else:

            def ctor(raw: _tp.Any) -> _tp.Any:
                return class_(raw, types)

    elif issubclass(type_, Record):

        def ctor(raw: _tp.Any) -> _tp.Any:
            return type_(raw)

    else:

        def ctor(raw: _tp.Any) -> _tp.Any:
            _typeguard.check_type("value", raw, type_)
            return raw

    return ctor


class _ConstRecordField(_tp.Generic[Tval]):
    __slots__ = ("path", "name", "val_ctor", "no_name")

    def __init__(
        self,
        *,
        path: _tp.List[str] = [],
        name: _tp.Optional[str] = None,
        no_name: bool = False,
    ) -> None:
        self.path = path
        self.name = name
        self.no_name = no_name
        self.val_ctor: _tp.Optional[_TConstCtor] = None

    def __set_name__(self, owner: _tp.Type[Record], name: str) -> None:
        if self.name is None:
            self.name = name

    def __set__(self, instance: Record, value: _Never) -> None:
        raise NotImplementedError

    def __delete__(self, instance: Record) -> None:
        raise NotImplementedError

    def __get__(self, instance: Record, owner: _tp.Type[Record]) -> Tval:
        assert self.name is not None

        container = instance._getter.get()

        for i in self.path:
            container = container.setdefault(i, {})

        if self.no_name:
            v = container
        else:
            try:
                v = container[self.name]
            except KeyError as e:
                raise AttributeError(e)

        if self.val_ctor is None:
            self.val_ctor = _make_const_ctor(_get_generic_types(self, 1, hint=f". Field {self.name}")[0])

        return _tp.cast(Tval, self.val_ctor(v))


class ConstRecordField(_ConstRecordField[T]):
    pass


class ConstDict(_tp.Mapping[Tkey, Tval], _ContextManagerMixin):
    __slots__ = ("_container", "_key_type", "_val_ctor")

    def __init__(self, container: _DictAny, types: _tp.Tuple[_tp.Type[_tp.Any], ...]) -> None:
        _ContextManagerMixin.__init__(self)
        _typeguard.check_type("container", container, dict)
        self._container = container
        self._key_type, val_type = types
        self._val_ctor = _make_const_ctor(val_type)

    def _reset_getter(self) -> None:
        self._container = None  # type: ignore

    def __len__(self) -> int:
        return len(self._container)

    def __getitem__(self, k: Tkey) -> Tval:
        _typeguard.check_type("key", k, self._key_type)
        return _tp.cast(Tval, self._val_ctor(self._container[k]))

    def __iter__(self) -> _tp.Iterator[Tkey]:
        def check_key(k: _tp.Any) -> Tkey:
            _typeguard.check_type("key", k, self._key_type)
            return _tp.cast(Tkey, k)

        return (check_key(i) for i in self._container.__iter__())


class ConstList(_tp.Sequence[T], _ContextManagerMixin):
    __slots__ = ("_container", "_type_ctor")

    def __init__(self, container: _tp.List[_tp.Any], types: _tp.Tuple[_tp.Type[_tp.Any], ...]) -> None:
        _ContextManagerMixin.__init__(self)
        self._container = container
        type_, = types
        self._type_ctor = _make_const_ctor(type_)

    def _reset_getter(self) -> None:
        self._container = None  # type: ignore

    def __len__(self) -> int:
        return len(self._container)

    @_tp.overload
    def __getitem__(self, i: int) -> T:
        ...

    @_tp.overload
    def __getitem__(self, i: slice) -> _tp.Sequence[T]:
        ...

    def __getitem__(self, i: _tp.Union[int, slice]) -> _tp.Union[T, _tp.Sequence[T]]:
        def validate(idx: int, v: _tp.Any) -> T:
            return _tp.cast(T, self._type_ctor(v))

        if isinstance(i, slice):
            start, stop, stride = i.indices(len(self._container))
            return [
                validate(start + stride * j, v)
                for j, v in enumerate(self._container[i])
            ]
        else:
            return validate(i, self._container[i])


def _make_ctor(
    type_: _tp.Type[_tp.Any],
    is_attr: bool
) -> _tp.Callable[[_Getter[_DictAny], _tp.Any], _tp.Any]:
    type2 = _generic_to_tuple(type_)

    if isinstance(type2, tuple):
        class_ = type2[0]
        types = type2[1:]

        def ctor(getter: _Getter[_DictAny], key: _tp.Any) -> _tp.Any:
            return class_(
                _NestedDictGetter(
                    getter,
                    [_PathComponent(key, is_attr)],
                    class_._container_type()
                ),
                types
            )

    elif issubclass(type_, Record):

        rec_class: _tp.Type[Record] = type_

        def ctor(getter: _Getter[_DictAny], key: _tp.Any) -> _tp.Any:
            return rec_class(
                _NestedDictGetter(
                    getter,
                    [_PathComponent(key, is_attr)],
                    _DictAny
                )
            )

    else:

        def ctor(getter: _Getter[_DictAny], key: _tp.Any) -> _tp.Any:
            result = getter.get()[key]
            _typeguard.check_type(key, result, type_)
            return result

    return ctor


def _copy_into(getter: _Getter[_DictAny], key: _tp.Any, val: _tp.Any) -> None:
    getter.get_autocreate()[key] = _copy.deepcopy(val._getter.get())


def _make_setter(
    type_: _tp.Type[_tp.Any],
    is_attr: bool
) -> _tp.Callable[[_Getter[_DictAny], _tp.Any, _tp.Any], None]:
    type2 = _generic_to_tuple(type_)

    if isinstance(type2, tuple):
        class_ = type2[0]
        types = type2[1:]

        def setter(getter: _Getter[_DictAny], key: _tp.Any, val: _tp.Any) -> None:
            if isinstance(val, class_):
                _copy_into(getter, key, val)
            else:
                obj = class_(
                    _NestedDictGetter(
                        getter,
                        [_PathComponent(key, is_attr)],
                        class_._container_type()
                    ),
                    types
                )
                obj._populate(val)

    elif issubclass(type_, Record):

        rec_class: _tp.Type[Record] = type_

        def setter(getter: _Getter[_DictAny], key: _tp.Any, val: _tp.Any) -> None:
            _typeguard.check_type("value", val, rec_class)
            _copy_into(getter, key, val)

    else:

        def setter(getter: _Getter[_DictAny], key: _tp.Any, val: _tp.Any) -> None:
            _typeguard.check_type("value", val, type_)
            getter.get_autocreate()[key] = val

    return setter


class _RecordField(_tp.Generic[Tval, Tsetter]):
    __slots__ = (
        "name",
        "val_ctor",
        "val_setter",
        "val_type",
        "val_setter_type"
    )

    def __init__(self, *, name: _tp.Optional[str] = None) -> None:
        self.name = name
        self.val_type: _tp.Optional[_tp.Type[_tp.Any]] = None

    def __set_name__(self, owner: _tp.Type[Record], name: str) -> None:
        if self.name is None:
            self.name = name

    def _lazy_init(self) -> None:
        if self.val_type is None:
            types = _get_generic_types(self, {1, 2}, hint=f". Field {self.name}")
            if len(types) == 1:
                types = types * 2
            self.val_type, self.val_setter_type = types
            self.val_ctor = _make_ctor(self.val_type, True)
            self.val_setter = _make_setter(self.val_type, True)

    def __set__(self, instance: Record, value: Tsetter) -> None:
        assert self.name is not None
        self._lazy_init()
        _typeguard.check_type("value", value, self.val_setter_type)
        self.val_setter(instance._getter, self.name, value)

    def __delete__(self, instance: Record) -> None:
        assert self.name is not None
        container = instance._getter.get()
        try:
            del container[self.name]
        except KeyError as e:
            raise AttributeError(e)

    def __get__(self, instance: Record, owner: _tp.Type[Record]) -> Tval:
        assert self.name is not None
        self._lazy_init()
        try:
            return _tp.cast(Tval, self.val_ctor(instance._getter, self.name))
        except KeyError as e:
            raise AttributeError(e)


class RecordField(_RecordField[Tval, Tsetter]):
    pass


class PlainRecordField(_RecordField[Tval, Tval]):
    pass


class _Dict(_tp.MutableMapping[Tkey, Tval], _tp.Generic[Tkey, Tval, Tsetter], _ContextManagerMixin):
    __slots__ = (
        "_getter",
        "_key_type",
        "_val_ctor",
        "_val_setter",
        "_val_setter_type"
    )

    @staticmethod
    def _container_type() -> _tp.Type[_tp.Any]:
        return dict

    def __init__(
        self,
        getter: _Getter[_DictAny],
        types: _tp.Tuple[_tp.Type[_tp.Any], ...],
    ) -> None:
        _ContextManagerMixin.__init__(self)
        self._getter = getter

        if len(types) == 2:
            types = types + (types[1],)

        self._key_type, val_type, self._val_setter_type = types
        self._val_ctor = _make_ctor(val_type, False)
        self._val_setter = _make_setter(val_type, False)

    def _reset_getter(self) -> None:
        self._getter = None  # type: ignore

    def _validate_key(self, k: Tkey) -> None:
        _typeguard.check_type("key", k, self._key_type)

    def __len__(self) -> int:
        return len(self._getter.get())

    def __getitem__(self, k: Tkey) -> Tval:
        self._validate_key(k)
        return _tp.cast(Tval, self._val_ctor(self._getter, k))

    def __iter__(self) -> _tp.Iterator[Tkey]:
        def check_key(k: _tp.Any) -> Tkey:
            self._validate_key(k)
            return _tp.cast(Tkey, k)

        return (check_key(i) for i in self._getter.get().__iter__())

    # ignore Argument 2 of "__setitem__" incompatible with supertype "MutableMapping"
    # which is due to Tval/Tsetter duality

    def __setitem__(self, k: Tkey, v: Tsetter) -> None:  # type: ignore
        self._validate_key(k)
        _typeguard.check_type("value", v, self._val_setter_type)
        self._val_setter(self._getter, k, v)

    def __delitem__(self, k: Tkey) -> None:
        self._validate_key(k)
        self._getter.get().__delitem__(k)

    def _populate(self, val: _tp.Any) -> None:
        _typeguard.check_type("value", val, dict)
        self.clear()
        for k, v in val.items():
            self[k] = v


class Dict(_Dict[Tkey, Tval, Tsetter]):
    pass


class PlainDict(_Dict[Tkey, Tval, Tval]):
    pass


class _List(_tp.MutableSequence[Tval], _tp.Generic[Tval, Tsetter], _ContextManagerMixin):
    __slots__ = (
        "_getter",
        "_val_ctor",
        "_val_setter",
        "_val_type",
        "_val_setter_type"
    )

    @staticmethod
    def _container_type() -> _tp.Type[_tp.Any]:
        return list

    def __init__(
        self,
        getter: _Getter[_ListAny],
        types: _tp.Tuple[_tp.Type[_tp.Any], ...],
    ) -> None:
        _ContextManagerMixin.__init__(self)
        self._getter = getter

        if len(types) == 1:
            types = types * 2

        self._val_type, self._val_setter_type = types

        type_ = _generic_to_tuple(self._val_type)

        def cast(val: _tp.Any) -> Tval:
            return _tp.cast(Tval, val)

        if isinstance(type_, tuple):
            class_ = type_[0]
            types = type_[1:]

            def check_type(i: int, raw: _tp.Any) -> None:
                _typeguard.check_type(f"[{i}]", raw, class_._container_type())

            if issubclass(class_, _List):

                def ctor(i: int, raw: _tp.Any) -> Tval:
                    check_type(i, raw)
                    return cast(class_(_DirectGetter(raw), types))

            else:

                def ctor(i: int, raw: _tp.Any) -> Tval:
                    check_type(i, raw)
                    return cast(class_(_NestedDictGetter(raw, [], class_._container_type()), types))

            def setter(val: _tp.Any) -> _tp.Any:
                if isinstance(val, class_):
                    return _copy.deepcopy(val._getter.get())
                else:
                    result = class_._container_type()()
                    obj = class_(_DirectGetter(result), types)
                    obj._populate(val)
                    return result

        elif issubclass(type_, Record):

            rec_class: _tp.Type[Record] = type_

            def ctor(i: int, raw: _tp.Any) -> Tval:
                _typeguard.check_type(f"[{i}]", raw, dict)
                return cast(rec_class(_NestedDictGetter(raw, [], _DictAny)))

            def setter(val: _tp.Any) -> _tp.Any:
                _typeguard.check_type("value", val, type_)
                return _copy.deepcopy(val._getter.get())

        else:

            def ctor(i: int, raw: _tp.Any) -> Tval:
                _typeguard.check_type(f"[{i}]", raw, type_)
                return cast(raw)

            def setter(val: _tp.Any) -> _tp.Any:
                _typeguard.check_type("value", val, type_)
                return _copy.deepcopy(val)

        self._val_ctor = ctor
        self._val_setter = setter

    def _reset_getter(self) -> None:
        self._getter = None  # type: ignore

    def _populate(self, val: _tp.Any) -> None:
        _typeguard.check_type("value", val, list)
        self._getter.get_autocreate().clear()
        for i in val:
            self.append(i)

    def __len__(self) -> int:
        return len(self._getter.get())

    @_tp.overload
    def __getitem__(self, i: int) -> Tval:
        ...

    @_tp.overload
    def __getitem__(self, i: slice) -> _tp.MutableSequence[Tval]:
        ...

    def __getitem__(self, i: _tp.Union[int, slice]) -> _tp.Union[Tval, _tp.MutableSequence[Tval]]:
        container = self._getter.get()

        if isinstance(i, slice):
            start, stop, stride = i.indices(len(container))
            return [
                self._val_ctor(start + stride * j, v)
                for j, v in enumerate(container[i])
            ]
        else:
            return self._val_ctor(i, container[i])

    # ignore Signature of "__setitem__" incompatible with supertype "MutableSequence"
    # which is due to Tval/Tsetter duality

    @_tp.overload  # type: ignore
    def __setitem__(self, i: int, v: Tsetter) -> None:
        ...

    @_tp.overload
    def __setitem__(self, i: slice, v: _tp.Iterable[Tsetter]) -> None:
        ...

    def __setitem__(self, i: _tp.Union[int, slice], v: _tp.Union[T, _tp.Iterable[Tsetter]]) -> None:
        container = self._getter.get()
        if isinstance(i, slice):
            container[i] = (self._val_setter(j) for j in _tp.cast(_tp.Iterable[Tsetter], v))
        else:
            container[i] = self._val_setter(v)

    @_tp.overload
    def __delitem__(self, i: int) -> None:
        ...

    @_tp.overload
    def __delitem__(self, i: slice) -> None:
        ...

    def __delitem__(self, i: _tp.Union[int, slice]) -> None:
        self._getter.get().__delitem__(i)

    # ignore Argument 2 of "insert" incompatible with supertype "MutableSequence"
    # which is due to Tval/Tsetter duality

    def insert(self, i: int, v: Tsetter) -> None:  # type: ignore
        self._getter.get().insert(i, self._val_setter(v))

    # ignore Argument 1 of "append" incompatible with supertype "MutableSequence"
    # which is due to Tval/Tsetter duality

    def append(self, v: Tsetter) -> None:  # type: ignore
        # note: a separate "append" is needed as default one does "len" before
        # self._getter.get_autocreate() is called, causing dict item autocreation to fail

        self._getter.get_autocreate().append(self._val_setter(v))

    def insert_new(self, i: int) -> T:
        pass
        # TODO
        # self._get_container_for_write().insert(i, self._validate_and_copy_value(0, v))


class List(_List[Tval, Tsetter]):
    pass


class PlainList(_List[Tval, Tval]):
    pass
