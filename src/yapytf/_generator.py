import ast
import keyword
import pathlib
from typing import (Any, Callable, Dict, Generator, Iterable, List, Mapping,
                    Optional, Tuple, Union)

from . import _pcache

DATA_TYPE_HINT = "_typing.Dict[str, _typing.Any]"
KIND_TO_KEY = {
    "data_source": "data",
    "resource": "resource",
}
STATE_KIND_TO_KEY = {
    "data_source": "data",
    "resource": "managed",
}

# https://www.terraform.io/docs/configuration/resources.html#meta-arguments
# TODO: this is incomplete

_RES_META_ARGS_SCHEMA = {
    "block": {
        "attributes": {
            "depends_on": {
                "type": ["set", "string"],
                "optional": True
            },
            "count": {
                "type": "number",
                "optional": True
            },
            "provider": {
                "type": "string",
                "optional": True
            },
        }
    }
}

_BuilderChunkGeneratorType = Generator[Tuple[int, str], None, None]


class Builder:
    def __init__(self) -> None:
        self.chunks: List[Callable[[str], _BuilderChunkGeneratorType]] = []
        self.indent_level = 0

    def produce(self, indent: str = "    ") -> str:
        result = []
        blanks = 0

        for chunk_producer in self.chunks:
            for n, s in chunk_producer(indent):
                blanks = max(blanks, n)

                if s:
                    for _ in range(blanks):
                        result.append("\n")

                    blanks = 0
                    result.append(s)

        return "".join(result)

    def blanks(self, n: int) -> None:
        def produce(indent: str) -> _BuilderChunkGeneratorType:
            yield n, ""

        if n:
            self.chunks.append(produce)

    def line(self, s: str) -> None:
        def produce(indent: str) -> _BuilderChunkGeneratorType:
            if s:
                tabs = 0

                while s[tabs:tabs + 1] == "\t":
                    tabs += 1

                for i in range(self.indent_level + tabs):
                    yield 0, indent

                yield 0, s[tabs:]

            yield 0, "\n"

        self.chunks.append(produce)

    def lines(self, lines: List[str]) -> None:
        for line in lines:
            self.line(line)

    def block(self, indented: bool = True) -> "Builder":
        block_builder = Builder()
        block_builder.indent_level = self.indent_level + int(indented)

        def produce(indent: str) -> _BuilderChunkGeneratorType:
            for chunk_producer in block_builder.chunks:
                yield from chunk_producer(indent)

        self.chunks.append(produce)

        return block_builder


def def_block(
    builder: Builder,
    blanks: Union[int, Tuple[int, int]],
    prefix: str,
    items: Optional[List[str]] = None,
    suffix: str = "",
    *,
    decorators: List[str] = [],
    lines: List[str] = [],
) -> Builder:
    if isinstance(blanks, int):
        blanks_before, blanks_after = blanks, blanks
    else:
        blanks_before, blanks_after = blanks

    if suffix:
        suffix = f" -> {suffix}"

    builder.blanks(blanks_before)

    for i in decorators:
        builder.line(f"@{i}")

    if items is None:
        builder.line(f"{prefix}{suffix}:")
    elif len(items) == 1:
        builder.line(f"{prefix}({items[0]}){suffix}:")
    else:
        builder.line(f"{prefix}(")
        items_block = builder.block()
        for item in items:
            items_block.line(f"{item},")
        builder.line(f"){suffix}:")

    block1 = builder.block()

    if blanks_after:
        block2 = block1.block(indented=False)
        block1.blanks(blanks_after)
        result = block2
    else:
        result = block1

    result.lines(lines)

    return result


def make_bag_of_class(
    *,
    builder: Builder,
    bag_class_name: str,
    instance_class_name: str,
    class_path: List[str],
    data_path: List[str],
    extra_properties: Mapping[str, Any] = {},
    reader: bool,
    key_type: str = "str",
) -> None:
    full_instance_class_name = ".".join(class_path + [instance_class_name])

    mapping_type = "Mapping" if reader else "MutableMapping"

    class_builder = def_block(
        builder,
        1 if class_path else 2,
        f"class {bag_class_name}",
        [f"_typing.{mapping_type}[{key_type}, \"{full_instance_class_name}\"]"]
    )
    class_builder.line("__slots__ = \"_data\"")

    def_block(
        class_builder,
        1,
        "def __init__",
        ["self", f"data: {DATA_TYPE_HINT}"],
        "None",
        lines=[
            f"self._data: {DATA_TYPE_HINT} = _genbase.dict_path_{'ro' if reader else 'rw'}(data, {data_path!r})"
        ]
    )

    if not reader:
        def_block(
            class_builder,
            1,
            "def __delitem__",
            ["self", f"key: {key_type}"],
            "None",
            lines=[
                "self._data.__delitem__(key)",
            ]
        )

    def_block(
        class_builder,
        1,
        "def __getitem__",
        ["self", f"key: {key_type}"],
        f"\"{full_instance_class_name}\"",
        lines=[
            # TODO validate key
            "return {}(self._data{})".format(
                full_instance_class_name,
                "[key]" if reader else ".setdefault(key, {})"
            ),
        ]
    )
    def_block(
        class_builder,
        1,
        "def __iter__",
        ["self"],
        f"_typing.Iterator[{key_type}]",
        lines=["return self._data.__iter__()"]
    )
    def_block(class_builder, 1, "def __len__", ["self"], "int", lines=["return len(self._data)"])

    if not reader:
        def_block(
            class_builder,
            1,
            "def __setitem__",
            ["self", f"key: {key_type}", f"value: \"{full_instance_class_name}\""],
            "None",
            lines=[
                # TODO validate key
                f"if not isinstance(value, {full_instance_class_name}):",
                f"\traise TypeError(\"expect {full_instance_class_name}, got {{type(value).__name__}}\")",
                "self._data[key] = copy.deepcopy(value._data)",
            ]
        )

    for prop_name, prop_key in extra_properties.items():
        assert prop_name.isidentifier()
        def_block(
            class_builder,
            1,
            f"def {prop_name}",
            ["self"],
            f"\"{full_instance_class_name}\"",
            decorators=["property"],
            lines=[f"return self[{repr(prop_key)}]"]
        )


def make_list_of_class(
    *,
    builder: Builder,
    list_class_name: str,
    instance_class_name: str,
    class_path: List[str],
) -> str:
    full_instance_class_name = ".".join(class_path + [instance_class_name])

    class_builder = def_block(
        builder,
        1 if class_path else 2,
        f"class {list_class_name}",
        [f"_typing.MutableSequence[\"{full_instance_class_name}\"]"]
    )
    class_builder.line("__slots__ = \"_data\"")

    def_block(
        class_builder,
        1,
        "def __init__",
        ["self", "data: _typing.List[_typing.Any]"],
        "None",
        lines=["self._data = data"]
    )
    def_block(
        class_builder,
        1,
        "def __delitem__",
        ["self", "idx: _typing.Union[int, slice]"],
        "None",
        lines=["self._data.__delitem__(idx)"]
    )
    def_block(
        class_builder,
        1,
        "def __getitem__",
        ["self", "idx: int"],
        f"\"{full_instance_class_name}\"",
        decorators=["_typing.overload"],
        lines=["..."]
    )
    def_block(
        class_builder,
        1,
        "def __getitem__",
        ["self", "idx: slice"],
        f"\"_typing.MutableSequence[{full_instance_class_name}]\"",
        decorators=["_typing.overload"],
        lines=["..."]
    )
    def_block(
        class_builder,
        1,
        "def __getitem__",
        ["self", "idx: _typing.Union[int, slice]"],
        f"_typing.Union[\"{full_instance_class_name}\", \"_typing.MutableSequence[{full_instance_class_name}]\"]",
        lines=[
            "if isinstance(idx, slice):",
            f"\treturn [{full_instance_class_name}(i) for i in self._data[idx]]",
            "else:",
            f"\treturn {full_instance_class_name}(self._data[idx])",
        ]
    )
    def_block(class_builder, 1, "def __len__", ["self"], "int", lines=["return len(self._data)"])
    def_block(
        class_builder,
        1,
        "def __setitem__",
        ["self", "idx: int", f"value: \"{full_instance_class_name}\""],
        f"None",
        decorators=["_typing.overload"],
        lines=["..."]
    )
    def_block(
        class_builder,
        1,
        "def __setitem__",
        ["self", "idx: slice", f"value: _typing.Iterable[\"{full_instance_class_name}\"]"],
        "None",
        decorators=["_typing.overload"],
        lines=["..."]
    )
    def_block(
        class_builder,
        1,
        "def __setitem__",
        [
            "self",
            "idx: _typing.Union[int, slice]",
            f"value: _typing.Union[\"{full_instance_class_name}\", _typing.Iterable[\"{full_instance_class_name}\"]]"
        ],
        "None",
        lines=[
            "if isinstance(value, _typing.Iterable):",
            f"\tfor i, j in enumerate(value):",
            f"\t\tif not isinstance(j, {full_instance_class_name}):",
            f"\t\t\traise TypeError(\"expect {full_instance_class_name}, got {{type(j).__name__}} at index {{i}}\")",
            f"\tself._data[idx] = [copy.deepcopy(i._data) for i in value]",
            "else:",
            f"\tif not isinstance(value, {full_instance_class_name}):",
            f"\t\traise TypeError(\"expect {full_instance_class_name}, got {{type(value).__name__}}\")",
            "\tself._data[idx] = copy.deepcopy(value._data)",
        ]
    )
    def_block(
        class_builder,
        1,
        "def insert",
        ["self", "index: int", f"object: \"{full_instance_class_name}\""],
        "None",
        decorators=["_typing.overload"],
        lines=["..."]
    )
    def_block(
        class_builder,
        1,
        "def insert",
        ["self", "index: int"],
        "None",
        decorators=["_typing.overload"],
        lines=["..."]
    )
    def_block(
        class_builder,
        1,
        "def insert",
        ["self", "index: int", f"object: _typing.Optional[\"{full_instance_class_name}\"] = None"],
        "None",
        lines=[
            "if object is None:",
            "\tself._data.insert(index, {})",
            "else:",
            f"\tif not isinstance(object, {full_instance_class_name}):",
            f"\t\traise TypeError(\"expect {full_instance_class_name}, got {{type(object).__name__}}\")",
            "\tself._data.insert(index, copy.deepcopy(object._data))",
        ]
    )
    def_block(
        class_builder,
        1,
        "def insert_new",
        ["self", "idx: int"],
        f"\"{full_instance_class_name}\"",
        lines=[
            f"data: {DATA_TYPE_HINT} = {{}}",
            "self._data.insert(idx, data)",
            f"return {full_instance_class_name}(data)",
        ]
    )
    def_block(
        class_builder,
        1,
        "def append_new",
        ["self"],
        f"\"{full_instance_class_name}\"",
        lines=["return self.insert_new(len(self))"]
    )

    return ".".join(class_path + [list_class_name])


def wrap_list(
    items: Iterable[str],
    sep: str = ",",
    width: int = 70,
) -> Generator[str, None, None]:
    line: List[str] = []
    line_len = 0

    sep_len = len(sep)

    for item in items:
        item_and_sep_len = len(item) + sep_len
        if line and line_len + item_and_sep_len >= width:
            yield "".join(line)
            line = []
            line_len = 0

        if line:
            line.append(" ")
            line_len += 1

        line.extend([item, sep])
        line_len += item_and_sep_len

    if line:
        yield "".join(line)


def make_ns_class(
    *,
    builder: Builder,
    class_name: str,
    data_type: str,
    props: Mapping[str, str],
    nested: bool = False,
) -> None:
    class_builder = def_block(
        builder,
        1 if nested else 2,
        f"class {class_name}",
    )

    slots = ["\"_data\""]
    slots.extend(f"\"_prop_{i}\"" for i in sorted(props))
    slots_lines = list(wrap_list(slots))
    if len(slots_lines) > 1:
        class_builder.line("__slots__ = (")
        class_builder.lines([f"\t{i}" for i in slots_lines])
        class_builder.line(")")
    else:
        class_builder.line(f"__slots__ = ({slots_lines[0]})")

    init_block = def_block(
        class_builder,
        1,
        "def __init__",
        ["self", f"data: {data_type}"],
        "None"
    )
    init_block.line("self._data = data")

    for prop_name, prop_type in props.items():
        assert prop_name.isidentifier()
        prop_name_slug = "_" if keyword.iskeyword(prop_name) else ""
        init_block.line(f"self._prop_{prop_name}: _typing.Optional[\"{prop_type}\"] = None")

        def_block(
            class_builder,
            1,
            f"def {prop_name}{prop_name_slug}",
            ["self"],
            f"\"{prop_type}\"",
            decorators=["property"],
            lines=[
                f"if self._prop_{prop_name} is None:",
                f"\tself._prop_{prop_name} = {prop_type}(self._data)",
                f"return self._prop_{prop_name}"
            ]
        )


def python_type_ann(tf_type: Union[List[str], str]) -> str:
    if isinstance(tf_type, list):
        if tf_type[0] in {"list", "set"} and len(tf_type) == 2:
            return f"_typing.List[{python_type_ann(tf_type[1])}]"
        if tf_type[0] == "map" and len(tf_type) == 2:
            return f"_typing.Dict[str, {python_type_ann(tf_type[1])}]"
    else:
        if tf_type == "bool":
            return "bool"
        if tf_type == "string":
            return "str"
        if tf_type == "number":
            return "int"

    assert 0, f"Unknow Terraform type {tf_type}"


def python_type_assert_cond(tf_type: Union[List[str], str]) -> str:
    if isinstance(tf_type, list):
        if tf_type[0] in {"list", "set"} and len(tf_type) == 2:
            return (
                f"isinstance(value, list) and "
                f"all({python_type_assert_cond(tf_type[1])} for value in value)"
            )
        if tf_type[0] == "map" and len(tf_type) == 2:
            return (
                f"isinstance(value, dict) and "
                f"all(isinstance(key, str) and {python_type_assert_cond(tf_type[1])} "
                f"for key, value in value.items())"
            )
            return f"_typing.Dict[str, {python_type_ann(tf_type[1])}]"
    else:
        if tf_type == "bool":
            return "isinstance(value, bool)"
        if tf_type == "string":
            return "isinstance(value, str)"
        if tf_type == "number":
            return "isinstance(value, int)"

    assert 0, f"Unknow Terraform type {tf_type}"


def make_schema_class(
    *,
    builder: Builder,
    class_name: str,
    schemas: Iterable[Mapping[str, Any]],
    class_path: List[str],
    reader: bool,
) -> str:
    full_class_name = ".".join(class_path + [class_name])
    class_builder = def_block(
        builder,
        1 if class_path else 2,
        f"class {class_name}"
    )
    class_builder.line("__slots__ = (\"_data\", \"_context_thing\")")

    def_block(
        class_builder,
        1,
        "def __init__",
        ["self", f"data: {DATA_TYPE_HINT}"],
        "None",
        lines=[
            "self._data = data",
            f"self._context_thing: \"_typing.Optional[{full_class_name}]\" = None",
        ]
    )

    def_block(
        class_builder,
        1,
        "def __enter__", ["self"],
        f"\"{full_class_name}\"",
        lines=[
            "assert self._context_thing is None",
            "self._context_thing = self.__class__(self._data)",
            "return self._context_thing",
        ]
    )

    def_block(
        class_builder,
        1,
        "def __exit__",
        ["self", "*args: _typing.Any"],
        "None",
        lines=[
            "assert self._context_thing is not None",
            "self._context_thing._data = None  # type: ignore",
            "self._context_thing = None",
        ]
    )

    for schema in schemas:
        for attr_name, attr_schema in schema.get("block", {}).get("attributes", {}).items():
            attr_optional = attr_schema.get("optional", False)
            attr_computed = attr_schema.get("computed", False)
            if not(reader or not(attr_computed) or attr_optional):
                continue

            assert attr_name.isidentifier()
            attr_slug = "_" if keyword.iskeyword(attr_name) else ""

            python_type = python_type_ann(attr_schema["type"])

            def_block(
                class_builder,
                1,
                f"def _validate_{attr_name}",
                ["value: _typing.Any"],
                f"{python_type}",
                decorators=["staticmethod"],
                lines=[
                    f"if not({python_type_assert_cond(attr_schema['type'])}):",
                    f"\traise TypeError(f\"expect {python_type}, got {{value!r}}\")",
                    "return value"
                ]
            )

            attr_py_optional = not(attr_computed and reader)

            if attr_py_optional:
                if_none = "return None"
            else:
                if_none = f"raise RuntimeError(\"state is missing computed \\\"{attr_name}\\\" attribute\")"

            def_block(
                class_builder,
                1,
                f"def {attr_name}{attr_slug}",
                ["self"],
                f"_typing.Optional[{python_type}]" if attr_py_optional else python_type,
                decorators=["property"],
                lines=[
                    f"result = self._data.get(\"{attr_name}\")",
                    "if result is None:",
                    f"\t{if_none}",
                    "else:",
                    f"\treturn self._validate_{attr_name}(result)",
                ]
            )

            if not reader:
                def_block(
                    class_builder,
                    1,
                    f"def {attr_name}{attr_slug}",
                    ["self", f"value: {python_type}"],
                    "None",
                    decorators=[f"{attr_name}{attr_slug}.setter"],
                    lines=[
                        f"self._validate_{attr_name}(value)",
                        f"self._data[\"{attr_name}\"] = value",
                    ]
                )

                def_block(
                    class_builder,
                    1,
                    f"def {attr_name}{attr_slug}",
                    ["self"],
                    "None",
                    decorators=[f"{attr_name}{attr_slug}.deleter"],
                    lines=[
                        f"self._data.pop(\"{attr_name}\", None)",
                    ]
                )

        for attr_name, attr_schema in schema.get("block", {}).get("block_types", {}).items():
            assert attr_name.isidentifier()
            attr_slug = "_" if keyword.iskeyword(attr_name) else ""

            attr_class_name = make_schema_class(
                builder=class_builder,
                class_name=f"_instance_{attr_name}",
                schemas=[attr_schema],
                class_path=class_path + [class_name],
                reader=reader
            )

            attr_nesting_mode = attr_schema["nesting_mode"]
            if attr_nesting_mode == "single":
                def_block(
                    class_builder,
                    1,
                    f"def {attr_name}{attr_slug}",
                    ["self"],
                    f"\"{attr_class_name}\"",
                    decorators=["property"],
                    lines=[
                        f"return {attr_class_name}(self._data.setdefault(\"{attr_name}\", {{}}))",
                    ]
                )

                if not reader:
                    def_block(
                        class_builder,
                        1,
                        f"def {attr_name}{attr_slug}",
                        ["self", f"value: \"{attr_class_name}\""],
                        "None",
                        decorators=[f"{attr_name}{attr_slug}.setter"],
                        lines=[
                            f"if not isinstance(value, {attr_class_name}):",
                            f"\traise TypeError(\"expect {attr_class_name}, got {{type(value).__name__}}\")",
                            f"self._data[\"{attr_name}\"] = copy.deepcopy(value._data)",
                        ]
                    )

                    def_block(
                        class_builder,
                        1,
                        f"def {attr_name}{attr_slug}",
                        ["self"],
                        "None",
                        decorators=[f"{attr_name}{attr_slug}.deleter"],
                        lines=[
                            f"self._data.pop(\"{attr_name}\", None)",
                        ]
                    )
            elif attr_nesting_mode in {"list", "set"}:
                attr_list_class_name = make_list_of_class(
                    builder=class_builder,
                    list_class_name=f"_list_of_{attr_name}",
                    instance_class_name=f"_instance_{attr_name}",
                    class_path=class_path + [class_name]
                )
                def_block(
                    class_builder,
                    1,
                    f"def {attr_name}{attr_slug}",
                    ["self"],
                    f"\"{attr_list_class_name}\"",
                    decorators=["property"],
                    lines=[
                        "return {}(self._data.setdefault(\"{}\", []))".format(
                            attr_list_class_name,
                            attr_name
                        )
                    ]
                )

                if not reader:
                    def_block(
                        class_builder,
                        1,
                        f"def {attr_name}{attr_slug}",
                        ["self", f"value: \"{attr_class_name}\""],
                        "None",
                        decorators=[f"{attr_name}{attr_slug}.setter"],
                        lines=[
                            f"if not isinstance(value, {attr_list_class_name}):",
                            f"\traise TypeError(\"expect {attr_list_class_name}, got {{type(value).__name__}}\")",
                            f"self._data[\"{attr_name}\"] = copy.deepcopy(value._data)",
                        ]
                    )

                    def_block(
                        class_builder,
                        1,
                        f"def {attr_name}{attr_slug}",
                        ["self"],
                        "None",
                        decorators=[f"{attr_name}{attr_slug}.deleter"],
                        lines=[
                            f"self._data.pop(\"{attr_name}\", None)",
                        ]
                    )
            else:
                assert 0, f"Unknown Terraform nesting_mode {attr_nesting_mode}"

    return full_class_name


def gen_provider_py(
    *,
    work_dir: pathlib.Path,
    terraform_version: str,
    provider_name: str,
    provider_version: str,
    provider_schema: Dict[str, Any],
) -> pathlib.Path:
    key = "provider-py-{}-{}-{}".format(
        terraform_version, provider_name, provider_version
    )

    def produce(dir_path: pathlib.Path) -> None:
        def finalize_builder(name: str, builder: Builder) -> None:
            fname = f"{name}.py"
            produced = builder.produce()
            try:
                ast.parse(produced, filename=fname)
            except SyntaxError:
                work_dir.joinpath(fname).write_text(produced)
                raise

            dir_path.joinpath(fname).write_text(produced)

        provider_name_ = f"{provider_name}_"

        pschema = provider_schema["provider_schemas"][provider_name]
        builder = Builder()
        imports_block = builder.block(indented=False)
        imports_block.lines([
            "import copy",
            "import typing as _typing",
            "from yapytf import _genbase",
        ])

        def make_v1() -> None:
            make_schema_class(
                builder=builder,
                class_name="v1_model_provider",
                schemas=[pschema["provider"]],
                class_path=[],
                reader=False,
            )
            make_bag_of_class(
                builder=builder,
                bag_class_name="v1_model_providers",
                instance_class_name="v1_model_provider",
                class_path=[],
                data_path=["provider", provider_name],
                extra_properties=dict(default=""),
                reader=False,
            )

            for kind in ["data_source", "resource"]:
                ns_props: Dict[str, Dict[str, str]] = {}

                for rname, rschema in pschema.get(f"{kind}_schemas", {}).items():
                    if rname == provider_name:
                        stripped_name = "X"
                    else:
                        assert rname.startswith(provider_name_)
                        stripped_name = rname[len(provider_name_):]

                    module_name = f"v1_{kind}_{stripped_name}"
                    imports_block.line(f"from . import {module_name}")
                    module_builder = Builder()

                    module_builder.lines([
                        "import copy",
                        "import typing as _typing",
                        "from yapytf import _genbase",
                    ])
                    module_builder.blanks(1)

                    for what in ["model", "state"]:
                        ns_props.setdefault(what, {})[stripped_name] = f"{module_name}.{what}_bag"

                    make_bag_of_class(
                        builder=module_builder,
                        bag_class_name="model_bag",
                        instance_class_name="model_instance",
                        class_path=[],
                        data_path=["tf", KIND_TO_KEY[kind], rname],
                        reader=False,
                    )

                    make_schema_class(
                        builder=module_builder,
                        class_name="model_instance",
                        schemas=[_RES_META_ARGS_SCHEMA, rschema],
                        class_path=[],
                        reader=False,
                    )

                    make_bag_of_class(
                        builder=module_builder,
                        bag_class_name="state_bag",
                        instance_class_name="state_inner_bag",
                        class_path=[],
                        data_path=[STATE_KIND_TO_KEY[kind], rname],
                        reader=True,
                    )

                    make_bag_of_class(
                        builder=module_builder,
                        bag_class_name="state_inner_bag",
                        instance_class_name="state_instance",
                        class_path=[],
                        data_path=[],
                        key_type="_typing.Any",
                        reader=True,
                        extra_properties=dict(x=None),
                    )

                    make_schema_class(
                        builder=module_builder,
                        class_name="state_instance",
                        schemas=[rschema],
                        class_path=[],
                        reader=True,
                    )

                    finalize_builder(module_name, module_builder)

                for what in ["model", "state"]:
                    make_ns_class(
                        builder=builder,
                        class_name=f"v1_{what}_{kind}s",
                        data_type=DATA_TYPE_HINT,
                        props=ns_props.get(what, {}),
                    )

        make_v1()

        imports_block.blanks(1)

        finalize_builder("__init__", builder)

    return _pcache.get(key, produce)


def gen_yapytfgen(
    *,
    module_dir: pathlib.Path,
    providers_paths: Mapping[str, pathlib.Path],
) -> None:
    module_fname = module_dir.joinpath("__init__.py")
    builder = Builder()

    builder.line("import typing as _typing")
    builder.lines([
        f"from . import {provider_name} as _{provider_name}"
        for provider_name in providers_paths
    ])
    builder.line("from yapytf import _genbase")
    builder.blanks(1)

    def ns(class_name: str, props: Mapping[str, str]) -> None:
        make_ns_class(
            builder=builder,
            class_name=class_name,
            data_type=DATA_TYPE_HINT,
            props=props,
        )

    ns(
        "model",
        {"tf": "model_tf"}
    )

    ns(
        "model_tf",
        {"v1": "model_tf_v1"}
    )

    ns(
        "model_tf_v1",
        {
            "l": "_genbase.Locals",
            "d": "model_tf_v1_data_sources",
            "r": "model_tf_v1_resources",
        }
    )

    for kind in ["data_source", "resource"]:
        ns(
            f"model_tf_v1_{kind}s",
            {
                provider_name: f"_{provider_name}.v1_model_{kind}s"
                for provider_name in providers_paths
            }
        )

    ns(
        "providers_model",
        {"v1": "model_tf_v1_providers"}
    )

    ns(
        f"model_tf_v1_providers",
        {
            provider_name: f"_{provider_name}.v1_model_providers"
            for provider_name in providers_paths
        }
    )

    ns(
        "state",
        {"v1": "state_v1"}
    )

    ns(
        "state_v1",
        {
            "d": "state_v1_data_sources",
            "r": "state_v1_resources",
        }
    )

    for kind in ["data_source", "resource"]:
        ns(
            f"state_v1_{kind}s",
            {
                provider_name: f"_{provider_name}.v1_state_{kind}s"
                for provider_name in providers_paths
            }
        )

    produced = builder.produce()
    module_fname.write_text(produced)
    ast.parse(produced, filename=str(module_fname))

    for provider_name, provider_path in providers_paths.items():
        module_dir.joinpath(provider_name).symlink_to(provider_path, target_is_directory=True)
