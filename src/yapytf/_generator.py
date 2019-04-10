import ast
import keyword
import pathlib
from typing import List, Tuple, Union

from . import _pcache


class Builder:
    def __init__(self):
        self.chunks = []
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
        def produce(indent: str):
            yield n, ""

        if n:
            self.chunks.append(produce)

    def line(self, s: str) -> None:
        def produce(indent: str):
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

        def produce(indent: str):
            for chunk_producer in block_builder.chunks:
                yield from chunk_producer(indent)

        self.chunks.append(produce)

        return block_builder


def def_block(
    builder: Builder,
    blanks: Union[int, Tuple[int, int]],
    prefix: str,
    items: List[str] = None,
    suffix: str = "",
    *,
    decorators: List[str] = [],
    lines: List[str] = [],
):
    if isinstance(blanks, int):
        blanks_before, blanks_after = blanks, blanks
    else:
        blanks_before, blanks_after = blanks

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
):
    full_instance_class_name = ".".join(class_path + [instance_class_name])

    class_builder = def_block(
        builder,
        1 if class_path else 2,
        f"class {bag_class_name}",
        [f"MutableMapping[str, \"{full_instance_class_name}\"]"]
    )
    class_builder.line("__slots__ = \"_data\"")

    def_block(
        class_builder,
        1,
        "def __init__",
        ["self", "data: Dict[str, Any]"],
        lines=[
            "self._data = data" + "".join(f".setdefault(\"{i}\", {{}})" for i in data_path),
        ]
    )
    def_block(
        class_builder,
        1,
        "def __delitem__",
        ["self", "key: str"],
        lines=[
            "self._data.__delitem__(key)",
        ]
    )
    def_block(
        class_builder,
        1,
        "def __getitem__",
        ["self", "key: str"],
        f" -> \"{full_instance_class_name}\"",
        lines=[
            # TODO validate key
            f"return {full_instance_class_name}(self._data.setdefault(key, {{}}))",
        ]
    )
    def_block(class_builder, 1, "def __iter__", ["self"], lines=["return self._data.__iter__()"])
    def_block(class_builder, 1, "def __len__", ["self"], lines=["return len(self._data)"])
    def_block(
        class_builder,
        1,
        "def __setitem__",
        ["self", "key: str", f"value: \"{full_instance_class_name}\""],
        lines=[
            # TODO validate key
            f"assert isinstance(value, {full_instance_class_name})",
            "self._data[key] = copy.deepcopy(value._data)",
        ]
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
        [f"MutableSequence[\"{full_instance_class_name}\"]"]
    )
    class_builder.line("__slots__ = \"_data\"")

    def_block(
        class_builder,
        1,
        "def __init__",
        ["self", "data: List"],
        lines=["self._data = data"]
    )
    def_block(
        class_builder,
        1,
        "def __delitem__",
        ["self", "idx: Union[int, slice]"],
        lines=["self._data.__delitem__(idx)"]
    )
    def_block(
        class_builder,
        1,
        "def __getitem__",
        ["self", "idx: int"],
        f" -> \"{full_instance_class_name}\"",
        decorators=["overload"],
        lines=["..."]
    )
    def_block(
        class_builder,
        1,
        "def __getitem__",
        ["self", "idx: slice"],
        f" -> \"MutableSequence[{full_instance_class_name}]\"",
        decorators=["overload"],
        lines=["..."]
    )
    def_block(
        class_builder,
        1,
        "def __getitem__",
        ["self", "idx"],
        lines=[
            "if isinstance(idx, slice):",
            f"\treturn [{full_instance_class_name}(i) for i in self._data[idx]]",
            "else:",
            f"\treturn {full_instance_class_name}(self._data[idx])",
        ]
    )
    def_block(class_builder, 1, "def __len__", ["self"], lines=["return len(self._data)"])
    def_block(
        class_builder,
        1,
        "def __setitem__",
        ["self", "idx: int", f"value: \"{full_instance_class_name}\""],
        f" -> None",
        decorators=["overload"],
        lines=["..."]
    )
    def_block(
        class_builder,
        1,
        "def __setitem__",
        ["self", "idx: slice", f"value: Iterable[\"{full_instance_class_name}\"]"],
        f" -> None",
        decorators=["overload"],
        lines=["..."]
    )
    def_block(
        class_builder,
        1,
        "def __setitem__",
        ["self", "idx", "value"],
        " -> None",
        lines=[
            "if isinstance(idx, slice):",
            f"\tassert all(isinstance(i, {full_instance_class_name}) for i in value)",
            f"\tself._data[idx] = [copy.deepcopy(i._data) for i in value]",
            "else:",
            f"\tassert isinstance(value, {full_instance_class_name})",
            "\tself._data[idx] = copy.deepcopy(value._data)",
        ]
    )
    def_block(
        class_builder,
        1,
        "def insert",
        ["self", "index: int", f"object: \"{full_instance_class_name}\""],
        " -> None",
        decorators=["overload"],
        lines=["..."]
    )
    def_block(
        class_builder,
        1,
        "def insert",
        ["self", "index: int"],
        " -> None",
        decorators=["overload"],
        lines=["..."]
    )
    def_block(
        class_builder,
        1,
        "def insert",
        ["self", "index", "object=None"],
        " -> None",
        lines=[
            "if object is None:",
            "\tself._data.insert(index, {})",
            "else:",
            f"\tassert isinstance(object, {full_instance_class_name})",
            "\tself._data.insert(index, copy.deepcopy(object._data))",
        ]
    )
    def_block(
        class_builder,
        1,
        "def insert_new",
        ["self", "idx"],
        f" -> \"{full_instance_class_name}\"",
        lines=[
            "data: Dict[str, Any] = {}",
            "self._data.insert(idx, data)",
            f"return {full_instance_class_name}(data)",
        ]
    )
    def_block(
        class_builder,
        1,
        "def append_new",
        ["self"],
        f" -> \"{full_instance_class_name}\"",
        lines=["return self.insert_new(len(self))"]
    )

    return ".".join(class_path + [list_class_name])


def python_type_ann(tf_type) -> str:
    if isinstance(tf_type, list):
        if tf_type[0] in {"list", "set"} and len(tf_type) == 2:
            return f"List[{python_type_ann(tf_type[1])}]"
        if tf_type[0] == "map" and len(tf_type) == 2:
            return f"Dict[str, {python_type_ann(tf_type[1])}]"
    else:
        if tf_type == "bool":
            return "bool"
        if tf_type == "string":
            return "str"
        if tf_type == "number":
            return "int"

    assert 0, f"Unknow Terraform type {tf_type}"


def python_type_assert_cond(tf_type) -> str:
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
            return f"Dict[str, {python_type_ann(tf_type[1])}]"
    else:
        if tf_type == "bool":
            return "isinstance(value, bool)"
        if tf_type == "string":
            return "isinstance(value, str)"
        if tf_type == "number":
            return "isinstance(value, int)"

    assert 0, f"Unknow Terraform type {tf_type}"


def gen_provider_py(
    *,
    work_dir: pathlib.Path,
    terraform_version: str,
    provider_name: str,
    provider_version: str,
    provider_schema: dict,
) -> pathlib.Path:
    key = "provider-py-{}-{}-{}".format(
        terraform_version, provider_name, provider_version
    )

    def produce(dir_path: pathlib.Path) -> None:
        dir_path1 = dir_path.joinpath(provider_name)
        dir_path1.mkdir()

        def finalize_builder(name, builder):
            fname = f"{name}.py"
            produced = builder.produce()
            try:
                ast.parse(produced, filename=fname)
            except SyntaxError:
                work_dir.joinpath(fname).write_text(produced)
                raise

            dir_path1.joinpath(fname).write_text(produced)

        provider_name_ = f"{provider_name}_"

        pschema = provider_schema["provider_schemas"][provider_name]
        builder = Builder()
        imports_block = builder.block(indented=False)

        def make_v1():
            KIND_TO_KEY = {
                "data_source": "data",
                "resource": "resource",
            }

            for kind in ["data_source", "resource"]:
                class_builder = def_block(builder, 2, f"class v1_{kind}s_{provider_name}")
                slots_block = class_builder.block(indented=False)
                class_slots = ["_data"]

                init_block = def_block(class_builder, 1, "def __init__", ["self", "data"])
                init_block.line("self._data = data")

                for rname, rschema in sorted(pschema.get(f"{kind}_schemas", {}).items()):
                    if rname == provider_name:
                        stripped_name = "_"
                    else:
                        assert rname.startswith(provider_name_)
                        stripped_name = rname[len(provider_name_):]

                    module_name = f"v1_{kind}_{stripped_name}"
                    imports_block.line(f"from . import {module_name}")
                    module_builder = Builder()

                    module_builder.lines([
                        "import copy",
                        "from typing import overload, Iterable, Optional, MutableMapping, MutableSequence, List, Dict, Any, Union"
                    ])
                    module_builder.blanks(1)

                    bag_of_prop_name = f"_prop_{stripped_name}"
                    class_slots.append(bag_of_prop_name)
                    init_block.line(
                        f"self.{bag_of_prop_name}: {module_name}.Bag = {module_name}.Bag(data)"
                    )

                    def_block(
                        class_builder,
                        1,
                        f"def {stripped_name}",
                        ["self"],
                        f" -> {module_name}.Bag",
                        decorators=["property"],
                        lines=[f"return self.{bag_of_prop_name}"]
                    )

                    make_bag_of_class(
                        builder=module_builder,
                        bag_class_name="Bag",
                        instance_class_name="Instance",
                        class_path=[],
                        data_path=[KIND_TO_KEY[kind], rname],
                    )

                    def make_res_instance():
                        def make_class(builder, class_name, schema, class_path) -> str:
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
                                ["self", "data"],
                                lines=[
                                    "self._data = data",
                                    f"self._context_thing: \"Optional[{full_class_name}]\" = None",
                                ]
                            )

                            def_block(
                                class_builder,
                                1,
                                "def __enter__", ["self"],
                                f" -> \"{full_class_name}\"",
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
                                ["self", "*args"],
                                lines=[
                                    "assert self._context_thing is not None",
                                    "self._context_thing._data = None",
                                    "self._context_thing = None",
                                ]
                            )

                            for attr_name, attr_schema in schema.get("block", {}).get("attributes", {}).items():
                                assert attr_name.isidentifier()
                                attr_slug = "_" if keyword.iskeyword(attr_name) else ""

                                python_type = python_type_ann(attr_schema["type"])

                                def_block(
                                    class_builder,
                                    1,
                                    f"def _validate_{attr_name}",
                                    ["value"],
                                    f" -> None",
                                    decorators=["staticmethod"],
                                    lines=[
                                        "assert " + python_type_assert_cond(attr_schema["type"])
                                    ]
                                )
                                def_block(
                                    class_builder,
                                    1,
                                    f"def {attr_name}{attr_slug}",
                                    ["self"],
                                    f" -> Optional[{python_type}]",
                                    decorators=["property"],
                                    lines=[
                                        f"result = self._data.get(\"{attr_name}\")",
                                        "if result is not None:",
                                        f"\tself._validate_{attr_name}(result)",
                                        "return result"
                                    ]
                                )

                                def_block(
                                    class_builder,
                                    1,
                                    f"def {attr_name}{attr_slug}",
                                    ["self", f"value: {python_type}"],
                                    " -> None",
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
                                    " -> None",
                                    decorators=[f"{attr_name}{attr_slug}.deleter"],
                                    lines=[
                                        f"self._data.pop(\"{attr_name}\", None)",
                                    ]
                                )

                            for attr_name, attr_schema in schema.get("block", {}).get("block_types", {}).items():
                                assert attr_name.isidentifier()
                                attr_slug = "_" if keyword.iskeyword(attr_name) else ""

                                attr_class_name = make_class(
                                    class_builder,
                                    f"_instance_{attr_name}",
                                    attr_schema,
                                    class_path + [class_name]
                                )

                                attr_nesting_mode = attr_schema["nesting_mode"]
                                if attr_nesting_mode == "single":
                                    def_block(
                                        class_builder,
                                        1,
                                        f"def {attr_name}{attr_slug}",
                                        ["self"],
                                        f" -> \"{attr_class_name}\"",
                                        decorators=["property"],
                                        lines=[
                                            f"return {attr_class_name}(self._data.setdefault(\"{attr_name}\", {{}}))",
                                        ]
                                    )

                                    def_block(
                                        class_builder,
                                        1,
                                        f"def {attr_name}{attr_slug}",
                                        ["self", f"value: \"{attr_class_name}\""],
                                        " -> None",
                                        decorators=[f"{attr_name}{attr_slug}.setter"],
                                        lines=[
                                            f"assert isinstance(value, {attr_class_name})",
                                            f"self._data[\"{attr_name}\"] = copy.deepcopy(value._data)",
                                        ]
                                    )

                                    def_block(
                                        class_builder,
                                        1,
                                        f"def {attr_name}{attr_slug}",
                                        ["self"],
                                        " -> None",
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
                                        f" -> \"{attr_list_class_name}\"",
                                        decorators=["property"],
                                        lines=[
                                            "return {}(self._data.setdefault(\"{}\", []))".format(
                                                attr_list_class_name,
                                                attr_name
                                            )
                                        ]
                                    )

                                    def_block(
                                        class_builder,
                                        1,
                                        f"def {attr_name}{attr_slug}",
                                        ["self", f"value: \"{attr_class_name}\""],
                                        " -> None",
                                        decorators=[f"{attr_name}{attr_slug}.setter"],
                                        lines=[
                                            f"assert isinstance(value, {attr_list_class_name})",
                                            f"self._data[\"{attr_name}\"] = copy.deepcopy(value._data)",
                                        ]
                                    )

                                    def_block(
                                        class_builder,
                                        1,
                                        f"def {attr_name}{attr_slug}",
                                        ["self"],
                                        " -> None",
                                        decorators=[f"{attr_name}{attr_slug}.deleter"],
                                        lines=[
                                            f"self._data.pop(\"{attr_name}\", None)",
                                        ]
                                    )
                                else:
                                    assert 0, f"Unknown Terraform nesting_mode {attr_nesting_mode}"

                            return full_class_name

                        make_class(module_builder, "Instance", rschema, [])

                    make_res_instance()

                    finalize_builder(module_name, module_builder)

                slots_block.line("__slots__ = ({})".format("".join(f"\"{i}\", " for i in class_slots)))

        make_v1()

        imports_block.blanks(1)

        finalize_builder("__init__", builder)

    return _pcache.get(key, produce)
