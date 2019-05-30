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
    doc_string: Optional[str] = None,
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

    if doc_string is not None:
        result.lines(['"""', doc_string, '"""'])

    result.lines(lines)

    return result


def join_class_path(path: List[str], name: str) -> str:
    return "\"{}\"".format(".".join(path + [name]))


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
    auto_create: bool,
) -> None:
    full_instance_class_name = join_class_path(class_path, instance_class_name)

    mapping_type = "DirectDictAccessor" if reader else "DirectMutableDictAccessor"

    class_builder = def_block(
        builder,
        1 if class_path else 2,
        f"class {bag_class_name}",
        [f"_containers.{mapping_type}[{key_type}, {full_instance_class_name}]"]
    )

    if data_path:
        class_builder.line("_path = ({})".format("".join(f"\"{i}\", " for i in data_path)))

    if auto_create:
        class_builder.line("_auto_create = True")

    for prop_name, prop_key in extra_properties.items():
        assert prop_name.isidentifier()
        def_block(
            class_builder,
            1,
            f"def {prop_name}",
            ["self"],
            f"{full_instance_class_name}",
            decorators=["property"],
            lines=[f"return self[{repr(prop_key)}]"]
        )


def make_ns_class(
    *,
    builder: Builder,
    class_name: str,
    props: Mapping[str, str],
    nested: bool = False,
    mutable: bool,
) -> None:
    class_builder = def_block(
        builder,
        1 if nested else 2,
        f"class {class_name}",
        ["_containers.MutableNamespace" if mutable else "_containers.Namespace"],
        doc_string="",
    )

    for prop_name, prop_type in props.items():
        assert prop_name.isidentifier()
        # TODO
        # prop_name_slug = "_" if keyword.iskeyword(prop_name) else ""
        class_builder.line(f"{prop_name}: \"{prop_type}\"")


def container_prefix(
    *,
    reader: bool,
    direct: bool,
) -> str:
    result = "_containers."

    if direct:
        result += "Direct"
    else:
        result += "DictItem"

    if not reader:
        result += "Mutable"

    return result


def python_type_ann(
    tf_type: Any,
    *,
    builder: Builder,
    reader: bool,
    direct: bool,
    attr_name: str,
    class_path: List[str],
) -> str:
    cp = container_prefix(reader=reader, direct=direct)

    if isinstance(tf_type, list):
        if tf_type[0] in {"list", "set"} and len(tf_type) == 2:
            child = python_type_ann(
                tf_type[1],
                direct=False,
                attr_name=attr_name,
                class_path=class_path,
                builder=builder,
                reader=reader,
            )
            return f"{cp}ListAccessor[{child}]"
        if tf_type[0] == "map" and len(tf_type) == 2:
            child = python_type_ann(
                tf_type[1],
                direct=True,
                attr_name=attr_name,
                class_path=class_path,
                builder=builder,
                reader=reader,
            )
            return f"{cp}DictAccessor[str, {child}]"
        if tf_type[0] == "object" and len(tf_type) == 2:
            assert isinstance(tf_type[1], dict)

            instance_class_name = f"_{attr_name}_type"
            full_instance_class_name = join_class_path(class_path, instance_class_name)

            class_builder = def_block(
                builder,
                1,
                f"class {instance_class_name}",
                [f"{cp}ObjectAccessor[{full_instance_class_name}]"]
            )

            for item_name, item_tf_type in sorted(tf_type[1].items()):
                assert item_name.isidentifier()
                # TODO slug -> _renames
                slug = "_" if keyword.iskeyword(item_name) else ""
                child = python_type_ann(
                    item_tf_type,
                    direct=direct,
                    attr_name=item_name,
                    class_path=class_path + [instance_class_name],
                    builder=class_builder,
                    reader=reader,
                )
                class_builder.line(f"{item_name}{slug}: {child}")

            if not tf_type[1].items():
                class_builder.line("pass")

            return full_instance_class_name
    else:
        if tf_type == "bool":
            return "bool"
        if tf_type == "string":
            return "str"
        if tf_type == "number":
            return "int"

    assert 0, f"Unknow Terraform type {tf_type}"


def make_schema_class(
    *,
    builder: Builder,
    class_name: str,
    schemas: Iterable[Mapping[str, Any]],
    class_path: List[str],
    reader: bool,
    direct: bool = False,
) -> str:
    full_class_name = join_class_path(class_path, class_name)

    cp = container_prefix(reader=reader, direct=direct)

    class_builder = def_block(
        builder,
        1 if class_path else 2,
        f"class {class_name}",
        [f"{cp}ObjectAccessor[{full_class_name}]"]
    )

    empty = True

    for schema in schemas:
        for attr_name, attr_schema in sorted(schema.get("block", {}).get("attributes", {}).items()):
            attr_optional = attr_schema.get("optional", False)
            attr_computed = attr_schema.get("computed", False)
            if not(reader or not(attr_computed) or attr_optional):
                continue

            assert attr_name.isidentifier()
            attr_slug = "_" if keyword.iskeyword(attr_name) else ""

            python_type = python_type_ann(
                attr_schema["type"],
                builder=class_builder,
                reader=reader,
                direct=False,
                attr_name=attr_name,
                class_path=class_path + [class_name],
            )
            class_builder.line(f"{attr_name}{attr_slug}: {python_type}")
            empty = False

        for attr_name, attr_schema in sorted(schema.get("block", {}).get("block_types", {}).items()):
            assert attr_name.isidentifier()
            attr_slug = "_" if keyword.iskeyword(attr_name) else ""

            attr_class_name = make_schema_class(
                builder=class_builder,
                class_name=f"_{attr_name}_type",
                schemas=[attr_schema],
                class_path=class_path + [class_name],
                reader=reader,
            )

            attr_nesting_mode = attr_schema["nesting_mode"]
            if attr_nesting_mode == "single":
                class_builder.line(f"{attr_name}{attr_slug}: {attr_class_name}")
            elif attr_nesting_mode in {"list", "set"}:
                class_builder.line(f"{attr_name}{attr_slug}: {cp}ListAccessor[{attr_class_name}]")
            else:
                assert 0, f"Unknown Terraform nesting_mode {attr_nesting_mode}"
    if empty:
        class_builder.line("pass")

    return full_class_name


def old_make_schema_class(
    *,
    builder: Builder,
    class_name: str,
    schemas: Iterable[Mapping[str, Any]],
    class_path: List[str],
    reader: bool,

    types_set: Dict[str, int],
) -> str:
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
            "import typing as _typing",
            "from yapytf import _containers, _containers2, _genbase",
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
                auto_create=True,
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
                        "import typing as _typing",
                        "from yapytf import _containers, _containers2, _genbase",
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
                        auto_create=True,
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
                        auto_create=False,
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
                        auto_create=False,
                    )

                    make_schema_class(
                        builder=module_builder,
                        class_name="state_instance",
                        schemas=[rschema],
                        class_path=[],
                        reader=True,
                        direct=True,
                    )

                    finalize_builder(module_name, module_builder)

                make_ns_class(
                    builder=builder,
                    class_name=f"v1_model_{kind}s",
                    props=ns_props.get("model", {}),
                    mutable=True,
                )
                make_ns_class(
                    builder=builder,
                    class_name=f"v1_state_{kind}s",
                    props=ns_props.get("state", {}),
                    mutable=False,
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

    builder.lines([
        f"from . import {provider_name} as _{provider_name}"
        for provider_name in providers_paths
    ])
    builder.line("from yapytf import _containers, _containers2, _genbase")
    builder.blanks(1)

    def xns(mutable: bool, class_name: str, props: Mapping[str, str]) -> None:
        make_ns_class(
            builder=builder,
            class_name=class_name,
            props=props,
            mutable=mutable,
        )

    def ns(class_name: str, props: Mapping[str, str]) -> None:
        xns(False, class_name, props)

    def mns(class_name: str, props: Mapping[str, str]) -> None:
        xns(True, class_name, props)

    mns(
        "model",
        {"tf": "model_tf"}
    )

    mns(
        "model_tf",
        {"v1": "model_tf_v1"}
    )

    mns(
        "model_tf_v1",
        {
            "l": "_genbase.Locals",
            "d": "model_tf_v1_data_sources",
            "r": "model_tf_v1_resources",
        }
    )

    for kind in ["data_source", "resource"]:
        mns(
            f"model_tf_v1_{kind}s",
            {
                provider_name: f"_{provider_name}.v1_model_{kind}s"
                for provider_name in providers_paths
            }
        )

    mns(
        "providers_model",
        {"v1": "model_tf_v1_providers"}
    )

    mns(
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
