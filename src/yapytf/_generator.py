import ast
import keyword
import pathlib
from typing import (Any, Callable, Dict, Generator, Iterable, List, Mapping,
                    Optional, Tuple, Union)

from . import _pcache

# DATA_TYPE_HINT = "_tp.Dict[str, _tp.Any]"
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
        },
    },
}

_COMMON_IMPORTS = [
    "import typing as _tp",
    "from yapytf import _containers as _c",
    "from yapytf import _genbase",
]

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


def make_ns_class(
    *,
    builder: Builder,
    class_name: str,
    props: Mapping[str, str],
    data_paths: Mapping[str, List[str]] = {},
    nested: bool = False,
) -> None:
    class_builder = def_block(
        builder,
        1 if nested else 2,
        f"class {class_name}",
        ["_c.Record"],
        doc_string="",
    )

    for prop_name, prop_type in sorted(props.items()):
        assert prop_name.isidentifier()
        prop_name_slug = "_" if keyword.iskeyword(prop_name) else ""

        extra = ""

        data_path = data_paths.get(prop_name, [])
        if data_path:
            extra += f", path={repr(data_path)}"

        class_builder.line(f"{prop_name}{prop_name_slug} = _c.ConstRecordField[{prop_type}](no_name=True{extra})")


def build_record(
    tf_type: Any,
    *,
    reader: bool,
    builder: Builder,
    class_path: List[str],
    class_name: str,
) -> None:
    class_builder = def_block(
        builder,
        1,
        f"class {class_name}",
        ["_c.Record"],
        doc_string="",
    )
    child_class_path = class_path + [class_name]

    def _one(tf_type: Any, attr_name: str, *, depth: int = 0) -> Union[str, Tuple[str, str]]:
        print("   " * depth, tf_type, attr_name, "...")
        r = _one(tf_type, attr_name, depth=depth)
        print("   " * depth, "...", r)
        return r

    def one(tf_type: Any, attr_name: str, *, depth: int = 0) -> Union[str, Tuple[str, str]]:
        if isinstance(tf_type, list):
            tf_type_kind, tf_type_inner = tf_type

            if tf_type_kind in {"list", "set", "map"}:
                child = one(tf_type_inner, attr_name, depth=depth + 1)

                if reader:
                    if tf_type_kind == "map":
                        return f"_c.ConstDict[str, {child}]"
                    else:
                        return f"_c.ConstList[{child}]"
                else:
                    ntg = f"_{attr_name}_g{depth}"
                    nts = f"_{attr_name}_s{depth}"

                    if isinstance(child, str):
                        t1 = "Plain"
                        t3 = f"{child}"
                        t4 = child
                    else:
                        t1 = ""
                        t3 = f"{child[0]}, {child[1]}"
                        t4 = child[1]

                    if tf_type_kind == "map":
                        t2 = "Dict[str, "
                        ts = f"_tp.Mapping[str, {t4}]"
                    else:
                        t2 = "List["
                        ts = f"_tp.Sequence[{t4}]"

                    class_builder.line(f"{ntg} = _c.{t1}{t2}{t3}]")
                    class_builder.line(f"{nts} = _tp.Union[{ntg}, {ts}]")

                    return ntg, nts

            if tf_type_kind == "object":
                assert isinstance(tf_type_inner, dict)

                child_class_name = f"_{attr_name}_type"

                build_record(
                    tf_type_inner,
                    reader=reader,
                    class_name=child_class_name,
                    builder=class_builder,
                    class_path=child_class_path,
                )

                return "\"{}\"".format(".".join(child_class_path + [child_class_name]))
        else:
            if tf_type == "bool":
                return "bool"
            if tf_type == "string":
                return "str"
            if tf_type == "number":
                return "int"

        assert 0, f"Unknown Terraform type {tf_type}"

    for item_name, item_tf_type in sorted(tf_type.items()):
        assert item_name.isidentifier()

        field_params: List[str] = []

        slug = "_" if keyword.iskeyword(item_name) else ""
        if slug:
            field_params.append(f"name=\"{item_name}\"")

        child = one(item_tf_type, item_name)

        if reader:
            assert isinstance(child, str)
            child2 = f"_c.ConstRecordField[{child}]"
        else:
            if isinstance(child, str):
                child2 = f"_c.PlainRecordField[{child}]"
            else:
                child2 = f"_c.RecordField[{child[0]}, {child[1]}]"

        class_builder.line(f"{item_name}{slug} = {child2}({', '.join(field_params)})")


def parse_tf_schema_block(
    schema: Mapping[str, Any],
    parsed: Dict[str, Any],
    reader: bool,
) -> None:
    block_schema = schema.get("block", {})

    for name, child_schema in block_schema.get("attributes", {}).items():
        assert name not in parsed
        attr_optional = child_schema.get("optional", False)
        attr_computed = child_schema.get("computed", False)
        if not(reader or not(attr_computed) or attr_optional):
            continue

        parsed[name] = child_schema["type"]

    for name, child_schema in block_schema.get("block_types", {}).items():
        assert name not in parsed

        attr_optional = child_schema.get("optional", False)
        attr_computed = child_schema.get("computed", False)
        if not(reader or not(attr_computed) or attr_optional):
            continue

        parsed_child: Dict[str, Any] = {}
        parse_tf_schema_block(child_schema, parsed_child, reader)
        child: Any = ["object", parsed_child]
        nesting_mode = child_schema["nesting_mode"]

        if nesting_mode == "single":
            pass
        elif nesting_mode in ("list", "set"):
            child = ["list", child]
        else:
            assert 0, f"Unknown Terraform nesting mode {nesting_mode}"

        parsed[name] = child


def make_schema_class(
    *,
    builder: Builder,
    class_name: str,
    schemas: Iterable[Mapping[str, Any]],
    reader: bool,
) -> None:
    parsed_schemas: Dict[str, Any] = {}

    for schema in schemas:
        parse_tf_schema_block(schema, parsed_schemas, reader)

    # import json
    # print(json.dumps(parsed_schemas, indent=4))
    #
    build_record(
        parsed_schemas,
        reader=reader,
        builder=builder,
        class_path=[],
        class_name=class_name,
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
        imports_block.lines(_COMMON_IMPORTS)

        def make_v1() -> None:
            make_schema_class(
                builder=builder,
                class_name="v1_model_provider",
                schemas=[pschema["provider"]],
                reader=False,
            )

            for kind in ["data_source", "resource"]:
                ns_props: Dict[str, Dict[str, str]] = {}
                ns_paths: Dict[str, Dict[str, List[str]]] = {}

                for rname, rschema in sorted(pschema.get(f"{kind}_schemas", {}).items()):
                    if rname == provider_name:
                        stripped_name = "X"
                    else:
                        assert rname.startswith(provider_name_)
                        stripped_name = rname[len(provider_name_):]

                    module_name = f"v1_{kind}_{stripped_name}"
                    imports_block.line(f"from . import {module_name}")
                    module_builder = Builder()

                    module_builder.lines(_COMMON_IMPORTS)
                    module_builder.blanks(1)

                    ns_props.setdefault("model", {})[stripped_name] = f"_c.PlainDict[str, {module_name}.model]"
                    ns_props.setdefault("state", {})[stripped_name] = \
                        f"_c.ConstDict[str, _c.ConstList[{module_name}.state]]"

                    ns_paths.setdefault("model", {})[stripped_name] = ["tf", KIND_TO_KEY[kind], rname]
                    ns_paths.setdefault("state", {})[stripped_name] = [STATE_KIND_TO_KEY[kind], rname]

                    make_schema_class(
                        builder=module_builder,
                        class_name="_model",
                        schemas=[_RES_META_ARGS_SCHEMA, rschema],
                        reader=False,
                    )

                    make_schema_class(
                        builder=module_builder,
                        class_name="_state",
                        schemas=[rschema],
                        reader=True,
                    )

                    for what in ["model", "state"]:
                        module_builder.line(f"{what} = _{what}")

                    finalize_builder(module_name, module_builder)

                make_ns_class(
                    builder=builder,
                    class_name=f"v1_model_{kind}s",
                    props=ns_props.get("model", {}),
                    data_paths=ns_paths.get("model", {}),
                )
                make_ns_class(
                    builder=builder,
                    class_name=f"v1_state_{kind}s",
                    props=ns_props.get("state", {}),
                    data_paths=ns_paths.get("state", {}),
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
    builder.lines(_COMMON_IMPORTS)
    builder.blanks(1)

    def ns(
        class_name: str,
        props: Mapping[str, str],
        data_paths: Mapping[str, List[str]] = {},
    ) -> None:
        make_ns_class(
            builder=builder,
            class_name=class_name,
            props=props,
            data_paths=data_paths,
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
        "model_tf_v1_providers",
        {
            provider_name: f"_genbase._DictWithDefault[str, _{provider_name}.v1_model_provider]"
            for provider_name in providers_paths
        },
        {
            provider_name: ["provider", provider_name]
            for provider_name in providers_paths
        },
    )

    ns(
        "providers_model",
        {"v1": "model_tf_v1_providers"},

    )

    for kind in ["data_source", "resource"]:
        ns(
            f"state_v1_{kind}s",
            {
                provider_name: f"_{provider_name}.v1_state_{kind}s"
                for provider_name in providers_paths
            }
        )

    ns(
        "state_v1",
        {
            "d": "state_v1_data_sources",
            "r": "state_v1_resources",
        }
    )

    ns(
        "state",
        {"v1": "state_v1"}
    )

    ns(
        "model_tf_v1",
        {
            "l": "_genbase.Locals",
            "d": "model_tf_v1_data_sources",
            "r": "model_tf_v1_resources",
        }
    )

    ns(
        "model_tf",
        {"v1": "model_tf_v1"}
    )

    ns(
        "model",
        {
            "tf": "model_tf",
            "x": "_genbase.Extras",
        },
        {
            "x": ["extras"],
        }
    )

    produced = builder.produce()
    module_fname.write_text(produced)
    ast.parse(produced, filename=str(module_fname))

    for provider_name, provider_path in providers_paths.items():
        module_dir.joinpath(provider_name).symlink_to(provider_path, target_is_directory=True)
