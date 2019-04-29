import functools
import importlib
import inspect
import json
import logging
import pathlib
import runpy
import shutil
import sys
import tempfile
from typing import (Any, Callable, Dict, Generator, Iterable, List, Mapping,
                    Optional, Tuple, Type, cast)

import click
import click_log
import jsonschema
import yaml

from implements import implements
import toposort

from . import (Configurator, JsonType, StateBackendConfig, _generator,
               _hashigetter, _tfrun, _tfschema, _tfstate)

logger = logging.getLogger(__name__)
click_log.basic_config(logger)

VERSIONS_SCHEMA = {
    "type": "object",
    "properties":
    {
        "terraform": {"type": "string"},
        "providers":
        {
            "type": "object",
            "additionalProperties": {"type": "string"},
        },
    },
    "required": ["terraform"],
    "additionalProperties": False
}


# Per https://stackoverflow.com/questions/17211078/how-to-temporarily-modify-sys-path-in-python
class AddSysPath:
    def __init__(self, path: str):
        self.path = path

    def __enter__(self) -> None:
        sys.path.append(self.path)

    def __exit__(self, *args: Any) -> None:
        try:
            sys.path.remove(self.path)
        except ValueError:
            pass


def wipe_dir(d: pathlib.Path) -> None:
    for i in d.iterdir():
        if i.is_dir():
            shutil.rmtree(i)
        else:
            i.unlink()


def produce_tf_files(
    *,
    dest: pathlib.Path,
    build_cb: Callable[[Dict[str, Any]], None],
    state_backend_cfg: StateBackendConfig,
) -> None:
    def write(path: pathlib.Path, data: Dict[str, Any]) -> None:
        with path.joinpath("main.tf.json").open("w") as f:
            json.dump(data, f, indent=4, sort_keys=True)

        with path.joinpath("debug.tf.yaml").open("w") as f:
            yaml.dump(data, default_flow_style=False, stream=f)

    root_data = {
        "terraform": {
            "backend": {
                state_backend_cfg.name: state_backend_cfg.vars
            }
        }
    }

    build_cb(root_data)

    write(dest, root_data)


class Model:
    _versions: Dict[str, Any]
    _yapytffile_path: pathlib.Path
    _terraform_path: Optional[pathlib.Path]
    _providers_paths: Optional[Mapping[str, pathlib.Path]]
    _providers_schemas: Optional[Mapping[str, Dict[str, Any]]]
    _work_dir: pathlib.Path
    _resource_type_to_provider: Optional[Dict[str, Dict[str, str]]]
    _resources_schemas_versions: Optional[Dict[Tuple[str, str], int]]
    _configurator_classes: List[Type[Configurator]]

    def __init__(
        self,
        *,
        path: pathlib.Path,
        model_classes: Iterable[str],
        work_dir: pathlib.Path,
    ):
        self._yapytffile_path = path
        self._work_dir = work_dir

        with AddSysPath(str(path.parent)):
            logger.debug("running %s, sys.path=%s", path, sys.path)
            py = runpy.run_path(str(path))

        if "yapytfgen" in sys.modules:
            raise click.ClickException(
                f'"{path}" has imported "yapytfgen" module. Please wrap the import with "if typing.TYPE_CHECKING".'
            )

        def get_class(name: str) -> Configurator:
            class_ = py.get(name)
            if not inspect.isclass(class_):
                raise click.ClickException(f'"{name}" is not defined or is not a class')

            return cast(Configurator, implements(Configurator)(class_))

        classes = {name: get_class(name) for name in model_classes}
        requires = {name: class_.requires() for name, class_ in classes.items()}
        unsatisfied = [
            f"{name}->{j}"
            for name, i in requires.items()
            for j in i
            if j not in model_classes
        ]
        if unsatisfied:
            raise click.ClickException("Unsatisfied model class requirements: " + ", ".join(unsatisfied))

        order = toposort.toposort_flatten(requires)
        logger.debug("sorted model classes: %s", order)

        self._configurator_classes = [classes[i] for i in order]

        self._versions = dict(providers=dict())
        for class_ in self._configurator_classes:
            class_.versions(self.versions)

        try:
            jsonschema.validate(self.versions, VERSIONS_SCHEMA)
        except jsonschema.exceptions.ValidationError as e:
            raise click.ClickException(e)

        self._terraform_path = None
        self._providers_paths = None
        self._providers_schemas = None
        self._resource_type_to_provider = None
        self._resources_schemas_versions = None

    @property
    def versions(self) -> Dict[str, Any]:
        return self._versions

    @property
    def terraform_version(self) -> str:
        return cast(str, self.versions["terraform"])

    @property
    def yapytffile_path(self) -> pathlib.Path:
        return self._yapytffile_path

    @property
    def terraform_path(self) -> pathlib.Path:
        if self._terraform_path is None:
            self._terraform_paths = _hashigetter.get_terraform(self.terraform_version)

        return self._terraform_paths

    @property
    def providers_versions(self) -> Mapping[str, str]:
        return self.versions.get("providers", {})

    @property
    def providers_paths(self) -> Mapping[str, pathlib.Path]:
        if self._providers_paths is None:
            self._providers_paths = dict(
                (name, _hashigetter.get_terraform_provider(name, ver))
                for name, ver in self.providers_versions.items()
            )

        return self._providers_paths

    @property
    def providers_schemas(self) -> Mapping[str, Dict[str, Any]]:
        if self._providers_schemas is None:
            self._providers_schemas = {
                provider_name: _tfschema.get(
                    work_dir=self._work_dir,
                    terraform_path=self.terraform_path,
                    terraform_version=self.terraform_version,
                    provider_name=provider_name,
                    provider_version=provider_version,
                    provider_path=self.providers_paths[provider_name]
                )
                for provider_name, provider_version in self.providers_versions.items()
            }

        return self._providers_schemas

    @property
    def resource_type_to_provider(self) -> Dict[str, Dict[str, str]]:
        if self._resource_type_to_provider is None:
            self._resource_type_to_provider = {
                kind_key: {
                    resource_type: provider_name
                    for provider_name, provider_schema in self.providers_schemas.items()
                    for resource_type in provider_schema["provider_schemas"][provider_name].get(f"{kind}_schemas", {})
                }
                for kind, kind_key in _generator.KIND_TO_KEY.items()
            }

        return self._resource_type_to_provider

    @property
    def resources_schemas_versions(self) -> Dict[Tuple[str, str], int]:
        if self._resources_schemas_versions is None:
            self._resources_schemas_versions = {
                (kind_key, resource_type): resource_schema["version"]
                for kind, kind_key in _generator.STATE_KIND_TO_KEY.items()
                for provider_name, provider_schema in self.providers_schemas.items()
                for resource_type, resource_schema
                in provider_schema["provider_schemas"][provider_name].get(f"{kind}_schemas", {}).items()
            }

        return self._resources_schemas_versions

    def gen_yapytfgen(self, module_dir: pathlib.Path) -> None:
        logger.debug("terraform_path: %s", self.terraform_path)
        logger.debug("providers_paths: %s", self.providers_paths)

        providers_py_paths: Dict[str, pathlib.Path] = {}

        for provider_name, provider_version in self.providers_versions.items():
            providers_py_paths[provider_name] = _generator.gen_provider_py(
                work_dir=self._work_dir,
                terraform_version=self.terraform_version,
                provider_name=provider_name,
                provider_version=provider_version,
                provider_schema=self.providers_schemas[provider_name]
            )

        _generator.gen_yapytfgen(
            module_dir=module_dir,
            providers_paths=providers_py_paths,
        )

    def prepare_steps(
        self,
        *,
        model_params: Dict[str, Any],
        destroy: bool = False
    ) -> Tuple[List[Configurator], List[pathlib.Path]]:
        schema: Dict[str, Any] = {
            "$schema": "http://json-schema.org/schema#",
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False
        }

        for class_ in self._configurator_classes:
            class_.schema(schema)

        try:
            jsonschema.validate(model_params, schema)
        except jsonschema.exceptions.ValidationError as e:
            raise click.ClickException(e)

        configurators = [class_(model_params) for class_ in self._configurator_classes]

        state_backend_cfg = StateBackendConfig()

        for i in configurators:
            i.state_backend_cfg(state_backend_cfg)

        steps: List[Tuple[str, Any]]

        if destroy:
            steps = [("destroy", None)]
        else:
            steps = [("create1", None)]

        module_dir = self._work_dir.joinpath("yapytfgen")
        module_dir.mkdir()

        self.gen_yapytfgen(module_dir)

        with AddSysPath(str(self._work_dir)):
            self.yapytfgen_module = importlib.import_module("yapytfgen")

        yapytfgen_model_class = getattr(self.yapytfgen_module, "model")
        yapytfgen_providers_model_class = getattr(self.yapytfgen_module, "providers_model")

        result: List[pathlib.Path] = []
        for step_name, step_data in steps:
            def build(data: Dict[str, Any]) -> None:
                d: Dict[str, Any] = {
                    "provider": {
                        provider_name: {'': {}}
                        for provider_name in self.providers_versions
                    },
                    "tf": {},
                }

                yapytfgen_providers_model = yapytfgen_providers_model_class(d)
                for i in configurators:
                    i.populate_providers(model=yapytfgen_providers_model)

                if not destroy:
                    yapytfgen_model = yapytfgen_model_class(d)
                    for i in configurators:
                        i.populate(model=yapytfgen_model, step_data=step_data)

                assert "provider" not in d["tf"]
                data["provider"] = [
                    {
                        provider_type:
                        {
                            **provider_data,
                            **({"alias": provider_alias} if provider_alias else {})
                        }
                    }
                    for provider_type, provider_aliases in d["provider"].items()
                    for provider_alias, provider_data in provider_aliases.items()
                ]

                def drop_empty(d: Dict[Any, Any]) -> Generator[Any, None, None]:
                    return ((k, v) for k, v in d.items() if v)

                data.update(drop_empty({k: dict(drop_empty(v)) for k, v in d["tf"].items()}))

                # expand "provider" properties
                for kind in ["data", "resource"]:
                    resource_type_to_provider = self.resource_type_to_provider[kind]
                    for resource_type, resources in data.get(kind, {}).items():
                        for resource_name, resource_data in resources.items():
                            provider_alias = resource_data.get("provider")
                            if provider_alias is not None:
                                provider_type = resource_type_to_provider[resource_type]
                                assert provider_alias in d["provider"][provider_type]
                                resource_data["provider"] = f"{provider_type}.{provider_alias}"

            step_dir = self._work_dir.joinpath(f"step.{step_name}")
            step_dir.mkdir()
            result.append(step_dir)

            produce_tf_files(dest=step_dir, build_cb=build, state_backend_cfg=state_backend_cfg)

            _tfrun.tf_init(
                work_dir=step_dir,
                terraform_path=self.terraform_path,
                providers_paths=self.providers_paths.values()
            )

            _tfrun.tf_validate(
                work_dir=step_dir,
                terraform_path=self.terraform_path,
            )

        return configurators, result


class PathType(click.Path):
    def coerce_path_result(self, rv) -> pathlib.Path:  # type: ignore
        return pathlib.Path(super().coerce_path_result(rv))


def click_wrapper(wrapper: Callable[..., None], wrapped: Callable[..., None]) -> Callable[..., None]:
    wrapped_params = getattr(wrapped, "__click_params__", [])
    wrapper_params = getattr(wrapper, "__click_params__", [])
    result = functools.update_wrapper(wrapper, wrapped)
    result.__click_params__ = wrapped_params + wrapper_params  # type: ignore
    return result


def base_command(func: Callable[..., None]) -> Callable[..., None]:
    @click_log.simple_verbosity_option(logger)  # type: ignore
    def wrapper(**opts: Any) -> None:
        func(**opts)

    return click_wrapper(wrapper, func)


def model_class_command(func: Callable[..., None]) -> Callable[..., None]:
    @base_command
    @click.option(
        "--dir",
        "-C",
        type=PathType(dir_okay=True, exists=True),
        default=".",
        help="Directory containing the model file, instead of the current working directory.",
    )
    @click.option(
        "--file",
        "-f",
        metavar="NAME",
        default="Yapytffile.py",
        help="Name of a model file to use.",
        show_default=True,
    )
    @click.option(
        "--classes",
        metavar="C1[,C2...]",
        default="Default",
        help="Comma separated list of names of a model classes to use.",
        show_default=True,
    )
    @click.option("--keep-work", is_flag=True, help="Keep working directory.")
    def wrapper(**opts: Any) -> None:
        model_path = opts["dir"].joinpath(opts["file"])
        if not model_path.exists():
            raise click.FileError(str(model_path), hint="no such file")

        work_dir = pathlib.Path(tempfile.mkdtemp(prefix="yapytf."))

        try:
            model = Model(
                path=model_path,
                model_classes=opts["classes"].split(","),
                work_dir=work_dir,
            )
            func(model, **opts)
        finally:
            if opts["keep_work"]:
                click.echo('Keeping working directory "{}"'.format(work_dir), err=True)
            else:
                shutil.rmtree(work_dir)

    return click_wrapper(wrapper, func)


def model_instance_command(func: Callable[..., None]) -> Callable[..., None]:
    @model_class_command
    @click.option(
        "--params",
        type=PathType(file_okay=True, exists=True),
        help='json file containing model parameters passed to constructors of the model classes.',
    )
    def wrapper(model: Model, **opts: Any) -> None:
        if opts["params"]:  # types: pathlib.Path
            with opts["params"].open() as f:
                try:
                    model_params = json.load(f)
                except json.JSONDecodeError as e:
                    raise click.ClickException("{}: {}".format(opts["params"], e))

            if not isinstance(model_params, dict):
                raise click.ClickException(
                    "{}: expected a json object".format(opts["params"])
                )
        else:
            model_params = {}

        func(model, model_params, **opts)

    return click_wrapper(wrapper, func)


@click.group()
@click.version_option()
def main(**opts: Any) -> None:
    """
    Yet Another Python Terraform Wrapper
    """


@main.command(hidden=True)
@click.argument("product")
@click.argument("ver")
def get(**opts: Any) -> None:
    """
    Download hashicorp product.
    """
    d = _hashigetter.get_hashi_product(opts["product"], opts["ver"])
    logger.info("got: %s", d)


@main.command()
@model_class_command
def dev(model: Model, **opts: Any) -> None:
    """
    Set up development environment.

    Sets up development environment in directory containing Yapytffile.py
    by (re-)generating "yapytfgen" module and symlinking "yapytf" module.
    This should allow auto-completion (typically based on "jedi")
    and type checkers such a mypy to "just work".
    Warning, existing "yapytfgen" directory will be completely wiped out.
    As a safeguard, a ".yapytfgen" marker file must exist in that directory.
    Any existing symlink named "yapytf" will be overwritten.
    """

    dest_dir: pathlib.Path = model.yapytffile_path.parent

    # yapytf symlink

    yapytf_target = pathlib.Path(__file__).parent
    yapytf_link = dest_dir.joinpath("yapytf")

    if yapytf_link.exists():
        if not yapytf_link.is_symlink():
            raise click.ClickException(
                f"Cowardly refusing to overwrite existing \"{yapytf_link}\", "
                + "which is not a symlink"
            )

        yapytf_link.unlink()

    yapytf_link.symlink_to(yapytf_target, target_is_directory=True)

    # yapytfgen module

    yapytfgen_dir: pathlib.Path = dest_dir.joinpath("yapytfgen")
    marker_file: pathlib.Path = yapytfgen_dir.joinpath(".yapytfgen")
    if yapytfgen_dir.exists():
        if not marker_file.exists():
            raise click.ClickException(
                f"Cowardly refusing to wipe existing \"{yapytfgen_dir}\", "
                + "which does not contain \".yapytfgen\" marker file"
            )

        shutil.rmtree(yapytfgen_dir)

    yapytfgen_dir.mkdir()
    marker_file.touch()
    yapytfgen_dir.joinpath(".gitignore").write_text("*\n")

    model.gen_yapytfgen(module_dir=yapytfgen_dir)


@main.command()
@model_instance_command
def lint(model: Model, model_params: JsonType, **opts: Any) -> None:
    """
    Validate model file.
    """

    model.prepare_steps(model_params=model_params)


@main.command()
@model_instance_command
@click.option(
    "--out",
    type=PathType(dir_okay=True),
    help="Directory into which to store model outputs. Warning: will completely wipe the directory! "
    "Note that outputs will only be written in case of a successful execution."
)
def apply(model: Model, model_params: JsonType, **opts: Any) -> None:
    """
    Build or change infrastructure.
    """

    configurators, steps = model.prepare_steps(model_params=model_params)
    if _tfrun.tf_apply(work_dir=steps[0], terraform_path=model.terraform_path):
        st = _tfrun.tf_get_state(work_dir=steps[0], terraform_path=model.terraform_path)
        st = _tfstate.get_resources_attrs(st, model.resources_schemas_versions)
        # print(yaml.dump(st, default_flow_style=False))

        yapytfgen_state_class = getattr(model.yapytfgen_module, "state")
        state = yapytfgen_state_class(st)

        out_dir: Optional[pathlib.Path] = opts["out"]
        if out_dir is not None:
            marker_file: pathlib.Path = out_dir.joinpath(".yapytf_out")

            if out_dir.exists():
                if not marker_file.exists():
                    raise click.ClickException(
                        f"Cowardly refusing to wipe existing \"{out_dir}\", "
                        + "which does not contain \".yapytf_out\" marker file"
                    )
                wipe_dir(out_dir)
            else:
                out_dir.mkdir()

            marker_file.touch()

            for i in configurators:
                i.output(state=state, dest=out_dir)


@main.command()
@model_instance_command
def destroy(model: Model, model_params: JsonType, **opts: Any) -> None:
    """
    Destroy Terraform-managed infrastructure.
    """

    configurators, steps = model.prepare_steps(model_params=model_params, destroy=True)
    _tfrun.tf_destroy(work_dir=steps[0], terraform_path=model.terraform_path)


@main.command()
@model_instance_command
@click.argument(
    "args",
    nargs=-1
)
def rawtf(ctx: click.Context, **opts: Any) -> None:
    """
    Execute raw terraform commands.
    """
    # TODO
