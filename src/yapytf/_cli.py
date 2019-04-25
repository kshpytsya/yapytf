import importlib
import importlib.machinery
import inspect
import json
import logging
import pathlib
import runpy
import shutil
import sys
import tempfile
from typing import Any, Callable, Dict, Generator, List, Mapping, Optional, Tuple

import click
import click_log
import jsonschema
import yaml

from implements import implements

from . import (_generator, _hashigetter, _tfrun, _tfschema, _tfstate,
               cfginterface)
from . import _genbase  # noqa: F401

logger = logging.getLogger(__name__)
click_log.basic_config(logger)


# Per https://stackoverflow.com/questions/17211078/how-to-temporarily-modify-sys-path-in-python
class AddSysPath:
    def __init__(self, path: str):
        self.path = path

    def __enter__(self) -> None:
        sys.path.insert(0, self.path)

    def __exit__(self, *args: Any) -> None:
        try:
            sys.path.remove(self.path)
        except ValueError:
            pass


class AddModuleAlias:
    def __init__(self, alias: str, target_module: str) -> None:
        class Finder(importlib.abc.MetaPathFinder):
            @classmethod
            def find_spec(cls, fullname, path=None, target=None):  # type: ignore
                if fullname == alias:
                    class Loader(importlib.abc.Loader):
                        def create_module(self, spec):  # type: ignore
                            return sys.modules[target_module]

                        def exec_module(self, module):  # type: ignore
                            pass

                    return importlib.machinery.ModuleSpec(fullname, Loader())
                else:
                    return None

        self._finder = Finder()

    def __enter__(self) -> None:
        sys.meta_path.insert(0, self._finder)

    def __exit__(self, *args: Any) -> None:
        try:
            sys.meta_path.remove(self._finder)
        except ValueError:
            pass


def wipe_dir(d: pathlib.Path) -> None:
    for i in d.iterdir():
        if i.is_dir():
            shutil.rmtree(i)
        else:
            i.unlink()


class ConfigurationBase:
    def __init__(self) -> None:
        pass

    def schema(self, schema: Dict[str, Any]) -> None:
        pass

    def state_backend_cfg(
        self,
        *,
        cfg: cfginterface.StateBackendConfiguration,
        data: cfginterface.JsonType,
    ) -> None:
        pass

    def on_success(
        self,
        *,
        state: "yapytfgen.state"  # type: ignore  # noqa
    ) -> None:
        pass

    def mementos(
        self,
        *,
        state: "yapytfgen.state",  # type: ignore  # noqa
        dest: pathlib.Path,
    ) -> None:
        pass


class Model:
    versions: Dict[str, Any]
    _yapytffile_path: pathlib.Path
    _terraform_path: Optional[pathlib.Path]
    _providers_paths: Optional[Mapping[str, pathlib.Path]]
    _providers_schemas: Optional[Mapping[str, Dict[str, Any]]]
    _state_backend_cfg: cfginterface.StateBackendConfiguration
    _model_params: Dict[str, Any]
    _work_dir: pathlib.Path
    _resource_type_to_provider: Optional[Dict[str, Dict[str, str]]]
    _resources_schemas_versions: Optional[Dict[Tuple[str, str], int]]

    def __init__(
        self,
        *,
        path: pathlib.Path,
        model_params: Dict[str, Any],
        model_class: str,
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

        class_ = py.get(model_class)
        if not inspect.isclass(class_):
            raise click.ClickException(f'"{model_class}" is not defined or is not a class')

        adapted_class = implements(cfginterface.IConfiguration)(type(model_class, (class_, ConfigurationBase), {}))
        self.model_obj: cfginterface.IConfiguration = adapted_class()

        schema: Dict[str, Any] = {
            "$schema": "http://json-schema.org/schema#",
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False
        }
        self.model_obj.schema(schema)

        try:
            jsonschema.validate(model_params, schema)
        except jsonschema.exceptions.ValidationError as e:
            raise click.ClickException(e)

        self._model_params = model_params

        self.versions = dict(providers=dict())
        self.model_obj.versions(self.versions)
        self._state_backend_cfg = cfginterface.StateBackendConfiguration()
        self.model_obj.state_backend_cfg(cfg=self._state_backend_cfg, data=model_params)

        self._terraform_path = None
        self._providers_paths = None
        self._providers_schemas = None
        self._resource_type_to_provider = None
        self._resources_schemas_versions = None

    @property
    def state_backend_cfg(self) -> cfginterface.StateBackendConfiguration:
        return self._state_backend_cfg

    @property
    def terraform_version(self) -> str:
        return self.versions["terraform"]

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

    def gen_yapytfgen(
        self,
        *,
        module_dir: pathlib.Path,
        make_genbase_link: bool,
    ) -> None:
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
            make_genbase_link=make_genbase_link
        )

    def produce_tf_files(
        self,
        *,
        dest: pathlib.Path,
        build_cb: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> None:
        def write(path: pathlib.Path, data: Dict[str, Any]) -> None:
            with path.joinpath("main.tf.json").open("w") as f:
                json.dump(data, f, indent=4, sort_keys=True)

            with path.joinpath("debug.tf.yaml").open("w") as f:
                yaml.dump(data, default_flow_style=False, stream=f)

        root_data = {
            "terraform": {
                "backend": {
                    self.state_backend_cfg.name: self.state_backend_cfg.vars
                }
            }
        }

        if build_cb:
            build_cb(root_data)

        write(dest, root_data)

    def prepare_steps(
        self,
        destroy: bool = False
    ) -> List[pathlib.Path]:
        if destroy:
            steps = [("destroy", ...)]
        else:
            steps = [("create1", None)]

        module_dir = self._work_dir.joinpath("yapytfgen")
        module_dir.mkdir()

        self.gen_yapytfgen(
            module_dir=module_dir,
            make_genbase_link=False,
        )

        with AddSysPath(str(self._work_dir)), AddModuleAlias("yapytfgen._genbase", "yapytf._genbase"):
            self.yapytfgen_module = importlib.import_module("yapytfgen")

        yapytfgen_model_class = getattr(self.yapytfgen_module, "model")

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

                yapytfgen_model = yapytfgen_model_class(d)
                self.model_obj.build(
                    model=yapytfgen_model,
                    data=self._model_params,
                    step_data=step_data
                )

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

            self.produce_tf_files(dest=step_dir, build_cb=build)

            _tfrun.tf_init(
                work_dir=step_dir,
                terraform_path=self.terraform_path,
                providers_paths=self.providers_paths.values()
            )

            _tfrun.tf_validate(
                work_dir=step_dir,
                terraform_path=self.terraform_path,
            )

        return result


class Context:
    model: Model
    work_dir: pathlib.Path


class PathType(click.Path):
    def coerce_path_result(self, rv) -> pathlib.Path:
        return pathlib.Path(super().coerce_path_result(rv))


@click.group()
@click_log.simple_verbosity_option(logger)
@click.version_option()
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
    default="Yapytffile.py",
    help="Name of a model file to use.",
    show_default=True,
)
@click.option(
    "--class",
    default="Default",
    help="Name of a model class to use.",
    show_default=True,
)
@click.option(
    "--params",
    type=PathType(file_okay=True, exists=True),
    help='json file containing model parameters passed to constructor of the model class.',
)
@click.option("--keep-work", is_flag=True, help="keep working directory")
@click.pass_context
def main(ctx: click.Context, **opts: Any) -> None:
    """
    Yet Another Python Terraform Wrapper
    """

    co = ctx.ensure_object(Context)  # type: Context

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

    co.work_dir = pathlib.Path(tempfile.mkdtemp(prefix="yapytf."))

    @ctx.call_on_close
    def cleanup_tmp_dir() -> None:
        if opts["keep_work"]:
            click.echo('Keeping working directory "{}"'.format(co.work_dir), err=True)
        else:
            shutil.rmtree(co.work_dir)

    model_path = opts["dir"].joinpath(opts["file"])
    if not model_path.exists():
        raise click.FileError(str(model_path), hint="no such file")

    co.model = Model(
        path=model_path,
        model_params=model_params,
        model_class=opts["class"],
        work_dir=co.work_dir,
    )


@main.command(hidden=True)
@click.argument("product")
@click.argument("ver")
@click.pass_context
def get(ctx: click.Context, **opts: Any) -> None:
    """
    Download hashicorp product
    """
    d = _hashigetter.get_hashi_product(opts["product"], opts["ver"])
    logger.info("got: %s", d)


@main.command()
@click.pass_context
def gen(ctx: click.Context, **opts: Any) -> None:
    """
    (Re-)generate "yapytfgen" module in directory containing Yapytffile.py.
    Warning, existing "yapytfgen" directory will be completely wiped out.
    As a safeguard, a ".yapytfgen" marker file must exist in that directory.
    """
    co: Context = ctx.find_object(Context)

    dest_dir: pathlib.Path = co.model.yapytffile_path.parent
    module_dir: pathlib.Path = dest_dir.joinpath("yapytfgen")
    marker_file: pathlib.Path = module_dir.joinpath(".yapytfgen")
    if module_dir.exists():
        if not marker_file.exists():
            raise click.ClickException(
                f"Cowardly refusing to wipe existing \"{module_dir}\", "
                + "which does not contain \".yapytfgen\" marker file"
            )

        shutil.rmtree(module_dir)

    module_dir.mkdir()
    marker_file.touch()
    module_dir.joinpath(".gitignore").write_text("*\n")

    co.model.gen_yapytfgen(module_dir=module_dir, make_genbase_link=True)


@main.command()
@click.pass_context
def lint(ctx: click.Context, **opts: Any) -> None:
    """
    Validate model file
    """
    co: Context = ctx.find_object(Context)

    co.model.prepare_steps()


@main.command()
@click.option(
    "--mementos",
    type=PathType(dir_okay=True),
    help="directory into which to store model mementos. Warning: will completely wipe the directory! "
    "Note that mementos will only be stored in case of a successful execution"
)
@click.pass_context
def apply(ctx: click.Context, **opts: Any) -> None:
    """
    Build or change infrastructure
    """
    co: Context = ctx.find_object(Context)

    steps = co.model.prepare_steps()
    if _tfrun.tf_apply(work_dir=steps[0], terraform_path=co.model.terraform_path):
        st = _tfrun.tf_get_state(work_dir=steps[0], terraform_path=co.model.terraform_path)
        st = _tfstate.get_resources_attrs(st, co.model.resources_schemas_versions)
        # print(yaml.dump(st, default_flow_style=False))

        yapytfgen_state_class = getattr(co.model.yapytfgen_module, "state")
        state = yapytfgen_state_class(st)
        co.model.model_obj.on_success(state=state)

        mementos_dir: Optional[pathlib.Path] = opts["mementos"]
        if mementos_dir is not None:
            marker_file: pathlib.Path = mementos_dir.joinpath(".yapytf_mementos")

            if mementos_dir.exists():
                if not marker_file.exists():
                    raise click.ClickException(
                        f"Cowardly refusing to wipe existing \"{mementos_dir}\", "
                        + "which does not contain \".yapytf_mementos\" marker file"
                    )
                wipe_dir(mementos_dir)
            else:
                mementos_dir.mkdir()

            marker_file.touch()

            co.model.model_obj.mementos(state=state, dest=mementos_dir)


@main.command()
@click.pass_context
def destroy(ctx: click.Context, **opts: Any) -> None:
    """
    Destroy Terraform-managed infrastructure
    """
    co: Context = ctx.find_object(Context)

    steps = co.model.prepare_steps(destroy=True)
    _tfrun.tf_destroy(work_dir=steps[0], terraform_path=co.model.terraform_path)


@main.command()
@click.pass_context
@click.argument(
    "args",
    nargs=-1
)
def rawtf(ctx: click.Context, **opts: Any) -> None:
    """
    Execute raw terraform commands
    """
    # co: Context = ctx.find_object(Context)
