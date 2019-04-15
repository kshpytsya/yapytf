import inspect
import importlib
import json
import logging
import pathlib
import runpy
import shutil
import sys
import subprocess
import tempfile
from typing import Any, Callable, Dict, Mapping, Optional

import click
import click_log
import jsonschema
import yaml
from implements import implements

from . import _generator, _hashigetter, _tfrun, _tfschema, cfginterface

logger = logging.getLogger(__name__)
click_log.basic_config(logger)


# Per https://stackoverflow.com/questions/17211078/how-to-temporarily-modify-sys-path-in-python
class AddSysPath:
    def __init__(self, path: str):
        self.path = path

    def __enter__(self):
        sys.path.insert(0, self.path)

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            sys.path.remove(self.path)
        except ValueError:
            pass


class ConfigurationBase:
    def schema(self, schema: Dict[str, Any]) -> None:
        pass

    def state_backend_cfg(self, cfg: cfginterface.StateBackendConfiguration) -> None:
        pass


class Model:
    versions: Mapping
    _terraform_path: Optional[pathlib.Path]
    _providers_paths: Optional[Mapping[str, pathlib.Path]]
    _state_backend_cfg: cfginterface.StateBackendConfiguration
    _model_params: Dict[str, Any]

    def __init__(
        self,
        *,
        path: pathlib.Path,
        model_params: dict,
        model_class: str,
    ):
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
        self.model_obj.state_backend_cfg(self._state_backend_cfg)

        self._terraform_path = None
        self._providers_paths = None

    @property
    def state_backend_cfg(self) -> cfginterface.StateBackendConfiguration:
        return self._state_backend_cfg

    @property
    def terraform_version(self) -> str:
        return self.versions["terraform"]

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

    def gen_yapytfgen(
        self,
        *,
        module_dir: pathlib.Path,
        work_dir: pathlib.Path
    ) -> None:
        logger.debug("terraform_path: %s", self.terraform_path)
        logger.debug("providers_paths: %s", self.providers_paths)

        providers_py_paths: Dict[str, pathlib.Path] = {}

        for provider_name, provider_version in self.providers_versions.items():
            schema = _tfschema.get(
                work_dir=work_dir,
                terraform_path=self.terraform_path,
                terraform_version=self.terraform_version,
                provider_name=provider_name,
                provider_version=provider_version,
                provider_path=self.providers_paths[provider_name]
            )
            providers_py_paths[provider_name] = _generator.gen_provider_py(
                work_dir=work_dir,
                terraform_version=self.terraform_version,
                provider_name=provider_name,
                provider_version=provider_version,
                provider_schema=schema
            )

        _generator.gen_yapytfgen(module_dir=module_dir, providers_paths=providers_py_paths)

    def produce_tf_files(
        self,
        *,
        dest: pathlib.Path,
        build_cb: Optional[Callable[[Dict], None]] = None
    ):
        def write(path: pathlib.Path, data) -> None:
            with path.joinpath("main.tf.json").open("w") as f:
                json.dump(data, f, indent=4, sort_keys=True)

            with path.joinpath("debug.tf.yaml").open("w") as f:
                yaml.dump(data, default_flow_style=False, stream=f)

        root_data = {
            "terraform": {
                "backend": {
                    self.state_backend_cfg.name: self.state_backend_cfg.cfg_vars
                }
            },
            "module": {
                "body": {
                    "source": "./body"
                }
            }
        }

        write(dest, root_data)

        body_data: Dict[str, Any] = {}

        if build_cb:
            build_cb(body_data)

        body_path: pathlib.Path = dest.joinpath("body")
        body_path.mkdir()
        write(body_path, body_data)

    def prepare_step(
        self,
        *,
        step_dir: pathlib.Path,
        step_data: Any,
    ) -> None:
        pass

    def prepare_steps(
        self,
        *,
        work_dir: pathlib.Path
    ) -> None:
        module_dir = work_dir.joinpath("yapytfgen")
        module_dir.mkdir()

        self.gen_yapytfgen(
            module_dir=module_dir,
            work_dir=work_dir
        )

        with AddSysPath(str(work_dir)):
            yapytfgen_module = importlib.import_module("yapytfgen")

        yapytfgen_model_class = getattr(yapytfgen_module, "model")

        def build(data: Dict[Any, str]) -> None:
            yapytfgen_model = yapytfgen_model_class(data)
            self.model_obj.build(
                model=yapytfgen_model,
                data=self._model_params,
                step_data=None
            )

        self.produce_tf_files(dest=work_dir, build_cb=build)

        _tfrun.tf_init(
            work_dir=work_dir,
            terraform_path=self.terraform_path,
            providers_paths=self.providers_paths.values()
        )


class Context:
    model: Model
    work_dir: pathlib.Path


class PathType(click.Path):
    def coerce_path_result(self, rv):
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
def main(ctx: click.Context, **opts):
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
    def cleanup_tmp_dir():
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
    )


@main.command(hidden=True)
@click.argument("product")
@click.argument("ver")
@click.pass_context
def get(ctx: click.Context, **opts):
    """
    Download hashicorp product
    """
    d = _hashigetter.get_hashi_product(opts["product"], opts["ver"])
    logger.info("got: %s", d)


@main.command()
@click.pass_context
@click.argument(
    "dir",
    type=PathType(dir_okay=True, exists=True),
)
def gen(ctx: click.Context, **opts):
    """
    (Re-)generate "yapytfgen" module in specified directory.
    Warning, existing "yapytfgen" directory will be completely wiped out.
    As a safeguard, a ".yapytfgen" marker file must exist in that directory.
    """
    co: Context = ctx.find_object(Context)

    dest_dir: pathlib.Path = opts["dir"]
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

    co.model.gen_yapytfgen(module_dir=module_dir, work_dir=co.work_dir)


@main.command()
@click.pass_context
def lint(ctx: click.Context, **opts):
    """
    Validate model file
    """
    co: Context = ctx.find_object(Context)

    co.model.prepare_steps(work_dir=co.work_dir)


@main.command()
@click.pass_context
@click.argument(
    "args",
    nargs=-1
)
def rawtf(ctx: click.Context, **opts):
    """
    Execute raw terraform commands
    """
    # co: Context = ctx.find_object(Context)
