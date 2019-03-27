import json
import logging
import pathlib
import runpy
import shutil
import sys
import tempfile
from typing import Mapping, Optional

import click
import click_log

from . import _hashigetter
from . import _tfschema

logger = logging.getLogger(__name__)
click_log.basic_config(logger)


# Per https://stackoverflow.com/questions/17211078/how-to-temporarily-modify-sys-path-in-python
class AddPath:
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        sys.path.insert(0, self.path)

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            sys.path.remove(self.path)
        except ValueError:
            pass


class Model:
    def __init__(self, path: pathlib.Path, model_params: dict):
        with AddPath(str(path.parent)):
            logger.debug("running %s, sys.path=%s", path, sys.path)
            py = runpy.run_path(str(path))

        if "gen" in sys.modules:
            raise click.ClickException(
                '"{}" has imported "gen" module. Please wrap the import with "if typing.TYPE_CHECKING".'.format(
                    path
                )
            )

        entry = py.get("entry")
        if not callable(entry):
            raise click.ClickException('"entry" is not defined or is not a callable')

        self.model_obj = entry(model_params)
        self.versions = self.model_obj.versions()  # type: dict

        self._terraform_path = None  # type: Optional[pathlib.Path]
        self._providers_paths = None  # type: Optional[Mapping[str, pathlib.Path]]

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
    "--params",
    type=PathType(file_okay=True, exists=True),
    help='json file containing model parameters passed to "entry" function in model file.',
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

    co.model = Model(model_path, model_params=model_params)


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
def lint(ctx: click.Context, **opts):
    """
    Validate model file
    """
    co = ctx.find_object(Context)  # type: Context
    logger.debug("terraform_path: %s", co.model.terraform_path)
    logger.debug("providers_paths: %s", co.model.providers_paths)

    for provider_name, provider_version in co.model.providers_versions.items():
        schema = _tfschema.get(
            work_dir=co.work_dir,
            terraform_path=co.model.terraform_path,
            terraform_version=co.model.terraform_version,
            provider_name=provider_name,
            provider_version=provider_version,
            provider_path=co.model.providers_paths[provider_name]
        )
