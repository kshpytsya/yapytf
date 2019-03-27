import json
import shutil
import subprocess
import sys
from typing import TYPE_CHECKING

import click

from . import _pcache

if TYPE_CHECKING:
    import pathlib


def get(
    *,
    work_dir: "pathlib.Path",
    terraform_path: "pathlib.Path",
    terraform_version: str,
    provider_name: str,
    provider_path: "pathlib.Path",
    provider_version: str,
) -> dict:
    key = "tfschema-{}-{}-{}".format(terraform_version, provider_name, provider_version)

    FNAME = "schema.json"
    # TODO remove once this is fixed
    PLUGIN_PATH_WORKAROUND = True

    def produce(dir_path: "pathlib.Path") -> None:
        this_work_dir = work_dir.joinpath("tfschema-{}".format(provider_name))
        this_work_dir.mkdir()

        with this_work_dir.joinpath("main.tf.json").open("w") as f:
            json.dump({"provider": [{provider_name: {}}]}, f)

        if PLUGIN_PATH_WORKAROUND:
            # fixed in terraform commit 30672faebea0590a3a84c34127805c915db89051
            # binaries for which are not yet published
            # Note: this fix is incompatible with Windows due the lack of usable symlinks
            nonlocal terraform_path
            nonlocal provider_path
            this_work_dir.joinpath("terraform").symlink_to(terraform_path.joinpath("terraform"))
            terraform_path = this_work_dir
            for i in provider_path.iterdir():
                this_work_dir.joinpath(i.name).symlink_to(i)
            provider_path = this_work_dir

        cp = subprocess.run(
            [
                terraform_path.joinpath("terraform"),
                "init",
                "-plugin-dir",
                provider_path,
            ],
            cwd=this_work_dir,
            capture_output=True,
        )

        if cp.returncode:
            sys.stderr.write(cp.stdout)
            raise click.ClickException(
                '"terraform init" failed with return code {}'.format(cp.returncode)
            )

        cp = subprocess.run(
            [
                terraform_path.joinpath("terraform"),
                "providers",
                "schema",
                "-json"
            ],
            cwd=this_work_dir,
            capture_output=True,
        )

        if cp.returncode:
            sys.stderr.buffer.write(cp.stderr)
            raise click.ClickException(
                '"terraform providers schema -json" failed with return code {}'.format(cp.returncode)
            )

        dir_path.joinpath(FNAME).write_bytes(cp.stdout)

    cache_dir = _pcache.get(key, produce)

    try:
        with cache_dir.joinpath(FNAME).open() as f:
            schema = json.load(f)

        if not isinstance(schema, dict):
            raise RuntimeError(
                'terraform schema for "{}" provider is not a json object'.format(
                    provider_name
                )
            )
    except Exception:
        shutil.rmtree(cache_dir)
        raise

    return schema
