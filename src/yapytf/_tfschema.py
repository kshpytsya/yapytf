import json
import pathlib
import shutil
import subprocess
import sys

import click

from . import _pcache, _tfrun


def get(
    *,
    work_dir: pathlib.Path,
    terraform_path: pathlib.Path,
    terraform_version: str,
    provider_name: str,
    provider_path: pathlib.Path,
    provider_version: str,
) -> dict:
    key = "tfschema-{}-{}-{}".format(terraform_version, provider_name, provider_version)

    FNAME = "schema.json"

    def produce(dir_path: pathlib.Path) -> None:
        this_work_dir = work_dir.joinpath("tfschema-{}".format(provider_name))
        this_work_dir.mkdir()

        with this_work_dir.joinpath("main.tf.json").open("w") as f:
            json.dump({"provider": [{provider_name: {}}]}, f)

        _tfrun.tf_init(
            work_dir=this_work_dir,
            terraform_path=terraform_path,
            providers_paths=[provider_path]
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
