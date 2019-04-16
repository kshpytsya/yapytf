import click
import pathlib
import subprocess
import sys
from typing import Iterable

# TODO remove once this is fixed
PLUGIN_PATH_WORKAROUND = True
TF_BIN = "terraform"


def tf_init(
    *,
    work_dir: pathlib.Path,
    terraform_path: pathlib.Path,
    providers_paths: Iterable[pathlib.Path],
) -> None:
    if PLUGIN_PATH_WORKAROUND:
        # fixed in terraform commit 30672faebea0590a3a84c34127805c915db89051
        # binaries for which are not yet published
        # Note: this fix is incompatible with Windows due the lack of usable symlinks
        work_dir.joinpath(TF_BIN).symlink_to(terraform_path.joinpath(TF_BIN))
        terraform_path = work_dir
        for provider_path in providers_paths:
            for i in provider_path.iterdir():
                work_dir.joinpath(i.name).symlink_to(i)
        providers_paths = [work_dir]

    cp = subprocess.run(
        [
            str(terraform_path.joinpath(TF_BIN)),
            "init",
        ]
        + [
            i
            for provider_path in providers_paths
            for i in ["-plugin-dir", str(provider_path)]
        ],
        cwd=work_dir,
        capture_output=True,
    )

    if cp.returncode:
        sys.stderr.write("============== terraform output ==============\n")
        sys.stderr.buffer.write(cp.stdout)
        sys.stderr.buffer.write(cp.stderr)
        sys.stderr.write("==============================================\n")
        raise click.ClickException(
            '"terraform init" failed with return code {}'.format(cp.returncode)
        )
