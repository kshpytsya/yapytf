import json
import pathlib
import shlex
import subprocess
import sys
from typing import cast, Any, Dict, Iterable

import click

TF_BIN = "terraform"


def tf_run(
    *,
    work_dir: pathlib.Path,
    terraform_path: pathlib.Path,
    capture_output: bool,
    args: Iterable[str],
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(terraform_path.joinpath(TF_BIN))] + list(args),
        cwd=work_dir,
        capture_output=capture_output
    )


def tf_raise(
    returncode: int,
    args: Iterable[str],
) -> None:
    if returncode:
        raise click.ClickException("\"terraform{}\" failed with return code {}".format(
            "".join(f" {shlex.quote(i)}" for i in args),
            returncode
        ))


def tf_run_non_interactive(
    *,
    work_dir: pathlib.Path,
    terraform_path: pathlib.Path,
    args: Iterable[str],
) -> bytes:
    cp = tf_run(
        work_dir=work_dir,
        terraform_path=terraform_path,
        args=args,
        capture_output=True
    )

    if cp.returncode:
        WIDTH = 80
        sys.stderr.writelines([" terraform output ".center(WIDTH, "="), "\n"])
        sys.stderr.buffer.write(cp.stdout)
        sys.stderr.buffer.write(cp.stderr)
        sys.stderr.writelines(["".center(WIDTH, "="), "\n"])

        tf_raise(cp.returncode, args)

    return cast(bytes, cp.stdout)


def tf_run_interactive(
    *,
    work_dir: pathlib.Path,
    terraform_path: pathlib.Path,
    args: Iterable[str],
) -> int:
    cp = tf_run(
        work_dir=work_dir,
        terraform_path=terraform_path,
        args=args,
        capture_output=False
    )
    return cp.returncode


def tf_init(
    *,
    work_dir: pathlib.Path,
    terraform_path: pathlib.Path,
    providers_paths: Iterable[pathlib.Path],
) -> None:
    tf_run_non_interactive(
        work_dir=work_dir,
        terraform_path=terraform_path,
        args=["init"] + [i for provider_path in providers_paths for i in ["-plugin-dir", str(provider_path)]]
    )


def tf_validate(
    *,
    work_dir: pathlib.Path,
    terraform_path: pathlib.Path,
) -> None:
    tf_run_non_interactive(
        work_dir=work_dir,
        terraform_path=terraform_path,
        args=["validate"]
    )


def tf_get_state(
    *,
    work_dir: pathlib.Path,
    terraform_path: pathlib.Path,
) -> Dict[str, Any]:
    output = tf_run_non_interactive(
        work_dir=work_dir,
        terraform_path=terraform_path,
        args=["state", "pull"]
    )
    json_output = json.loads(output)
    if not isinstance(json_output, dict):
        raise RuntimeError('Output of "terraform state pull" is not a json object')

    return json_output


def tf_apply(
    *,
    work_dir: pathlib.Path,
    terraform_path: pathlib.Path,
) -> bool:
    returncode = tf_run_interactive(
        work_dir=work_dir,
        terraform_path=terraform_path,
        args=[
            "apply",
            "-input=false"
        ]
    )
    return returncode == 0


def tf_destroy(
    *,
    work_dir: pathlib.Path,
    terraform_path: pathlib.Path,
) -> None:
    args = [
        "destroy",
        "-input=false"
    ]
    returncode = tf_run_interactive(
        work_dir=work_dir,
        terraform_path=terraform_path,
        args=args
    )
    tf_raise(returncode, args)
