import logging
import os
import pathlib
import stat
import urllib.parse
import zipfile

import click
import requests
import tqdm

from . import _pcache


logger = logging.getLogger(__name__)


# Per https://stackoverflow.com/questions/42326428/zipfile-in-python-file-permission
def _extract_all_with_executable_permission(zf, target_dir):
    ZIP_UNIX_SYSTEM = 3

    for info in zf.infolist():
        extracted_path = zf.extract(info, target_dir)

        if info.create_system == ZIP_UNIX_SYSTEM and os.path.isfile(extracted_path):
            unix_attributes = info.external_attr >> 16
            if unix_attributes & stat.S_IXUSR:
                os.chmod(extracted_path, os.stat(extracted_path).st_mode | stat.S_IXUSR)


def _download(url: str, dest: pathlib.Path, *, what: str = None):
    logger.debug("downloading %s", url)
    req = requests.get(url, stream=True)
    try:
        req.raise_for_status()
    except requests.HTTPError as e:
        raise click.ClickException(str(e))

    total_size = int(req.headers.get("content-length", 0))
    block_size = 1024
    written = 0

    with dest.open("wb") as f, tqdm.tqdm(
        desc="Downloading {}".format(what or url),
        total=total_size,
        unit="B",
        unit_divisor=1024,
        unit_scale=True,
    ) as pbar:
        for data in req.iter_content(block_size):
            f.write(data)
            pbar.update(len(data))
            written = written + len(data)

    if total_size and written != total_size:
        raise RuntimeError(
            "Download size mismatch: expected {} got {} bytes".format(
                total_size, written
            )
        )


def get(url: str, *, unzip: bool = True, keep_zip: bool = False) -> pathlib.Path:
    def produce(dir_path: pathlib.Path) -> None:
        url_fname = pathlib.PurePosixPath(urllib.parse.urlparse(url).path).name
        download_path = dir_path.joinpath(url_fname)

        _download(url, download_path, what=url_fname)

        if unzip and zipfile.is_zipfile(download_path):
            with zipfile.ZipFile(download_path) as zf:
                _extract_all_with_executable_permission(zf, dir_path)

            if not keep_zip:
                download_path.unlink()

    return _pcache.get(url, produce)
