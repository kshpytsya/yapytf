import errno
import contextlib
import os
import pathlib
import shutil
import urllib.parse
from typing import Callable

import appdirs

from . import APPAUTHOR, APPNAME

CACHE_DIR = pathlib.Path(appdirs.user_cache_dir(appname=APPNAME, appauthor=APPAUTHOR))


def get(key: str, produce_cb: Callable[[pathlib.Path], None]) -> pathlib.Path:
    dir_name = urllib.parse.quote(key, safe="")
    dir_path = CACHE_DIR.joinpath(dir_name)

    CACHE_DIR.mkdir(exist_ok=True, parents=True)

    if not dir_path.exists():
        tmp_dir_name = "{}.tmp{}".format(dir_name, os.getpid())
        tmp_dir_path = CACHE_DIR.joinpath(tmp_dir_name)

        with contextlib.ExitStack() as es:
            tmp_dir_path.mkdir()

            @es.callback
            def cleanup_tmp_dir():
                shutil.rmtree(tmp_dir_path)

            produce_cb(tmp_dir_path)

            # in case of race when othen process has managed to produce the same
            # item before this process, let the former win
            if not dir_path.exists():
                renamed = True

                try:
                    tmp_dir_path.rename(dir_path)
                except OSError as e:
                    if e.errno == errno.ENOTEMPTY:
                        renamed = False
                    else:
                        raise

                if renamed:
                    es.pop_all()

    return dir_path
