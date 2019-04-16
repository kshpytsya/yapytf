import os
from typing import TYPE_CHECKING

from . import _getter
from . import _goplat

if TYPE_CHECKING:
    import pathlib


def make_url(product: str, ver: str) -> str:
    baseurl = os.environ.get("HASHI_RELEASES", "https://releases.hashicorp.com")
    if "-dev" in ver:
        baseurl = os.environ.get("HASHI_DEV_RELEASES", baseurl)

    return "{baseurl}/{product}/{ver}/{product}_{ver}_{goos}_{goarch}.zip".format(
        baseurl=baseurl,
        product=product,
        ver=ver,
        goos=_goplat.goos(),
        goarch=_goplat.goarch(),
    )


def get_hashi_product(product: str, ver: str) -> "pathlib.Path":
    return _getter.get(make_url(product, ver))


def get_terraform(ver: str) -> "pathlib.Path":
    return get_hashi_product("terraform", ver)


def get_terraform_provider(name: str, ver: str) -> "pathlib.Path":
    return get_hashi_product("terraform-provider-{}".format(name), ver)
