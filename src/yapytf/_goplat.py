import platform


def goos() -> str:
    try:
        return {
            "Darwin": "darwin",
            "Linux": "linux",
            "Windows": "windows",
        }[platform.system()]
    except KeyError:
        raise RuntimeError("Unsupported system \"{}\"".format(platform.system()))


def goarch() -> str:
    try:
        return {
            "AMD64": "amd64",
            "x86_64": "amd64",
        }[platform.machine()]
    except KeyError:
        raise RuntimeError("Unsupported architecture \"{}\"".format(platform.machine()))
