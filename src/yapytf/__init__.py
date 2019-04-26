from ._public import *  # noqa: F401, F403

try:
    __version__ = __import__('pkg_resources').get_distribution(__name__).version
except:  # noqa # pragma: no cover
    pass

APPNAME = __name__
APPAUTHOR = "Kyrylo Shpytsya"
