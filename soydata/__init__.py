import sys

__author__ = "lovit"
__version__ = "0.1.0"


try:
    __SOYDATA_SETUP__
except NameError:
    __SOYDATA_SETUP__ = False


if __SOYDATA_SETUP__:
    sys.stderr.write("Running in setup\n")

else:
    from . import data
    from . import visualize
