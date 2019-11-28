from . import unsupervised

from ._data_generator import make_moons
from ._data_generator import make_spiral
from ._data_generator import make_swiss_roll
from ._data_generator import make_radial
from ._data_generator import make_two_layer_radial
from ._data_generator import make_rectangular
from ._data_generator import make_triangular

from ._generated_data import get_decision_tree_data_1
from ._generated_data import get_decision_tree_data_2

__all__ = ['make_swiss_roll',
           'make_spiral',
           'make_moons',
           'make_radial',
           'make_two_layer_radial',
           'make_rectangular',
           'make_triangular',
           'get_decision_tree_data_1',
           'get_decision_tree_data_2'
          ]