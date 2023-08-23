# Monkey patch numpy
import numpy as np

np.float = float
np.int = int

from ._musicbert import *
from .composer_classification import *
