"""
Distortion Aware Gated Routing based Speech Enhancement using Supervised Mixture of Models
"""

__version__ = '1.0.0'
__author__ = 'Aditya Raikar and Sheetal Varshney'
__description__ = 'Distortion Aware Gated Routing based Speech Enhancement using Supervised Mixture of Models'


# Make key components easily accessible

from . import utils
from . import models
from . import training
from . import data

__all__ = ["utils","data","models","training"]
