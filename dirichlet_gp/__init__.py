"""Dirichlet-based Gaussian-process classification for remote-sensing land-cover data."""

from .data import CLASS_CODES, CLASS_NAMES, Standardizer, load_statlog
from .gp import GPDirichletClassifier, GroupedKernel, LinearKernel, SumKernel

__all__ = [
    "CLASS_CODES",
    "CLASS_NAMES",
    "GPDirichletClassifier",
    "GroupedKernel",
    "LinearKernel",
    "Standardizer",
    "SumKernel",
    "load_statlog",
]
__version__ = "1.0.0"
