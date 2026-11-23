"""FORGE 3D Princeton Ocean Model (Double-Gyre) experimental package."""

from .config import ForgePOM3DConfig, POM3DExperimentConfig
from .network import ForgePOMNet

__version__ = "0.2.0"

__all__ = [
    "ForgePOM3DConfig",
    "ForgePOMNet",
    "POM3DExperimentConfig",
    "__version__",
]
