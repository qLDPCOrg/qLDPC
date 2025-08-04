import importlib.metadata
import sys

from . import abstract, cache, codes, decoders, external, math, objects, stim

# make qldpc.circuits an alias for qldpc.stim
circuits = stim
sys.modules["qldpc.circuits"] = stim

__version__ = importlib.metadata.version("qldpc")

__all__ = [
    "__version__",
    "abstract",
    "cache",
    "circuits",
    "codes",
    "decoders",
    "external",
    "math",
    "objects",
    "stim",
]
