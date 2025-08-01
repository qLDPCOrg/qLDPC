import importlib.metadata

from . import abstract, cache, codes, decoders, external, math, objects, stim

circuits = stim  # alias
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
