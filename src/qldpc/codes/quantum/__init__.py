# SPDX-License-Identifier: Apache-2.0

"""Quantum error-correcting codes: named codes, and families organized by construction.

Each module holds the codes of one construction, such as two-block codes or hypergraph products,
and named.py holds the named codes from the literature that have a fixed length and dimension.
"""

from typing import TYPE_CHECKING

from qldpc import _deprecations

from .balanced_product import BalancedProductCode
from .hypergraph_product import (
    BaconShorCode,
    CHGPCode,
    CRCode,
    DistanceBalancedCode,
    FieldOrRingArray,
    HGPCode,
    SHPCode,
    SHYPSCode,
)
from .lifted_product import LPCode, SLPCode
from .named import (
    C4Code,
    C6Code,
    FiveQubitCode,
    FiveQuditCode,
    GrossCode,
    QuantumGolayCode,
    SteaneCode,
    TetrahedralCode,
    TwoGrossCode,
)
from .pair_partition import PairPartitionCode
from .quantum_tanner import LQTCode, QTCode
from .self_orthogonal import (
    IcebergCode,
    ManyHypercubeCode,
    QuantumHammingCode,
    QuantumReedMullerCode,
)
from .surface import GeneralizedSurfaceCode, SurfaceCode, T4Code, ToricCode
from .trivial import TrivialCode
from .two_block import BBCode, GALACode, MBCode, TBCode

__all__ = [
    "BBCode",
    "BaconShorCode",
    "BalancedProductCode",
    "C4Code",
    "C6Code",
    "CHGPCode",
    "CRCode",
    "DistanceBalancedCode",
    "FieldOrRingArray",
    "FiveQubitCode",
    "FiveQuditCode",
    "GALACode",
    "GeneralizedSurfaceCode",
    "GrossCode",
    "HGPCode",
    "IcebergCode",
    "LPCode",
    "LQTCode",
    "MBCode",
    "ManyHypercubeCode",
    "PairPartitionCode",
    "QTCode",
    "QuantumGolayCode",
    "QuantumHammingCode",
    "QuantumReedMullerCode",
    "SHPCode",
    "SHYPSCode",
    "SLPCode",
    "SteaneCode",
    "SurfaceCode",
    "T4Code",
    "TBCode",
    "TetrahedralCode",
    "ToricCode",
    "TrivialCode",
    "TwoGrossCode",
]

if not TYPE_CHECKING:  # DEPRECATED: see qldpc._deprecations

    def __getattr__(name: str) -> object:
        return _deprecations.get_deprecated_attribute(__name__, name)
