# SPDX-License-Identifier: Apache-2.0

"""Quantum error-correcting codes."""

from typing import TYPE_CHECKING

from qldpc import _deprecations

from .two_block import (
    BaconShorCode,
    BBCode,
    BPCode,
    C4Code,
    C6Code,
    CHGPCode,
    CRCode,
    DistanceBalancedCode,
    FieldOrRingArray,
    FiveQubitCode,
    FiveQuditCode,
    GALACode,
    GeneralizedSurfaceCode,
    GrossCode,
    HGPCode,
    IcebergCode,
    LPCode,
    ManyHypercubeCode,
    MBCode,
    PairPartitionCode,
    QTCode,
    QuantumGolayCode,
    QuantumHammingCode,
    QuantumReedMullerCode,
    SHPCode,
    SHYPSCode,
    SLPCode,
    SteaneCode,
    SurfaceCode,
    T4Code,
    TBCode,
    TetrahedralCode,
    ToricCode,
    TrivialCode,
    TwoGrossCode,
)

__all__ = [
    "BBCode",
    "BPCode",
    "BaconShorCode",
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
