# SPDX-License-Identifier: Apache-2.0

"""Experimental Stim-circuit tools.

This subpackage mirrors the layout of ``qldpc.circuits``, where its tools may move once their API
settles.
"""

from . import benchmarking
from .benchmarking import (
    ConsumerProfile,
    ResidualWeightHistogram,
    estimate_downstream_fault_weight,
    estimate_residual_weight_histogram,
)

__all__ = [
    "ConsumerProfile",
    "ResidualWeightHistogram",
    "benchmarking",
    "estimate_downstream_fault_weight",
    "estimate_residual_weight_histogram",
]
