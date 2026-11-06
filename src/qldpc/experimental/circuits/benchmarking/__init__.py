# SPDX-License-Identifier: Apache-2.0

"""Decoder-free benchmarks of noisy state-preparation circuits.

``estimate_residual_weight_histogram`` estimates, over the shots of a noisy state-preparation
circuit that pass post-selection, the probability distribution of the residual weight of the fault
that occurs in a shot.  The residual weight needs no decoder, and it accounts for Pauli errors that
a chosen logical readout cannot detect.  A ``ConsumerProfile`` says how the circuits that consume
the prepared state respond to each residual weight, and
``ResidualWeightHistogram.get_excess_failure_probability`` combines the two into a single figure of
merit.  ``estimate_downstream_fault_weight`` estimates the expected fault weight of a consumer from
the noise of a circuit.
"""

from .consumer_profiles import ConsumerProfile, estimate_downstream_fault_weight
from .residual_weights import ResidualWeightHistogram, estimate_residual_weight_histogram

__all__ = [
    "ConsumerProfile",
    "ResidualWeightHistogram",
    "estimate_downstream_fault_weight",
    "estimate_residual_weight_histogram",
]
