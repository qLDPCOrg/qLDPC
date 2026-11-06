# SPDX-License-Identifier: Apache-2.0

"""Benchmarks of noisy state-preparation circuits by the residual weights of their errors.

``estimate_residual_weight_histogram`` estimates, over the shots of a noisy state-preparation
circuit that pass post-selection, the probability distribution of the residual weight of the fault
that occurs in a shot.  The residual weight needs no decoder, and it accounts for Pauli errors that
a chosen logical readout cannot detect.  A ``ConsumerProfile`` says how the circuits that consume
the prepared state respond to each residual weight, and
``ResidualWeightHistogram.get_excess_failure_probability`` combines the two into a single figure of
merit.  ``estimate_downstream_fault_weight`` estimates the expected fault weight of a consumer from
the noise of a circuit.  A ``ConsumerCalibration`` scores preparation circuits with excess-failure
tables for X-type and Z-type errors.  See the `consumer-calibration guide
<https://github.com/qLDPCOrg/qLDPC/blob/main/src/qldpc/experimental/circuits/benchmarking/consumer_calibration.md>`_
for the model and validation guidance.
"""

from .consumer_profiles import ConsumerProfile, estimate_downstream_fault_weight
from .consumers import ConsumerCalibration
from .residual_weights import ResidualWeightHistogram, estimate_residual_weight_histogram

__all__ = [
    "ConsumerCalibration",
    "ConsumerProfile",
    "ResidualWeightHistogram",
    "estimate_downstream_fault_weight",
    "estimate_residual_weight_histogram",
]
