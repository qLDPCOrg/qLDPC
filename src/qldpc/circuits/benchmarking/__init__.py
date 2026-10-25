# SPDX-License-Identifier: Apache-2.0

"""Fault distances, logical error rates, and state-preparation checks."""

from .fault_propagation import (
    BoundedFaultPropagationCertificate,
    BoundedFaultPropagationCounterexample,
    BoundedFaultPropagationStatistics,
    certify_bounded_fault_propagation,
)
from .logical_errors import get_fault_distance, get_logical_error_and_discard_rate
from .state_prep_diagnostics import (
    get_state_prep_diagnostic_circuit,
    get_state_prep_diagnostic_tasks,
)

__all__ = [
    "BoundedFaultPropagationCertificate",
    "BoundedFaultPropagationCounterexample",
    "BoundedFaultPropagationStatistics",
    "certify_bounded_fault_propagation",
    "get_fault_distance",
    "get_logical_error_and_discard_rate",
    "get_state_prep_diagnostic_circuit",
    "get_state_prep_diagnostic_tasks",
]
