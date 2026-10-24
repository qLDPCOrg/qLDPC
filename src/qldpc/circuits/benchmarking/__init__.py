# SPDX-License-Identifier: Apache-2.0

"""Fault distances, logical error rates, and state-preparation checks."""

from .logical_errors import get_fault_distance, get_logical_error_and_discard_rate
from .state_prep_certification import (
    FaultSpreadCertificate,
    FaultSpreadCounterexample,
    FaultSpreadStatistics,
    certify_state_prep_spread,
)
from .state_prep_diagnostics import (
    get_state_prep_diagnostic_circuit,
    get_state_prep_diagnostic_tasks,
)

__all__ = [
    "FaultSpreadCertificate",
    "FaultSpreadCounterexample",
    "FaultSpreadStatistics",
    "certify_state_prep_spread",
    "get_fault_distance",
    "get_logical_error_and_discard_rate",
    "get_state_prep_diagnostic_circuit",
    "get_state_prep_diagnostic_tasks",
]
