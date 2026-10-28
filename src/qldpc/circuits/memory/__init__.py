# SPDX-License-Identifier: Apache-2.0

"""Memory experiments and the syndrome measurement strategies that they use."""

from typing import TYPE_CHECKING

from qldpc import _deprecations

from .alpha_syndrome import AlphaSyndrome
from .memory import (
    CheckKey,
    CheckProductKey,
    MemoryExperimentParts,
    get_logical_bell_prep,
    get_memory_experiment,
    get_memory_experiment_parts,
    get_observables,
    get_qubit_coordinates,
)
from .syndrome_measurement import (
    EdgeColoring,
    EdgeColoringXZ,
    SyndromeMeasurementStrategy,
)

__all__ = [
    "AlphaSyndrome",
    "CheckKey",
    "CheckProductKey",
    "EdgeColoring",
    "EdgeColoringXZ",
    "MemoryExperimentParts",
    "SyndromeMeasurementStrategy",
    "get_logical_bell_prep",
    "get_memory_experiment",
    "get_memory_experiment_parts",
    "get_observables",
    "get_qubit_coordinates",
]

if not TYPE_CHECKING:  # DEPRECATED: see qldpc._deprecations

    def __getattr__(name: str) -> object:
        return _deprecations.get_deprecated_attribute(__name__, name)
