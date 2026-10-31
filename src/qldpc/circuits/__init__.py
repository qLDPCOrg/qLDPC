# SPDX-License-Identifier: Apache-2.0

"""Stim circuits: encoding, memory, noise, benchmarking, certification, and transversal gates.

The top-level modules and subpackages of this package import each other in layers, from shared
foundations to complete workflows: each imports only from itself and from lower layers, which rules
out import cycles between them.  ``layering_test.py`` lists the layers and checks these imports,
apart from any that are marked as exceptions.
"""

from typing import TYPE_CHECKING

from qldpc import _deprecations

from .benchmarking import (
    get_logical_error_and_discard_rate,
    get_state_prep_diagnostic_circuit,
    get_state_prep_diagnostic_tasks,
)
from .bookkeeping import (
    DetectorRecord,
    MeasurementRecord,
    QubitIDs,
    Record,
    get_unaddressed_measurements,
)
from .certification import (
    BoundedFaultPropagationCertificate,
    BoundedFaultPropagationCounterexample,
    certify_bounded_fault_propagation,
    get_fault_distance,
)
from .common import (
    StimCircuitLike,
    StimCircuitProtocol,
    get_pauli_product_measurements,
)
from .encoding import (
    get_encoder_and_decoder,
    get_encoding_circuit,
    get_encoding_tableau,
    get_logical_state_stabilizers,
    get_logical_tableau,
    get_state_stabilizers,
    restrict_tableau,
)
from .memory import (
    AlphaSyndrome,
    EdgeColoring,
    EdgeColoringXZ,
    MemoryExperimentParts,
    SyndromeMeasurementStrategy,
    get_logical_bell_prep,
    get_memory_experiment,
    get_memory_experiment_parts,
    get_observables,
    get_qubit_coordinates,
)
from .noise_model import (
    DEFAULT_IMMUNE_OP_TAG,
    DEFAULT_IMMUNE_QUBIT_TAG,
    AbstractPauliChannel,
    DepolarizingNoiseModel,
    NoiseModel,
    NoiseRule,
    PauliChannel,
    PauliChannelSequence,
    SI1000NoiseModel,
    as_noiseless_circuit,
)
from .transforms import remap_qubit_target, with_remapped_qubits
from .transversal import (
    get_transversal_automorphism_group,
    get_transversal_circuit,
    get_transversal_circuits,
    get_transversal_ops,
    get_transversal_s,
)

__all__ = [
    "DEFAULT_IMMUNE_OP_TAG",
    "DEFAULT_IMMUNE_QUBIT_TAG",
    "AbstractPauliChannel",
    "AlphaSyndrome",
    "BoundedFaultPropagationCertificate",
    "BoundedFaultPropagationCounterexample",
    "DepolarizingNoiseModel",
    "DetectorRecord",
    "EdgeColoring",
    "EdgeColoringXZ",
    "MeasurementRecord",
    "MemoryExperimentParts",
    "NoiseModel",
    "NoiseRule",
    "PauliChannel",
    "PauliChannelSequence",
    "QubitIDs",
    "Record",
    "SI1000NoiseModel",
    "StimCircuitLike",
    "StimCircuitProtocol",
    "SyndromeMeasurementStrategy",
    "as_noiseless_circuit",
    "certify_bounded_fault_propagation",
    "get_encoder_and_decoder",
    "get_encoding_circuit",
    "get_encoding_tableau",
    "get_fault_distance",
    "get_logical_bell_prep",
    "get_logical_error_and_discard_rate",
    "get_logical_state_stabilizers",
    "get_logical_tableau",
    "get_memory_experiment",
    "get_memory_experiment_parts",
    "get_observables",
    "get_pauli_product_measurements",
    "get_qubit_coordinates",
    "get_state_prep_diagnostic_circuit",
    "get_state_prep_diagnostic_tasks",
    "get_state_stabilizers",
    "get_transversal_automorphism_group",
    "get_transversal_circuit",
    "get_transversal_circuits",
    "get_transversal_ops",
    "get_transversal_s",
    "get_unaddressed_measurements",
    "remap_qubit_target",
    "restrict_tableau",
    "with_remapped_qubits",
]

if not TYPE_CHECKING:  # DEPRECATED: see qldpc._deprecations

    def __getattr__(name: str) -> object:
        return _deprecations.get_deprecated_attribute(__name__, name)
