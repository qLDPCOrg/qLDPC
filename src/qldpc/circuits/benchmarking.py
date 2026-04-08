"""Methods for benchmarking circuits

Copyright 2023 The qLDPC Authors and Infleqtion Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt
import sinter
import stim

import qldpc
from qldpc import codes

from .common import get_pauli_product_measurements, restrict_to_qubits
from .noise_model import DepolarizingNoiseModel, NoiseModel, as_noiseless_circuit


@restrict_to_qubits
def get_state_prep_diagnostic_circuit(
    code: codes.QuditCode,
    state_prep_circuit: stim.Circuit,
    *,
    observables: npt.NDArray[np.int_] | Sequence[stim.PauliString] | None = None,
) -> stim.Circuit:
    """Annotate a logical state prep circuit with diagnostics for computing logical error rates.

    This method assume that all measurements in the provided circuit are post-selection flags,
    meaning that circuit runs in which these measurement outcomes are nonzero are discarded.

    More specifically, this method returns a diagnostic circuit that appends the following to the
    provided circuit:
    - A detector for each measurement in the input circuit.
    - Noiseless measurements of all stabilizers of the code.
    - A detector for each stabilizer measurement.
    - Annotations of observables that should stabilize the state prepared by the provided circuit.

    The logical error rate of the diagnostic circuit is the probability with which any of the
    annotated observables are flipped after post-selecting on flags and decoding stabilizer
    measurement outcomes.

    Args:
        code: The code whose logical state is prepared by the provided state_prep_circuit.
        state_prep_circuit: A circuit that prepares a logical state of the provided code.
        observables: The observables that should stabilize the prepared state, or (by default) None.
            If not None, the observables should be either a a matrix of symplectic row vectors, with
            shape (num_observables, 2 * len(code)), or a sequence of Pauli strings supported on the
            data qubits of the code.  If None, observables are determined automatically by finding
            all logical Pauli operators of the code that stabilize the state prepared by
            state_prep_circuit.

    Returns:
        An annotated circuit ready for stim/sinter simulations that compute logical error rates.
    """

    # flag detectors
    flag_detectors = stim.Circuit()
    for meas_index in range(-state_prep_circuit.num_measurements, 0):
        flag_detectors.append("DETECTOR", [stim.target_rec(meas_index)])

    # stabilizer measurements and detectors
    stabilizer_measurements = get_pauli_product_measurements(code.get_stabilizer_ops())
    stabilizer_detectors = stim.Circuit()
    for meas_index in range(-stabilizer_measurements.num_measurements, 0):
        stabilizer_detectors.append("DETECTOR", [stim.target_rec(meas_index)])

    # identify the symplectic matrix of observables to measure and annotate
    if observables is None:
        # identify logical operators that stabilize the state prepared by the circuit
        ...
        if not np.any(observables):
            raise ValueError(
                "The provided circuit prepares a state that is not stabilized by any logical"
                " operators of the code"
            )
    elif not isinstance(observables, np.ndarray):
        # convert Pauli strings into symplectic vectors
        observables = np.array(
            [qldpc.math.string_to_op(string) for string in observables], dtype=int
        )

    # observable measurements and annotations
    logical_op_measurements = get_pauli_product_measurements(observables)
    logical_op_annotations = stim.Circuit()
    for meas_index in range(-logical_op_measurements.num_measurements, 0):
        op_index = meas_index + logical_op_measurements.num_measurements
        logical_op_annotations.append(
            "OBSERVABLE_INCLUDE", [stim.target_rec(meas_index)], [op_index]
        )

    # collect data used for logical error rate calculations
    measurements_and_detectors = as_noiseless_circuit(
        flag_detectors
        + stabilizer_measurements
        + stabilizer_detectors
        + logical_op_measurements
        + logical_op_annotations
    )

    return state_prep_circuit + measurements_and_detectors


def get_state_prep_diagnostic_tasks(
    code: codes.QuditCode,
    state_prep_circuit: stim.Circuit,
    error_rates: Sequence[float],
    noise_model_family: Callable[[float], NoiseModel] = DepolarizingNoiseModel,
    *,
    observables: npt.NDArray[np.int_] | Sequence[stim.PauliString] | None = None,
) -> list[sinter.Task]:
    """Build sinter Tasks that compute logical error rates of a logical state preparation circuit.

    This method is essentially a helper function that wraps get_state_prep_diagnostic_circuit.
    See help(get_state_prep_diagnostic_circuit) for additional information.

    Args:
        code: The code whose logical state is prepared by the provided state_prep_circuit.
        state_prep_circuit: A circuit that prepares a logical state of the provided code.
        error_rates: The error rates at which to evaluate the provided family of noise models.
        noise_model_family: A single-parameter family of noise models for adding noise to circuits.
        observables: The observables that should stabilize the prepared state, or (by default) None.
            If not None, the observables should be either a a matrix of symplectic row vectors, with
            shape (num_observables, 2 * len(code)), or a sequence of Pauli strings supported on the
            data qubits of the code.  If None, observables are determined automatically by finding
            all logical Pauli operators of the code that stabilize the state prepared by
            state_prep_circuit.

    Returns:
        A list of sinter Tasks, one-to-one with the provided error_rates.
    """
    diagnostic_circuit = get_state_prep_diagnostic_circuit(
        code, state_prep_circuit, observables=observables
    )
    postselection_mask = np.zeros(diagnostic_circuit.num_detectors, dtype=int)
    postselection_mask[: state_prep_circuit.num_measurements] = 1
    postselection_mask_bit_packed = np.packbits(postselection_mask, bitorder="little")
    return [
        sinter.Task(
            circuit=noise_model_family(error_rate).noisy_circuit(diagnostic_circuit),
            postselection_mask=postselection_mask_bit_packed,
            json_metadata={"p": error_rate},
        )
        for error_rate in error_rates
    ]
