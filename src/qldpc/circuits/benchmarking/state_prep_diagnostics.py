# SPDX-License-Identifier: Apache-2.0

"""State-preparation diagnostics and Sinter task construction."""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence

import numpy as np
import numpy.typing as npt
import sinter
import stim

from qldpc import codes, math

from .._post_selection import get_flag_detectors, get_selected_detectors
from ..bookkeeping import DetectorRecord, QubitIDs
from ..common import _restrict_to_qubits, get_pauli_product_measurements
from ..encoding import get_logical_state_stabilizers, get_state_stabilizers
from ..noise_model import DepolarizingNoiseModel, NoiseModel, as_noiseless_circuit
from .logical_errors import _get_postselection_mask

####################################################################################################
# state preparation diagnostics


@_restrict_to_qubits
def get_state_prep_diagnostic_circuit(
    code: codes.QuditCode,
    state_prep_circuit: stim.Circuit,
    *,
    add_flags: bool = False,
    qubit_ids: QubitIDs | None = None,
    observables: npt.NDArray[np.int_]
    | Sequence[Sequence[int]]
    | Sequence[stim.PauliString]
    | None = None,
    skip_validation: bool = False,
) -> tuple[stim.Circuit, DetectorRecord]:
    """Annotate a logical state prep circuit with diagnostics for computing logical error rates.

    More specifically, this method returns a diagnostic circuit that appends the following to the
    provided circuit:

    - If ``add_flags is True``, a detector for each measurement that is not addressed by an existing
      detector or observable in the provided circuit.  The added detectors are called "flag
      detectors".
    - Noiseless measurements of all stabilizers of the code.
    - A detector for each of the noiseless stabilizer measurements.
    - Noiseless measurements of observables that stabilize the state prepared by
      ``state_prep_circuit``.
    - Annotations of the measured observables (with ``OBSERVABLE_INCLUDE``).

    The logical error rate of the diagnostic circuit is nominally the probability with which any of
    the annotated observables are flipped after decoding flag and stabilizer measurement outcomes.
    However, the details of decoding and the option to post-select on some detectors are left up to
    the user.

    The second returned value is a ``DetectorRecord`` whose ``get_events`` method maps keys to
    detector indices:

    - ``get_events("prep")`` returns the indices of detectors already present in the provided
      ``state_prep_circuit``.
    - ``get_events("flags")`` returns the indices of the flag detectors.
    - ``get_events(stab_index)[0]`` is the index of the detector for the stabilizer
      ``code.get_stabilizer_ops()[stab_index]``.

    Args:
        code: The code whose logical state is prepared by the provided state_prep_circuit.
        state_prep_circuit: A circuit that prepares a logical state of the provided code.

    Keyword args:
        add_flags: Whether to add a flag detector for each measurement in the circuit that no
            detector or observable addresses.  Default: False.
        qubit_ids: A QubitIDs object specifying the indices of the data qubits of the code.  If
            None, the data qubits of the code are assumed to be range(len(code)).
        observables: The observables that should stabilize the prepared state, or (by default) None.
            If not None, the observables should be either a matrix of symplectic row vectors, with
            shape (num_observables, 2 * len(code)), or a sequence of Pauli strings supported on the
            data qubits of the code.  If None, observables are determined automatically by finding
            all logical Pauli operators of the code that stabilize the state prepared by
            state_prep_circuit.
        skip_validation: If True, skip the check to assert that the provided circuit prepares a
            logical state of the provided code.

    Returns:
        stim.Circuit: An annotated circuit for stim/sinter simulations of logical error rates.
        circuits.DetectorRecord: A record of the circuit's detectors (keys described above).

    Raises:
        ValueError: If the code is not a qubit code; if ``qubit_ids`` does not match the code; if
            ``skip_validation`` is False and the circuit does not deterministically prepare a pure
            logical state of the code; or if an observable is the identity, has a symplectic row
            that is not binary, or is not a Pauli operator on ``len(code)`` qubits.
    """
    qubit_ids = QubitIDs.validated(qubit_ids or QubitIDs.from_code(code), code)
    if not skip_validation:
        _assert_pure_logical_state(state_prep_circuit, code, qubit_ids)

    # if necessary, identify the pure logical operators that stabilize the prepared state
    if observables is None:
        observables = get_logical_state_stabilizers(code, state_prep_circuit, qubit_ids)

    # if applicable, convert Pauli strings into symplectic vectors
    if len(observables) > 0 and any(isinstance(obs, stim.PauliString) for obs in observables):
        observables = np.array(
            [
                math.string_to_op(obs) if isinstance(obs, stim.PauliString) else obs
                for obs in observables
            ],
        ).astype(int)

    # observables are Pauli operators, so their symplectic rows are binary
    if not all(np.isin(np.array(obs, dtype=int), (0, 1)).all() for obs in observables):
        raise ValueError("Observables must be Pauli operators, with binary symplectic rows")

    # initialize a record of the detectors in the circuit
    detector_record = DetectorRecord({"prep": range(state_prep_circuit.num_detectors)})

    # add flag detectors for unused measurements, if applicable
    flag_detectors = get_flag_detectors(state_prep_circuit) if add_flags else stim.Circuit()
    detector_record.append({"flags": range(len(flag_detectors))})

    # stabilizer measurements and detectors
    stabilizer_measurements = get_pauli_product_measurements(
        code.get_stabilizer_ops(), qubits=qubit_ids.data
    )
    stabilizer_detectors = stim.Circuit()
    for meas_index in range(-stabilizer_measurements.num_measurements, 0):
        stabilizer_detectors.append("DETECTOR", [stim.target_rec(meas_index)])
    detector_record.append({ss: ss for ss in range(len(code.get_stabilizer_ops()))})

    # observable measurements and annotations
    logical_op_measurements = get_pauli_product_measurements(observables, qubits=qubit_ids.data)
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

    return state_prep_circuit + measurements_and_detectors, detector_record


def get_state_prep_diagnostic_tasks(
    code: codes.QuditCode,
    state_prep_circuit: stim.Circuit,
    error_rates: Sequence[float] | npt.NDArray[np.floating],
    noise_model_family: Callable[[float], NoiseModel] = DepolarizingNoiseModel,
    *,
    add_flags: bool = False,
    post_select: bool | Collection[int] | None = None,
    post_select_observables: bool | Collection[int] = False,
    qubit_ids: QubitIDs | None = None,
    observables: npt.NDArray[np.int_]
    | Sequence[Sequence[int]]
    | Sequence[stim.PauliString]
    | None = None,
    skip_validation: bool = False,
    metadata: Mapping[str, object] | None = None,
) -> list[sinter.Task]:
    r"""Helper method to build sinter Tasks for benchmarking a logical state preparation circuit.

    This method is essentially a wrapper for get_state_prep_diagnostic_circuit.  See
    help(get_state_prep_diagnostic_circuit) for additional information.

    As an example, if::

        tasks = get_state_prep_diagnostic_tasks(...)
        decoder = qldpc.decoders.SinterDecoder(...)

    then we can collect statistics with::

        stats = sinter.collect(
            tasks=tasks,
            decoders=["custom"],
            custom_decoders={"custom": decoder},
            num_workers=os.cpu_count(),
            max_shots=10**5,
            max_errors=100,
        )

    and plot the results with::

        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(5, 4))
        sinter.plot_error_rate(
            ax=axis,
            stats=stats,
            x_func=lambda stats: stats.json_metadata["p"],
        )
        axis.axline(
            (0, 0),
            slope=1,
            color="k",
            linestyle=":",
            label=r"$p_{\mathrm{log}}=p_{\mathrm{phys}}$",
        )
        axis.set_ylabel("logical error rate")
        axis.set_xlabel("physical error rate")
        axis.loglog()
        axis.grid(which="both")
        figure.tight_layout()
        plt.show()

    Args:
        code: The code whose logical state is prepared by the provided state_prep_circuit.
        state_prep_circuit: A circuit that prepares a logical state of the provided code.
        error_rates: The error rates at which to evaluate the provided family of noise models.
        noise_model_family: A single-parameter family of noise models for adding noise to circuits.

    Keyword args:
        add_flags: Whether to add a flag detector for each unaddressed measurement in the circuit,
            as in get_state_prep_diagnostic_circuit.  By default, ``post_select`` then selects these
            flag detectors, so that shots in which a flag fires are discarded.
        post_select: The detectors to post-select on, out of the selectable detectors: those already
            present in the provided circuit followed by any added flag detectors.  None (the
            default) selects exactly the added flag detectors, if any; True selects all selectable
            detectors; False selects none, which leaves any flag detectors to the decoder; and a
            collection of integers selects the selectable detectors with those indices (counted in
            the order above, and possibly negative to count back from the last of them).  The final
            (noiseless) stabilizer detectors are never selectable.  Shots in which any selected
            detector fires are discarded.
        post_select_observables: The observables to post-select on: True for all observables,
            False (the default) for none, or a collection of observable indices.  Any time the
            decoder's prediction for these observable flips do not agree with the actual observable
            flips, the shot is discarded rather than counting as an error.
        qubit_ids: A QubitIDs object specifying the indices of the data qubits of the code.  If
            None, the data qubits of the code are assumed to be range(len(code)).
        observables: The observables that should stabilize the prepared state, or (by default) None.
            If not None, the observables should be either a matrix of symplectic row vectors, with
            shape (num_observables, 2 * len(code)), or a sequence of Pauli strings supported on the
            data qubits of the code.  If None, observables are determined automatically by finding
            all logical Pauli operators of the code that stabilize the state prepared by
            state_prep_circuit.
        skip_validation: If True, skip the check to assert that the provided circuit prepares a
            logical state of the provided code.
        metadata: Extra key-value pairs to attach to the json_metadata of every returned task
            (merged with the per-task error rate "p").

    Returns:
        A list of sinter Tasks, one-to-one with the provided error_rates.  The error rate of an
            individual task is task.json_metadata["p"].

    Raises:
        TypeError: If ``post_select`` is not None, a bool, or a collection of integer detector
            indices, or if ``post_select_observables`` is neither a bool nor a collection of integer
            observable indices.
        ValueError: If the code is not a qubit code; if ``qubit_ids`` does not match the code; if
            ``skip_validation`` is False and the circuit does not deterministically prepare a pure
            logical state of the code; if an observable is the identity, has a symplectic row that
            is not binary, or is not a Pauli operator on ``len(code)`` qubits; if ``post_select``
            has an index outside the selectable detectors; or if ``post_select_observables`` has an
            index outside the observables.
    """
    diagnostic_circuit, detector_record = get_state_prep_diagnostic_circuit(
        code,
        state_prep_circuit,
        add_flags=add_flags,
        qubit_ids=qubit_ids,
        observables=observables,
        skip_validation=skip_validation,
    )
    postselection_mask = _get_postselection_mask(
        get_selected_detectors(post_select, detector_record),
        detector_record.num_events,
        "Detector",
    )
    postselected_observables_mask = _get_postselection_mask(
        post_select_observables, diagnostic_circuit.num_observables, "Observable"
    )
    return [
        sinter.Task(
            circuit=noise_model_family(float(error_rate)).noisy_circuit(diagnostic_circuit),
            postselection_mask=postselection_mask,
            postselected_observables_mask=postselected_observables_mask,
            json_metadata=dict(metadata or {}) | {"p": float(error_rate)},
        )
        for error_rate in error_rates
    ]


def _assert_pure_logical_state(
    state_prep_circuit: stim.Circuit, code: codes.QuditCode, qubit_ids: QubitIDs
) -> None:
    """Assert that the given circuit prepares a pure logical state of the given code."""
    # test that all stabilizers have expectation value +1, including on data qubits that the circuit
    # leaves untouched in |0>
    num_qubits = max(state_prep_circuit.num_qubits, max(qubit_ids.data) + 1)
    simulator = stim.TableauSimulator()
    simulator.do(state_prep_circuit.without_noise())
    for op in code.get_stabilizer_ops():
        compact_string = math.op_to_string(op)
        string = stim.PauliString(num_qubits)
        for qubit, pauli in zip(qubit_ids.data, str(compact_string)[1:], strict=True):
            if pauli != "_":
                string[qubit] = pauli
        if not simulator.peek_observable_expectation(string) == 1:
            raise ValueError("The provided circuit does not prepare a logical state of the code")

    # test that the data qubits of the code are not entangled with other qubits
    state_stabilizers = get_state_stabilizers(state_prep_circuit, qubit_ids.data)
    if not len(state_stabilizers) == len(code):
        raise ValueError(
            "The provided circuit does not _deterministically_ prepare a _pure_ logical state of"
            " the code"
        )
