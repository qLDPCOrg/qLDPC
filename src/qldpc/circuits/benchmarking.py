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

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt
import sinter
import stim

from qldpc import codes, decoders, math

from .bookkeeping import DetectorRecord
from .common import get_encoder_and_decoder, get_pauli_product_measurements, restrict_to_qubits
from .noise_model import DepolarizingNoiseModel, NoiseModel, as_noiseless_circuit


@restrict_to_qubits
def get_state_prep_diagnostic_circuit(
    code: codes.QuditCode,
    state_prep_circuit: stim.Circuit,
    *,
    observables: npt.NDArray[np.int_]
    | Sequence[Sequence[int]]
    | Sequence[stim.PauliString]
    | None = None,
    skip_validation: bool = False,
) -> tuple[stim.Circuit, DetectorRecord]:
    """Annotate a logical state prep circuit with diagnostics for computing logical error rates.

    The first len(code) qubits addressed by the circuit must be the data qubits of the code.

    More specifically, this method returns a diagnostic circuit that appends the following to the
    provided circuit:
    - A detector for each measurement in the provided circuit.  These are called "flag" detectors".
    - Noiseless measurements of all stabilizers of the code.
    - A detector for each of the noiseless stabilizer measurements.
    - Noisless measurements of observables that stabilize the state prepared by state_prep_circuit.
    - Annotations of the measured observables (with OBSERVABLE_INCLUDE).

    The logical error rate of the diagnostic circuit is nominally the probability with which any of
    the annotated observables are flipped after decoding flag and stabilizer measurement outcomes.
    However, the details of decoding and the option to post-select on some detectors are left up
    to the user.

    Args:
        code: The code whose logical state is prepared by the provided state_prep_circuit.
        state_prep_circuit: A circuit that prepares a logical state of the provided code.

    Keyword args:
        observables: The observables that should stabilize the prepared state, or (by default) None.
            If not None, the observables should be either a a matrix of symplectic row vectors, with
            shape (num_observables, 2 * len(code)), or a sequence of Pauli strings supported on the
            data qubits of the code.  If None, observables are determined automatically by finding
            all logical Pauli operators of the code that stabilize the state prepared by
            state_prep_circuit.
        skip_validation: If True, skip the check to assert that the provided circuit prepares a
            logical state fo the provided code.

    Returns:
        stim.Circuit: An annotated circuit for stim/sinter simulations of logical error rates.
        circuits.DetectorRecord: A record of the detectors in the circuit, for which
            - DetectorRecord.get_events("flag") is a list of indices for the flag detectors.
            - DetectorRecord.get_events(stab_index)[0] is the index of the detector for the
                stabilizer represented by code.get_stabilizer_ops()[stab_index].
    """
    if not skip_validation:
        _assert_logical_state_preparation(code, state_prep_circuit)

    # initialize a record of the detectors in the circuit
    detector_record = DetectorRecord()

    # flag detectors
    flag_detectors = stim.Circuit()
    for meas_index in range(-state_prep_circuit.num_measurements, 0):
        flag_detectors.append("DETECTOR", [stim.target_rec(meas_index)])
    detector_record.append({"flag": range(state_prep_circuit.num_measurements)})

    # stabilizer measurements and detectors
    stabilizer_measurements = get_pauli_product_measurements(code.get_stabilizer_ops())
    stabilizer_detectors = stim.Circuit()
    for meas_index in range(-stabilizer_measurements.num_measurements, 0):
        stabilizer_detectors.append("DETECTOR", [stim.target_rec(meas_index)])
    detector_record.append({ss: [ss] for ss in range(len(code.get_stabilizer_ops()))})

    # if none were provided, automatically find the logical Pauli stabilizers of the prepared state
    if observables is None:
        observables = get_nontrivial_logical_stabilizers(
            code, state_prep_circuit, skip_validation=True
        )

    # if applicable, convert Pauli strings into symplectic vectors
    if len(observables) > 0 and isinstance(observables[0], stim.PauliString):
        observables = np.array([math.string_to_op(string) for string in observables], dtype=int)

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

    return state_prep_circuit + measurements_and_detectors, detector_record


@restrict_to_qubits
def get_nontrivial_logical_stabilizers(
    code: codes.QuditCode, state_prep_circuit: stim.Circuit, *, skip_validation: bool = False
) -> npt.NDArray[np.int_]:
    """Identify a complete basis for the nontrivial logical Pauli stabilizers of the prepared state.

    The first len(code) qubits addressed by the circuit must be the data qubits of the code.

    Args:
        code: The code whose logical state is prepared by the provided state_prep_circuit.
        state_prep_circuit: A circuit that prepares a logical state of the provided code.

    Keyword args:
        skip_validation: If True, skip the check to assert that the provided circuit prepares a
            logical state fo the provided code.

    Returns:
        A list of logical Pauli operators supported on the data qubits of the provided code.
    """
    if not skip_validation:  # pragma: no cover
        _assert_logical_state_preparation(code, state_prep_circuit)

    # convert the circuit into a tableau
    full_tableau = state_prep_circuit.to_tableau(
        ignore_noise=True, ignore_measurement=True, ignore_reset=True
    )

    # TODO: assert that the tableau does not prepare a logical state that is entangled with ancillas

    # remove ancilla qubits from the tableau
    x2x, x2z, z2x, z2z, x_signs, z_signs = full_tableau.to_numpy()
    tableau = stim.Tableau.from_numpy(
        x2x=x2x[: len(code), : len(code)],
        x2z=x2z[: len(code), : len(code)],
        z2x=z2x[: len(code), : len(code)],
        z2z=z2z[: len(code), : len(code)],
        x_signs=x_signs[: len(code)],
        z_signs=z_signs[: len(code)],
    )

    # identify logical stabilizers of the code, in the logical Pauli basis
    logical_stabilizers = []
    encoder, decoder = get_encoder_and_decoder(code)
    for stabilizer in tableau.to_stabilizers():
        stabilizer_in_logical_basis = stabilizer.after(decoder, targets=range(len(code)))
        logical_stabilizer = math.string_to_op(stabilizer_in_logical_basis[: code.dimension])
        logical_stabilizers.append(logical_stabilizer)

    # row-reduce to find a minimal basis of logical stabilizers
    logical_stabilizers_rref = code.field(logical_stabilizers).row_reduce()
    logical_stabilizers_rref = logical_stabilizers_rref[np.any(logical_stabilizers_rref, axis=1), :]
    assert logical_stabilizers_rref.shape == (code.dimension, 2 * code.dimension)

    # convert back into the basis of physical Pauli operators
    return logical_stabilizers_rref @ code.get_logical_ops()


def get_state_prep_diagnostic_tasks(
    code: codes.QuditCode,
    state_prep_circuit: stim.Circuit,
    error_rates: Sequence[float] | npt.NDArray[np.floating],
    noise_model_family: Callable[[float], NoiseModel] = DepolarizingNoiseModel,
    *,
    observables: npt.NDArray[np.int_]
    | Sequence[Sequence[int]]
    | Sequence[stim.PauliString]
    | None = None,
    post_select_on_flags: bool = False,
    skip_validation: bool = False,
) -> list[sinter.Task]:
    r"""Build sinter Tasks that compute logical error rates of a logical state preparation circuit.

    This method is essentially a helper function that wraps get_state_prep_diagnostic_circuit.
    See help(get_state_prep_diagnostic_circuit) for additional information.

    As an example, if

        tasks = get_state_prep_diagnostic_tasks(...)
        decoder = qldpc.decoders.SinterDecoder(...)

    then we can collect statistics with

        stats = sinter.collect(
            tasks=tasks,
            decoders=["custom"],
            custom_decoders={"custom": decoder},
            num_workers=os.cpu_count(),
            max_shots=10**5,
            max_errors=100,
        )

    and plot the results with

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
        observables: The observables that should stabilize the prepared state, or (by default) None.
            If not None, the observables should be either a a matrix of symplectic row vectors, with
            shape (num_observables, 2 * len(code)), or a sequence of Pauli strings supported on the
            data qubits of the code.  If None, observables are determined automatically by finding
            all logical Pauli operators of the code that stabilize the state prepared by
            state_prep_circuit.
        post_select_on_flags: If True, post-select samples on nonzero measurement outcomes in the
            provided state_prep_circuit.  Default: False.
        skip_validation: If True, skip the check to assert that the provided circuit prepares a
            logical state fo the provided code.

    Returns:
        A list of sinter Tasks, one-to-one with the provided error_rates.  The error rate of an
            individual task is task.json_metadata["p"].
    """
    diagnostic_circuit, detector_record = get_state_prep_diagnostic_circuit(
        code, state_prep_circuit, observables=observables
    )
    if post_select_on_flags:
        postselection_mask = np.zeros(diagnostic_circuit.num_detectors, dtype=int)
        postselection_mask[detector_record.get_events("flag")] = 1
        postselection_mask_bit_packed = np.packbits(postselection_mask, bitorder="little")
        raise ValueError(
            "Post selecting on flags is unsupported due to a bug in sinter:\n"
            "https://github.com/quantumlib/Stim/pull/844"
        )
    else:
        postselection_mask_bit_packed = None
    return [
        sinter.Task(
            circuit=noise_model_family(error_rate).noisy_circuit(diagnostic_circuit),
            postselection_mask=postselection_mask_bit_packed,
            json_metadata={"p": error_rate},
        )
        for error_rate in error_rates
    ]


def get_logical_error_and_discard_rates(
    code: codes.QuditCode,
    state_prep_circuit: stim.Circuit,
    error_rates: Sequence[float] | npt.NDArray[np.floating],
    noise_model_family: Callable[[float], NoiseModel] = DepolarizingNoiseModel,
    *,
    sinter_decoder: sinter.Decoder | Sequence[sinter.Decoder],
    num_samples: int | Sequence[float],
    observables: npt.NDArray[np.int_]
    | Sequence[Sequence[int]]
    | Sequence[stim.PauliString]
    | None = None,
    post_select_on_flags: bool = False,
    skip_validation: bool = False,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Compute logical error rates of the provided logical state prep circuit for the provided code.

    The first len(code) qubits addressed by the circuit must be the data qubits of the code.

    Each logical error rate is a fraction of the (possibly post-selected) shots in which observable
    flips are predicted incorrectly by the provided decoder.

    This method is provided as an alternative to get_state_prep_diagnostic_tasks, which currently
    cannot support post-selection due to a sinter bug: https://github.com/quantumlib/Stim/pull/844
    Once this bug is fixed, it is recommended to instead use get_state_prep_diagnostic_tasks.

    Args:
        code: The code whose logical state is prepared by the provided state_prep_circuit.
        state_prep_circuit: A circuit that prepares a logical state of the provided code.
        error_rates: The error rates at which to evaluate the provided family of noise models.
        noise_model_family: A single-parameter family of noise models for adding noise to circuits.
            Default: qldpc.circuits.DepolarizingNoiseModel.

    Keyword args:
        sinter_decoder: The circuit-level decoder used to predict observable flips, or a sequence of
            circuit-level decoders (one for each error rate).
        num_samples: The number of times to sample each noisy circuit, or a sequence of sample
            numbers (one for each error rate).
        observables: The observables that should stabilize the prepared state, or (by default) None.
            If not None, the observables should be either a a matrix of symplectic row vectors, with
            shape (num_observables, 2 * len(code)), or a sequence of Pauli strings supported on the
            data qubits of the code.  If None, observables are determined automatically by finding
            all logical Pauli operators of the code that stabilize the state prepared by
            state_prep_circuit.
        post_select_on_flags: If True, post-select samples on nonzero measurement outcomes in the
            provided state_prep_circuit.  Default: False.
        skip_validation: If True, skip the check to assert that the provided circuit prepares a
            logical state fo the provided code.

    Returns:
        An array of estimated logical error rates.
        An array of discard rates, or the fraction of shots (for each simulated error rate) that
            were discarded due to post-selection on state prep flags.  If post_select_on_flags is
            False, this array contains only zeros.
    """
    diagnostic_circuit, detector_record = get_state_prep_diagnostic_circuit(
        code, state_prep_circuit, observables=observables
    )
    if not hasattr(num_samples, "__getitem__"):
        num_samples = [num_samples] * len(error_rates)
    if not hasattr(sinter_decoder, "__getitem__"):
        sinter_decoder = [sinter_decoder] * len(error_rates)

    logical_error_rates = np.zeros(len(error_rates), dtype=float)
    discard_rates = np.zeros(len(error_rates), dtype=float)
    for pp, error_rate in enumerate(error_rates):
        # sample detector and observable flips in the circuit
        noise_model = noise_model_family(error_rate)
        noisy_circuit = noise_model.noisy_circuit(diagnostic_circuit)
        dem_arrays = decoders.DetectorErrorModelArrays(
            noisy_circuit.detector_error_model(), simplify=True
        )
        dem = dem_arrays.to_dem()
        sampler = dem.compile_sampler()
        det_data, obs_data, err_data = sampler.sample(shots=num_samples[pp])

        # if applicable, post-select on flag detectors
        if post_select_on_flags:
            # identify shots and detectors to remove
            flag_dets = detector_record.get_events("flag")
            shot_mask = ~np.any(det_data[:, flag_dets], axis=1)
            detector_mask = np.ones(dem.num_detectors, dtype=bool)
            detector_mask[flag_dets] = False

            # post-select simulated data
            det_data = det_data[shot_mask][:, detector_mask]
            obs_data = obs_data[shot_mask]
            dem = dem_arrays.post_selected_on(detector_record.get_events("flag")).to_dem()

            # record the fraction of shots that were discarded
            discard_rates[pp] = 1 - np.sum(shot_mask) / len(shot_mask)

        # compile a decoder for this detector error model
        compiled_sinter_decoder = sinter_decoder[pp].compile_decoder_for_dem(dem)

        # decode and compute the logical error rate
        predicted_flips = compiled_sinter_decoder.decode_shots(det_data)
        obs_flips = obs_data ^ predicted_flips
        failures = np.any(obs_flips, axis=1)
        logical_error_rates[pp] = np.sum(failures) / len(failures)

    return logical_error_rates, discard_rates


def _assert_logical_state_preparation(
    code: codes.QuditCode, state_prep_circuit: stim.Circuit
) -> None:
    """Assert that the the provided circuit prepare a logical state of the provided code.

    The first len(code) qubits addressed by the circuit must be the data qubits of the code.
    """
    simulator = stim.TableauSimulator()
    simulator.do(state_prep_circuit.without_noise())
    if not all(
        simulator.peek_observable_expectation(math.op_to_string(row)) == 1
        for row in code.get_stabilizer_ops()
    ):
        raise ValueError(
            "The provided circuit does not prepare a logical state of the provided code."
        )
