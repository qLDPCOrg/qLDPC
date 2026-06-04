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

from collections.abc import Callable, Collection, Hashable, Sequence

import numpy as np
import numpy.typing as npt
import sinter
import stim

from qldpc import codes, decoders, math

from .bookkeeping import DetectorRecord
from .common import get_encoding_tableau, get_pauli_product_measurements, restrict_to_qubits
from .noise_model import DepolarizingNoiseModel, NoiseModel, as_noiseless_circuit


@restrict_to_qubits
def get_state_prep_diagnostic_circuit(
    code: codes.QuditCode,
    state_prep_circuit: stim.Circuit,
    *,
    add_flags: bool = False,
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
    - If 'add_flags is True', a detector for each measurement that is not addressed by an existing
        detector in the provided circuit.  The added detectors are called "flag detectors".
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
        add_flags: Whether to add a flag detector for each unaddressed measurement in the circuit.
        observables: The observables that should stabilize the prepared state, or (by default) None.
            If not None, the observables should be either a a matrix of symplectic row vectors, with
            shape (num_observables, 2 * len(code)), or a sequence of Pauli strings supported on the
            data qubits of the code.  If None, observables are determined automatically by finding
            all logical Pauli operators of the code that stabilize the state prepared by
            state_prep_circuit.
        skip_validation: If True, skip the check to assert that the provided circuit prepares a
            logical state of the provided code.

    Returns:
        stim.Circuit: An annotated circuit for stim/sinter simulations of logical error rates.
        circuits.DetectorRecord: A record of the detectors in the circuit, for which
            - DetectorRecord.get_events("prep") is a list of indices for detectors that were already
                present in the provided state_prep_circuit.
            - DetectorRecord.get_events("flags") is a list of indices for the flag detectors.
            - DetectorRecord.get_events(stab_index)[0] is the index of the detector for the
                stabilizer represented by code.get_stabilizer_ops()[stab_index].
    """
    # if no observables were provided, identify the logical Pauli stabilizers of the prepared state
    if observables is None:
        observables = get_nontrivial_logical_stabilizers(
            code, state_prep_circuit, skip_validation=skip_validation
        )
    elif not skip_validation:  # pragma: no cover
        _assert_valid_code_state(code, state_prep_circuit)

    # if applicable, convert Pauli strings into symplectic vectors
    if len(observables) > 0 and any(isinstance(obs, stim.PauliString) for obs in observables):
        observables = np.array(
            [
                math.string_to_op(obs) if isinstance(obs, stim.PauliString) else obs
                for obs in observables
            ],
        ).astype(int)

    # initialize a record of the detectors in the circuit
    detector_record = DetectorRecord({"prep": range(state_prep_circuit.num_detectors)})

    # add flag detectors for unused measurements, if applicable
    flag_detectors = stim.Circuit()
    if add_flags:
        for measurement in get_unaddressed_measurements(state_prep_circuit):
            target = stim.target_rec(measurement - state_prep_circuit.num_measurements)
            flag_detectors.append("DETECTOR", [target])
    detector_record.append({"flags": range(len(flag_detectors))})

    # stabilizer measurements and detectors
    stabilizer_measurements = get_pauli_product_measurements(code.get_stabilizer_ops())
    stabilizer_detectors = stim.Circuit()
    for meas_index in range(-stabilizer_measurements.num_measurements, 0):
        stabilizer_detectors.append("DETECTOR", [stim.target_rec(meas_index)])
    detector_record.append({ss: ss for ss in range(len(code.get_stabilizer_ops()))})

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


def get_state_prep_diagnostic_tasks(
    code: codes.QuditCode,
    state_prep_circuit: stim.Circuit,
    error_rates: Sequence[float] | npt.NDArray[np.floating],
    noise_model_family: Callable[[float], NoiseModel] = DepolarizingNoiseModel,
    *,
    post_select: bool | Collection[int] = False,
    observables: npt.NDArray[np.int_]
    | Sequence[Sequence[int]]
    | Sequence[stim.PauliString]
    | None = None,
    skip_validation: bool = False,
    metadata: dict[str, Hashable] | None = None,
) -> list[sinter.Task]:
    r"""Helper method to build sinter Tasks for benchmarking a logical state preparation circuit.

    This method is essentially a wrapper for get_state_prep_diagnostic_circuit.
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
        post_select: If True, add a flag detector for each unused measurement in the provided
            circuit and post-select on those detectors.  If provided a collection of integers,
            post-select on corresponding detectors that are already present in the provided circuit.
        observables: The observables that should stabilize the prepared state, or (by default) None.
            If not None, the observables should be either a a matrix of symplectic row vectors, with
            shape (num_observables, 2 * len(code)), or a sequence of Pauli strings supported on the
            data qubits of the code.  If None, observables are determined automatically by finding
            all logical Pauli operators of the code that stabilize the state prepared by
            state_prep_circuit.
        skip_validation: If True, skip the check to assert that the provided circuit prepares a
            logical state of the provided code.

    Returns:
        A list of sinter Tasks, one-to-one with the provided error_rates.  The error rate of an
            individual task is task.json_metadata["p"].
    """
    add_flags = post_select is True
    diagnostic_circuit, detector_record = get_state_prep_diagnostic_circuit(
        code,
        state_prep_circuit,
        add_flags=add_flags,
        observables=observables,
        skip_validation=skip_validation,
    )
    postselection_mask = _get_postselection_mask(post_select, detector_record)
    return [
        sinter.Task(
            circuit=noise_model_family(error_rate).noisy_circuit(diagnostic_circuit),
            postselection_mask=postselection_mask,
            json_metadata={"p": error_rate} | (metadata or {}),
        )
        for error_rate in error_rates
    ]


def get_logical_error_and_discard_rate(
    circuit_or_dem: stim.Circuit | stim.DetectorErrorModel,
    sinter_decoder: sinter.Decoder,
    *,
    num_samples: int,
    post_select: Collection[int] = (),
    dem_to_decode: stim.DetectorErrorModel | None = None,
) -> tuple[float, float]:
    """Compute a logical error rate and discard rate from samples of the provided circuit.

    Each logical error rate is a fraction of the (possibly post-selected) shots in which observable
    flips are predicted incorrectly by the provided decoder.

    This method is provided for convenience, but if you are doing heavy numerics you should probably
    build a sinter.Task and call sinter.collect.  In this case, circuit_or_dem should just be a
    circuit, and the sinter.Task would be built as follows:
        postselection_mask = np.zeros(circuit.num_detectors, dtype=int)
        postselection_mask[post_select] = 1
        task = sinter.Task(
            circuit=circuit,
            detector_error_model=dem_to_decode,
            postselection_mask=np.packbits(postselection_mask, bitorder="little"),
        )
    Sampling data would then be collected with:
        stats = sinter.collect(
            tasks=[task],  # or more maybe more tasks
            decoders=["custom"],
            custom_decoders={"custom": sinter_decoder},
            num_shots=num_samples,
            # other options such as num_workers=os.cpu_count() or max_errors=100
        )

    Args:
        circuit_or_dem: The circuit or detector error model we wish to sample.
        sinter_decoder: The circuit-level decoder used to predict observable flips.

    Keyword args:
        num_samples: The number of times to the circuit_or_dem.
        post_select: The detectors in circuit_or_dem to post-select on.
        dem_to_decode: The detector error model to decode.  If post-selecting, this DEM should _not_
            include any of the the detectors that are post-selected on.  If have a DEM that includes
            _all_ detectors in the circuit_or_dem, you can remove the post-selected detectors with
                old_dem_arrays = decoders.DetectorErrorModelArrays(old_dem)
                new_dem = old_dem_arrays.post_selected_on(post_select).simplified().to_dem()
            If dem_to_decode is None, this method decodes with the same DEM that it samples from.

    Returns:
        A fraction of samples in which at least one observable was decoded incorrectly.
        A fraction of samples that were discarded due to post-selection.
    """
    # identify and simplify the DEM to sample
    dem_arrays = decoders.DetectorErrorModelArrays(circuit_or_dem, simplify=True)
    dem = dem_arrays.to_dem()

    if dem_to_decode is not None:
        same_num_observables = dem_to_decode.num_observables == dem.num_observables
        same_num_detectors = dem_to_decode.num_detectors == dem.num_detectors - len(post_select)
        if not same_num_observables or not same_num_detectors:
            raise ValueError(
                f"Incompatible detector error models."
                "\n(num_detectors, num_observables) in the DEM to sample (after post-selection):"
                f" {(dem.num_detectors - len(post_select), dem.num_observables)}"
                "\n(num_detectors, num_observables) in the DEM to decode:"
                f" {(dem_to_decode.num_detectors, dem_to_decode.num_observables)}"
            )

    # sample detector and observable flips in the circuit
    sampler = dem.compile_sampler()
    det_data, obs_data, _ = sampler.sample(shots=num_samples, bit_packed=True)

    # if applicable, post-select on flag detectors
    if post_select:
        post_select = list(post_select)
        detector_record = DetectorRecord({"prep": range(circuit_or_dem.num_detectors)})
        postselection_mask = _get_postselection_mask(post_select, detector_record)
        assert postselection_mask is not None  # to help mypy

        # remove rows corresponding to shots in which post-selection detectors fired
        shot_mask = ~np.any(det_data & postselection_mask, axis=1)
        det_data = det_data[shot_mask]
        obs_data = obs_data[shot_mask]

        # remove post-selected detectors from the detector sample data
        detector_mask = np.ones(dem.num_detectors, dtype=bool)
        detector_mask[post_select] = False
        det_data_unpacked = np.unpackbits(
            det_data, count=dem.num_detectors, bitorder="little", axis=1
        )
        det_data = np.packbits(det_data_unpacked[:, detector_mask], bitorder="little", axis=1)

        # record the fraction of shots that were discarded
        discard_rate = 1 - np.sum(shot_mask) / len(shot_mask)

        if dem_to_decode is None:
            # remove irreleant error mechanisms from the DEM
            dem_arrays = dem_arrays.post_selected_on(post_select).simplified()
            dem = dem_arrays.to_dem()

    else:  # pragma: no cover
        discard_rate = 0

    # compile a decoder for this detector error model
    compiled_sinter_decoder = sinter_decoder.compile_decoder_for_dem(dem_to_decode or dem)

    # decode and compute the logical error rate
    predicted_flips = compiled_sinter_decoder.decode_shots_bit_packed(det_data)
    obs_flips = obs_data ^ predicted_flips
    failures = np.any(obs_flips, axis=1)
    logical_error_rate = np.sum(failures) / len(failures)

    return logical_error_rate, discard_rate


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
            logical state of the provided code.

    Returns:
        A list of logical Pauli operators supported on the data qubits of the provided code.
    """
    # identify stabilizers of the prepared state that are supported on the data qubits of the code
    decoded_stabilizers = get_state_stabilizers(code, state_prep_circuit, decoded=True)
    if not skip_validation:
        _assert_valid_code_state(code, decoded_stabilizers)

    # identify stabilizers of the state that are pure logicals
    logical_stab_mat = [math.string_to_op(stab[: code.dimension]) for stab in decoded_stabilizers]
    logical_stab_mat_rref = code.field(logical_stab_mat).row_reduce()
    logical_stab_mat_rref = logical_stab_mat_rref[np.any(logical_stab_mat_rref, axis=1)]
    return logical_stab_mat_rref @ code.get_logical_ops()


@restrict_to_qubits
def get_state_stabilizers(
    code: codes.QuditCode, state_prep_circuit: stim.Circuit, *, decoded: bool = False
) -> list[stim.PauliString]:
    """Identify stabilizers of the prepared state that are supported on the data qubits of the code.

    If 'decoded is True', return these stabilizers in the "decoded" basis of logical operators, gauge
    operators, and stabilizers/destabilizers.  That is, for each string in the returned list of Pauli
    strings, the first code.dimension qubits represent logical operators, the next
    code.gauge_dimension qubits represent gauge operators, and the remaining qubits represent
    stabilizer/destabilizer qubits.

    The strategy in this method is as follows.  If we prepend reset operations to make an initial
    |0...0⟩ initial state explicit, then all stabilizer flows of the circuit should have the form
        1 -> output_generator,
    where each output_generator is an XOR of
        (a) a Pauli string,
        (b) measurements, and
        (c) observables.
    That is, the circuit prepares a state for which, after identifying {+1,-1} <-> {0,1} as
    necessary, the XOR of (a), (b), and (c) for each output_generator is 0.

    To identify a basis of output generators that are supported only on the data qubits of the code,
    we collect all output generators into a binary matrix, and row-reduce this matrix.
    """
    resets = stim.Circuit("R " + " ".join(map(str, range(state_prep_circuit.num_qubits))))
    flow_generators = (resets + state_prep_circuit.without_noise()).flow_generators()

    # identify some useful numbers
    num_qubits = state_prep_circuit.num_qubits
    num_measurements = state_prep_circuit.num_measurements
    num_observables = state_prep_circuit.num_observables
    num_columns = 2 * num_qubits + 1 + num_measurements + num_observables
    num_rows = len(flow_generators) + state_prep_circuit.num_detectors

    # build a matrix of stabilizers for the entire circuit output, determined by the flows
    matrix = code.field.Zeros((num_rows, num_columns))
    for gg, flow in enumerate(flow_generators):
        pauli_string = flow.output_copy()
        if pauli_string:
            xs, zs = pauli_string.to_numpy()
            matrix[gg, :num_qubits] = xs.astype(np.uint8)
            matrix[gg, num_qubits : 2 * num_qubits] = zs.astype(np.uint8)
            matrix[gg, 2 * num_qubits] = 0 if pauli_string.sign == 1 else 1
        for measurement in flow.measurements_copy():
            matrix[gg, -num_observables - num_measurements + measurement] = 1
        for observable in flow.included_observables_copy():  # pragma: no cover technical edge case
            matrix[gg, -num_observables + observable] = 1

    # add stabilizers supported on measurements, as identified by detectors in the provided circuit
    detector_counter = 0
    measurement_counter = 0
    for instruction in state_prep_circuit.flattened():
        measurement_counter += instruction.num_measurements
        if instruction.name == "DETECTOR":
            row = len(flow_generators) + detector_counter
            for target in instruction.targets_copy():
                col = -num_observables - num_measurements + measurement_counter + target.value
                matrix[row, col] = 1
            detector_counter += 1

    # extract stabilizers that are supported entirely on the data qubits of the code
    stabilizers = []
    for row in matrix.row_reduce():
        xs = row[: len(code)]
        zs = row[num_qubits : num_qubits + len(code)]
        other_xs = row[len(code) : num_qubits]
        other_zs = row[num_qubits + len(code) : 2 * num_qubits]
        meas_obs = row[2 * num_qubits + 1 :]
        any_on_others = np.any(other_xs) or np.any(other_zs) or np.any(meas_obs)
        if (np.any(xs) or np.any(zs)) and not any_on_others:
            sign = -1 if row[2 * num_qubits] else 1
            string = stim.PauliString.from_numpy(xs=xs != 0, zs=zs != 0, sign=sign)
            stabilizers.append(string)

    if decoded:
        encoder = get_encoding_tableau(code)
        stabilizers = [stab.before(encoder, targets=range(len(code))) for stab in stabilizers]
    return stabilizers


def get_unaddressed_measurements(circuit: stim.Circuit) -> list[int]:
    """Identify measurements, by index, that are not addressed by any detectors in the circuit."""
    measurements: list[int] = []
    addressed_measurements = set()
    for instruction in circuit.flattened():
        new_measurements = range(
            len(measurements),
            len(measurements) + instruction.num_measurements,
        )
        measurements.extend(new_measurements)
        if instruction.name == "DETECTOR":
            addressed_measurements |= {
                measurements[target.value] for target in instruction.targets_copy()
            }
    return sorted(set(measurements) - addressed_measurements)


def _get_postselection_mask(
    post_select: bool | Collection[int], detector_record: DetectorRecord
) -> npt.NDArray[np.uint8] | None:
    """Build a post-selection mask for sinter."""
    if not post_select:
        return None

    num_prep = len(detector_record.get_events("prep"))
    num_flags = len(detector_record.get_events("flags"))
    num_detectors = num_prep + num_flags
    if isinstance(post_select, bool):
        post_select = range(num_flags) if post_select else ()
    if not all(-num_detectors <= dd < num_detectors for dd in post_select):
        raise ValueError(
            f"The provided circuit contains {num_detectors} detectors, so we can only post-select"
            f" on detectors with an index the range [-{num_detectors}, {num_detectors});"
            f" requested: {post_select}"
        )

    postselection_array = np.zeros(detector_record.num_events, dtype=int)
    postselection_array[list(post_select)] = 1
    return np.packbits(postselection_array, bitorder="little")


def _assert_valid_code_state(
    code: codes.QuditCode,
    decoded_stabilizers_or_state_prep_circuit: Collection[stim.PauliString] | stim.Circuit,
) -> None:
    """Assert that the provided stabilizers specify a unique logical state of the provided code."""
    error_message = (
        "The provided circuit does not deterministically prepare a logical code state that is"
        " unentangled from ancillas"
    )

    decoded_stabilizers = (
        get_state_stabilizers(code, decoded_stabilizers_or_state_prep_circuit, decoded=True)
        if isinstance(decoded_stabilizers_or_state_prep_circuit, stim.Circuit)
        else decoded_stabilizers_or_state_prep_circuit
    )
    if len(decoded_stabilizers) != len(code):
        raise ValueError(error_message)

    # collect decoded stabilizers into a binary matrix, including the sign bit, and row-reduce
    matrix = code.field.Zeros((len(decoded_stabilizers), 2 * len(code) + 1))
    for row, string in enumerate(decoded_stabilizers):
        xx, zz = string.to_numpy()
        matrix[row, : len(code)] = xx.astype(matrix.dtype)
        matrix[row, len(code) : -1] = zz.astype(matrix.dtype)
        matrix[row, -1] = 0 if string.sign.real == 1 else 1
    matrix_rref = matrix.row_reduce()

    # After row-reduction the matrix should look like
    #     [ LX  0  0 LZ  0  0  LS]
    #     [ 0  GX  0  0 GZ  0  GS]
    #     [ 0   0  0  0  0  I   0]
    # where
    # - (LX, LZ, LS) correspond to signed logical operators,
    # - (GX, GZ, GS) correspond to signed gauge operators,
    # - I is an identity matrix for stabilizers.
    # To verify this form, we zero out sectors appropriately and test equality with the zero matrix.
    rows_l = slice(code.dimension)
    rows_g = slice(rows_l.stop, rows_l.stop + code.gauge_dimension)
    rows_s = slice(rows_g.stop, None)
    cols_lx = slice(code.dimension)
    cols_lz = slice(len(code), len(code) + code.dimension)
    cols_gx = slice(cols_lx.stop, cols_lx.stop + code.gauge_dimension)
    cols_gz = slice(cols_lz.stop, cols_lz.stop + code.gauge_dimension)
    cols_s = slice(cols_gz.stop, -1)
    matrix_rref[rows_l, cols_lx] = 0
    matrix_rref[rows_l, cols_lz] = 0
    matrix_rref[rows_l, -1] = 0
    matrix_rref[rows_g, cols_gx] = 0
    matrix_rref[rows_g, cols_gz] = 0
    matrix_rref[rows_g, -1] = 0
    matrix_rref[rows_s, cols_s] -= code.field.Identity(code.num_checks)
    if np.any(matrix_rref):
        raise ValueError(
            "The provided circuit does not deterministically prepare a logical code state that is"
            " unentangled from ancillas"
        )
