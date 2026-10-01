# SPDX-License-Identifier: Apache-2.0

"""Circuit construction utilities for quantum error-corrected memory experiments."""

from collections.abc import Hashable, Sequence
from typing import NamedTuple

import galois
import numpy as np
import numpy.typing as npt
import stim

from qldpc import codes, math
from qldpc._util import format_docstring
from qldpc.objects import Node, Pauli, PauliXZ, PauliXZLike

from ..bookkeeping import DetectorRecord, MeasurementRecord, QubitIDs
from ..common import get_pauli_product_measurements, restrict_to_qubits, with_remapped_qubits
from ..encoding import get_encoding_circuit
from ..noise_model import (
    DEFAULT_IMMUNE_OP_TAG,
    DEFAULT_IMMUNE_QUBIT_TAG,
    GATE_OP_TYPES,
    NoiseModel,
    as_noiseless_circuit,
    op_type,
)
from .syndrome_measurement import (
    EdgeColoring,
    SyndromeMeasurementStrategy,
    validate_gauge_layers,
)

# default strategy used to schedule the two-qubit gates of a syndrome measurement circuit
DEFAULT_STRATEGY = EdgeColoring()


class MemoryExperimentParts(NamedTuple):
    initialization: stim.Circuit
    qec_cycle: stim.Circuit
    readout: stim.Circuit
    measurement_record: MeasurementRecord
    detector_record: DetectorRecord
    qubit_ids: QubitIDs


@format_docstring(DEFAULT_IMMUNE_OP_TAG=DEFAULT_IMMUNE_OP_TAG)
def get_memory_experiment(
    code: codes.QuditCode | codes.ClassicalCode,
    basis: PauliXZLike | None = Pauli.X,
    num_rounds: int = 1,
    *,
    noise_model: NoiseModel | None = None,
    qubit_ids: QubitIDs | None = None,
    syndrome_measurement_strategy: SyndromeMeasurementStrategy = DEFAULT_STRATEGY,
) -> stim.Circuit:
    """Construct a circuit for testing the performance of a code as a quantum memory.

    In a nutshell, the circuit constructed by this method performs (generally multiple) rounds
    quantum error correction (QEC) for the given code.  Each round of QEC measures all parity checks
    of the code, and detectors are added to enforce that
    (a) the syndrome from the first round of QEC is trivial, and
    (b) every subsequent round of QEC yields the same syndrome as the preceding round.
    We refer to ``num_rounds`` rounds of syndrome measurement as one logical QEC cycle.

    If ``basis`` is ``Pauli.X`` or ``Pauli.Z``, then the memory experiment only tracks errors in the
    logical Pauli operators of that type.  If ``basis is None``, the circuit entangles the code with
    a noiseless ancilla to track errors in all logical Pauli operators.

    More specifically, if ``basis`` is ``Pauli.X`` or ``Pauli.Z`` then the memory experiment
    performs the following:

    1. Initialize all data qubits to a +1 eigenstate of the specified basis: ``|0>`` for Z,
       ``|+>`` for X.
    2. Perform an initial round of QEC, adding detectors for the basis-type stabilizers.
    3. Perform ``num_rounds - 1`` additional QEC rounds, adding detectors to enforce that basis-type
       stabilizers have not changed between adjacent rounds of QEC.
    4. Measure all data qubits in the specified basis.
    5. Add detectors for all stabilizers that can be inferred from the data qubit measurements.

    If a ``noise_model`` is provided, then noise is added to the assembled fixed-basis circuit so
    that moments are accounted for across the initialization, QEC, and readout seams.

    If ``basis is None``, then the memory experiment noiselessly initializes each logical qubit of
    the code in a maximally entangled state with an (unphysical) noiseless ancilla qubit before
    running a noisy logical QEC cycle.  This initialization makes it possible to meaningfully track
    errors in both X-type and Z-type logical operators of a code.  The probability of an error in
    any logical operator is then essentially the process infidelity (or entanglement infidelity) of
    the logical QEC cycle.

    More specifically, if ``basis is None`` then the memory experiment performs the following:

    1. Prepare a logical all-``|0>`` state of the code.
    2. For each logical qubit of the code, prepare an ancilla qubit in ``|+>``, and apply an
       ancilla-controlled-logical-NOT gate to the logical qubit, thereby preparing Bell states
       ``|00> + |11>`` of logical qubits with their respective ancillas.
    3. Perform a logical QEC cycle as before, but now adding detectors for all stabilizers.
    4. Measure all stabilizers (with MPP gates).

    Unlike the fixed-basis experiment, the combined basis experiment only makes sense when starting
    from the Bell state.  It is also no longer possible to measure out all data qubits to infer all
    stabilizers.  Initialization and readout (measuring final stabilizers) are therefore noiseless.
    If a ``noise_model`` is provided, then noise is added to the logical QEC cycle alone.
    Otherwise, the initialization and readout sub-circuits are wrapped in a single-repetition
    stim.CircuitRepeatBlock tagged with "{DEFAULT_IMMUNE_OP_TAG}" to indicate that these
    sub-circuits should be immune to noise.  Tagged coordinate annotations likewise identify the
    reference qubits as immune when noise is added later.

    Remembering that observables in Stim are formally detectors, or circuit-level parity checks that
    must evaluate to 0 in the absence of errors, the preparation of Bell pairs allows us to annotate
    XX and ZZ observables for each Bell pair.  Here one of the "X"s in XX is a logical X for a
    logical qubit of the code, and the other "X" is a physical X on an associated ancilla qubit;
    likewise with ZZ.  Since the ancilla qubit is noiseless, we can attribute an error in XX or ZZ
    to a logical qubit error.

    Having said all of that, we do not actually annotate memory simulation circuits with the XX and
    ZZ observables described above.  Instead, we recognize that Bell-pair XX and ZZ operators are
    exact stabilizers of the circuit immediately after noiseless initialization, which allows us to
    freely multiply the XX and ZZ operators at the end of the circuit by XX and ZZ operators before
    the logical QEC cycle, thereby obtaining two-time XXXX and ZZZZ observables.  The chief (albeit
    perhaps aesthetic) benefit to this trick is that the support of these observables on the
    (noiseless) ancilla qubits cancels out, leaving us with two-time logical XX and ZZ observables
    supported on the data qubits alone.

    For a subsystem code, the parity checks of the code are gauge operators that need not commute,
    so each round of QEC measures one layer of mutually commuting checks at a time, in the order
    set by ``syndrome_measurement_strategy.get_gauge_layers(code)`` (by default, all X-type checks
    and then all Z-type checks of a CSS code).  Individual gauge measurement outcomes may then be
    random, but any product of checks in a single layer that is a stabilizer of the code has a
    deterministic value.  Detectors therefore track such products of checks, rather than individual
    checks, and the tracked products are chosen to generate all (basis-type) stabilizers of the
    code.  A ValueError is raised if the gauge measurement schedule does not make that possible.

    Qubits and detectors are assigned coordinates as follows:

    - The data qubit addressed by column C of the parity check matrix gets coordinate (0, C).
    - The check qubit associated with row R of the parity check matrix gets coordinate (1, R).
    - The ancilla qubit associated with logical qubit L of the code gets coordinate (2, L).
    - The K-th detector in measurement round M gets coordinate (M, 0, K).

    Args:
        code: An error-correcting code on qubits, which may be a subsystem code.  If passed a
            classical code, treat it as a quantum CSS code that protects only basis-type
            logical operators (or X-type logicals, if basis is None).
        basis: Pauli.X, Pauli.Z, or None to indicate which type of logical operators to track (where
            "None" means "both X and Z").  The strings "X" and "Z" (case-insensitive) are also
            accepted.  Default: Pauli.X.
        num_rounds: The number of syndrome measurement rounds to perform in one logical QEC cycle.
            Must be at least 1.  Default: 1.
        noise_model: The noise model to apply to the circuit after construction, or None to return a
            noiseless circuit.  Default: None.
        qubit_ids: A QubitIDs object specifying the index of data and check qubits.  Defaults to
            labeling data and check qubits according to their corresponding column/row of the parity
            check matrix, with data qubits numbered from 0 and check qubits numbered from len(code).
            For a combined-basis experiment, ``qubit_ids.reference`` contains noiseless Bell
            reference qubits.
        syndrome_measurement_strategy: The syndrome measurement strategy that defines how each
            round of QEC measures the parity checks of the code.  Default: circuits.EdgeColoring().

    Returns:
        stim.Circuit: A circuit ready for simulation via Stim or Sinter.

    Example::

        from qldpc import circuits, codes
        from qldpc.objects import Pauli

        # Create a 3-qubit repetition code
        rep_code = codes.RepetitionCode(3)

        # Generate 5-round Z-basis memory experiment with depolarizing noise
        noise_model = circuits.DepolarizingNoiseModel(1e-2)
        circuit = circuits.get_memory_experiment(
            rep_code,
            basis=Pauli.Z,
            num_rounds=5,
            noise_model=noise_model,
        )

        # The circuit is ready for simulation!
        # We can now sample detector and observable flips.
        sampler = circuit.compile_detector_sampler()
        detectors, observables = sampler.sample(shots=1000, separate_observables=True)
    """
    basis = None if basis is None else Pauli.coerce_xz(basis)
    initialization, qec_cycle, readout, _, _, qubit_ids = get_memory_experiment_parts(
        code,
        basis=basis,
        num_rounds=num_rounds,
        qubit_ids=qubit_ids,
        syndrome_measurement_strategy=syndrome_measurement_strategy,
    )

    # Add noise to the assembled fixed-basis experiment so the boundaries between initialization,
    # QEC, and readout do not introduce artificial idle moments.
    if basis is not None and noise_model is not None:
        return noise_model.noisy_circuit(initialization + qec_cycle + readout)

    # if tracking all logical operators, only the logical QEC cycle is noisy
    if basis is None:
        if noise_model is not None:
            qec_cycle = noise_model.noisy_circuit(
                qec_cycle,
                system_qubits=qubit_ids.data + qubit_ids.check + qubit_ids.ancilla,
            )
        else:
            # noise will be added later, so make initialization and readout noiseless
            initialization = as_noiseless_circuit(initialization)
            readout = as_noiseless_circuit(readout)

    return initialization + qec_cycle + readout


@restrict_to_qubits
def get_memory_experiment_parts(
    code: codes.QuditCode | codes.ClassicalCode,
    basis: PauliXZLike | None,
    num_rounds: int = 1,
    *,
    qubit_ids: QubitIDs | None = None,
    syndrome_measurement_strategy: SyndromeMeasurementStrategy = DEFAULT_STRATEGY,
) -> MemoryExperimentParts:
    """Noiseless components of a memory experiment.

    See help(qldpc.circuits.get_memory_experiment) for additional information.

    For a stabilizer code, the detector record keys detectors by the check qubit whose stabilizer
    they track.  For a subsystem code, the detector record keys detectors by the tuple of check
    qubits whose (gauge) measurement outcomes multiply to the stabilizer that they track.

    Returns:
        initialization: A circuit that sets the coordinates and initializes data qubit state.
        qec_cycle: A circuit for one logical QEC cycle, with num_rounds syndrome measurements.
        readout: A circuit that reads out final stabilizers.
        measurement_record: A record of all measurements in the above circuits.
        detector_record: A record of all detectors in the above circuits.
        qubit_ids: A QubitIDs object specifying the index of data and check qubits.
    """
    basis = None if basis is None else Pauli.coerce_xz(basis)
    if isinstance(code, codes.ClassicalCode):
        # wrap classical inputs as one-sided CSS codes for the shared circuit path
        matrix_z = code.matrix if basis is Pauli.Z else code.field.Zeros((0, len(code)))
        matrix_x = code.field.Zeros((0, len(code))) if basis is Pauli.Z else code.matrix
        code = codes.CSSCode(matrix_x, matrix_z)

    if num_rounds < 1:
        raise ValueError("num_rounds must be at least 1")

    if basis is None:
        return _get_combined_memory_simulation_parts(
            code,
            num_rounds=num_rounds,
            qubit_ids=qubit_ids,
            syndrome_measurement_strategy=syndrome_measurement_strategy,
        )
    return _get_basis_memory_experiment_parts(
        code,
        basis=basis,
        num_rounds=num_rounds,
        qubit_ids=qubit_ids,
        syndrome_measurement_strategy=syndrome_measurement_strategy,
    )


def _get_basis_memory_experiment_parts(
    code: codes.QuditCode,
    basis: PauliXZ,
    num_rounds: int = 1,
    *,
    qubit_ids: QubitIDs | None = None,
    syndrome_measurement_strategy: SyndromeMeasurementStrategy = DEFAULT_STRATEGY,
) -> MemoryExperimentParts:
    """Components of a memory experiment that tracks logical operators of a fixed type (basis).

    See help(qldpc.circuits.get_memory_experiment) for additional information.
    """
    if not isinstance(code, codes.CSSCode):
        raise TypeError("Memory experiments in a fixed basis only support CSS codes")

    # identify all qubits by index
    qubit_ids = QubitIDs.validated(qubit_ids, code) if qubit_ids else QubitIDs.from_code(code)
    data_ids, check_ids, _ = qubit_ids

    # set qubit coordinates
    coordinates = get_qubit_coordinates(data_ids, check_ids)

    # reset data qubits to the appropriate basis
    data_reset = stim.Circuit()
    data_reset.append(f"R{basis}", data_ids)

    # build a logical QEC cycle
    layers = _get_gauge_layers(code, syndrome_measurement_strategy)
    detectors = _get_stabilizer_detectors(code, qubit_ids, layers, basis)
    qec_cycle, measurement_record, detector_record = _get_qec_cycle(
        code, num_rounds, qubit_ids, detectors, syndrome_measurement_strategy, layers
    )

    # measure out the data qubits
    readout = stim.Circuit()
    readout.append(f"M{basis}", data_ids)
    measurement_record.append({data_id: mm for mm, data_id in enumerate(data_ids)})
    measurement_count = qec_cycle.num_measurements + readout.num_measurements
    measurement_record.validate_num_measurements(measurement_count)

    # detectors for stabilizers that can be inferred from data qubit measurements
    readout.append("SHIFT_COORDS", [], (1, 0, 0))
    for kk, detector in enumerate(detectors):
        basis_support = detector.stabilizer.reshape(2, len(code))[0 if basis is Pauli.X else 1]
        readout.append(
            "DETECTOR",
            [
                measurement_record.get_target_rec(data_ids[qq])
                for qq in np.flatnonzero(basis_support)
            ]
            + [measurement_record.get_target_rec(check_id) for check_id in detector.check_ids],
            (0, 0, kk),
        )
    detector_record.append({detector.key: dd for dd, detector in enumerate(detectors)})

    # annotate all basis-type observables
    targets = [measurement_record.get_target_rec(data_id) for data_id in data_ids]
    observables = get_observables(code, data_ids, basis=basis, on_measurements=targets)

    return MemoryExperimentParts(
        coordinates + data_reset,
        qec_cycle,
        readout + observables,
        measurement_record,
        detector_record,
        qubit_ids,
    )


def _get_combined_memory_simulation_parts(
    code: codes.QuditCode,
    num_rounds: int = 1,
    *,
    qubit_ids: QubitIDs | None = None,
    syndrome_measurement_strategy: SyndromeMeasurementStrategy = DEFAULT_STRATEGY,
) -> MemoryExperimentParts:
    """Components of a memory experiment that tracks all logical operators.

    See help(qldpc.circuits.get_memory_experiment) for additional information.
    """
    # identify all qubits by index
    qubit_ids = QubitIDs.validated(qubit_ids, code) if qubit_ids else QubitIDs.from_code(code)
    if qubit_ids.reference and len(qubit_ids.reference) != code.dimension:
        raise ValueError(
            "Combined-basis memory experiments require either no reference qubits or exactly one "
            "reference per logical qubit"
        )
    if not qubit_ids.reference:
        qubit_ids.add_references(code.dimension)
    data_ids, check_ids, _ = qubit_ids
    reference_ids = qubit_ids.reference

    # set qubit coordinates
    coordinates = get_qubit_coordinates(data_ids, check_ids)
    for kk, qubit in enumerate(reference_ids):
        coordinates.append(
            "QUBIT_COORDS",
            qubit,
            (2, kk),
            tag=DEFAULT_IMMUNE_QUBIT_TAG,
        )

    # noiselessly prepare all logical qubits in Bell states with ancillas
    state_prep = get_logical_bell_prep(code, data_ids, reference_ids)

    # build a logical QEC cycle
    layers = _get_gauge_layers(code, syndrome_measurement_strategy)
    detectors = _get_stabilizer_detectors(code, qubit_ids, layers)
    qec_cycle, measurement_record, detector_record = _get_qec_cycle(
        code, num_rounds, qubit_ids, detectors, syndrome_measurement_strategy, layers
    )
    # reject strategies that would use noiseless Bell reference qubits as work qubits
    operated_qubits = {
        target.qubit_value
        for instruction in qec_cycle.flattened()
        if op_type(instruction.name) in GATE_OP_TYPES
        for target in instruction.targets_copy()
        if target.qubit_value is not None
    }
    if reused_bell_ancillas := sorted(set(reference_ids) & operated_qubits):
        raise ValueError(
            "Syndrome measurement strategies cannot operate on Bell-reference ancillas"
            f" {reused_bell_ancillas}"
        )

    # measure all stabilizers
    readout = get_pauli_product_measurements(
        [detector.stabilizer for detector in detectors], data_ids
    )

    # identify the final syndrome measurements that detectors compare against stabilizer readout
    last_syndrome_measurements = [
        [measurement_record.get_events(check_id)[-1] for check_id in detector.check_ids]
        for detector in detectors
    ]

    # update the measurement record, add detectors, and update the detector record
    readout.append("SHIFT_COORDS", [], (1, 0, 0))
    measurement_record.append({detector.key: mm for mm, detector in enumerate(detectors)})
    measurement_count = qec_cycle.num_measurements + readout.num_measurements
    measurement_record.validate_num_measurements(measurement_count)
    for kk, (detector, measurements) in enumerate(zip(detectors, last_syndrome_measurements)):
        targets = [measurement_record.get_target_rec(detector.key)] + [
            stim.target_rec(measurement - measurement_record.num_events)
            for measurement in measurements
        ]
        readout.append("DETECTOR", targets, (0, 0, kk))
    detector_record.append({detector.key: dd for dd, detector in enumerate(detectors)})

    # annotate all observables
    observables = get_observables(code, data_ids)

    return MemoryExperimentParts(
        coordinates + state_prep + observables,
        qec_cycle,
        readout + observables,
        measurement_record,
        detector_record,
        qubit_ids,
    )


def get_qubit_coordinates(
    data_qubits: Sequence[int] = (),
    check_qubits: Sequence[int] = (),
    ancilla_qubits: Sequence[int] = (),
) -> stim.Circuit:
    """Circuit declaring coordinates for the given qubits.

    Args:
        data_qubits: The indices of data qubits.  data_qubits[kk] gets index (0, kk).
        check_qubits: The indices of check qubits.  check_qubits[kk] gets index (1, kk).
        ancilla_qubits: The indices of ancilla qubits.  ancilla_qubits[kk] gets index (2, kk).

    Returns:
        A circuit declaring qubit coordinates.
    """
    circuit = stim.Circuit()
    for kk, qubit in enumerate(data_qubits):
        circuit.append("QUBIT_COORDS", qubit, (0, kk))
    for kk, qubit in enumerate(check_qubits):
        circuit.append("QUBIT_COORDS", qubit, (1, kk))
    for kk, qubit in enumerate(ancilla_qubits):
        circuit.append("QUBIT_COORDS", qubit, (2, kk))
    return circuit


@restrict_to_qubits
def get_observables(
    code: codes.QuditCode,
    data_qubits: Sequence[int] | None = None,
    *,
    basis: PauliXZLike | None = None,
    on_measurements: Sequence[stim.GateTarget] | bool = False,
    observable_indices: Sequence[int] | None = None,
) -> stim.Circuit:
    """Construct a circuit of logical observable annotations.

    Args:
        code: The code whose observables we wish to annotate.
        data_qubits: Indices of the data qubits of the code.  Default: the first len(code) integers.
        basis: The type of observable (Pauli.X or Pauli.Z, or equivalently a case-insensitive "X"
            or "Z" string) we wish to annotate, or None for both.
        on_measurements: If provided a sequence of measurement targets, assume that they correspond
            to measurements of the data qubits in a specified basis (which in this case is not
            allowed to be None), and define observables using these measurements.  If True, define
            observables identically on the last len(code) measurements.  Otherwise (if False),
            define observable using Pauli targets.  Default: False.
        observable_indices: Indices to use for the observables.  Default: range(num_observables).

    Returns:
        A Stim circuit of OBSERVABLE_INCLUDE instructions.
    """
    basis = None if basis is None else Pauli.coerce_xz(basis)
    data_qubits = range(len(code)) if data_qubits is None else data_qubits
    if len(data_qubits) != len(code):
        raise ValueError("data_qubits must contain one target per data qubit")
    num_observables = code.dimension * (2 if basis is None else 1)
    observable_indices = (
        range(num_observables) if observable_indices is None else observable_indices
    )

    if on_measurements is True:
        on_measurements = [stim.target_rec(mm) for mm in range(-len(code), 0)]

    # consistency checks
    if on_measurements is not False and len(on_measurements) != len(code):
        raise ValueError("on_measurements must contain one target per data qubit")
    if len(observable_indices) != num_observables:
        raise ValueError(f"Expected {num_observables} observable indices")

    # build a graph of edges directed from observables to the data qubits they address
    logical_ops = code.get_logical_ops(basis, symplectic=True)
    logical_op_graph = codes.QuditCode.matrix_to_graph(logical_ops)

    if on_measurements and (not isinstance(code, codes.CSSCode) or basis is None):
        raise ValueError(
            "Defining observables on measurements is only allowed (a) for CSS codes (b) with a"
            " fixed measurement basis (Pauli.X or Pauli.Z)"
        )

    circuit = stim.Circuit()
    for node_index, observable_index in enumerate(observable_indices):
        observable_node = Node(node_index, is_data=False)
        targets = [
            on_measurements[data_node.index]
            if on_measurements
            else stim.target_pauli(data_qubits[data_node.index], str(edge_data[Pauli]))
            for _, data_node, edge_data in logical_op_graph.edges(observable_node, data=True)
        ]
        circuit.append("OBSERVABLE_INCLUDE", targets, [observable_index])

    return circuit


@restrict_to_qubits
def get_logical_bell_prep(
    code: codes.QuditCode,
    data_qubits: Sequence[int] | None = None,
    reference_qubits: Sequence[int] | None = None,
) -> stim.Circuit:
    """Noiselessly prepare the logical qubits of the given code in Bell states with references.

    Args:
        code: The code for which we are constructing a logical Bell-state preparation circuit.
        data_qubits: Indices of the code's data qubits.  Default: the first len(code) integers.
        reference_qubits: Indices of the reference qubits to entangle with the code's logical
            qubits.  Default: the first code.dimension integers after the data qubit indices.

    Returns:
        A circuit that noiselessly initializes all logical qubits into Bell pairs with references.
    """
    data_qubits = range(len(code)) if data_qubits is None else data_qubits
    reference_qubits = (
        range(max(data_qubits, default=-1) + 1, max(data_qubits, default=-1) + 1 + code.dimension)
        if reference_qubits is None
        else reference_qubits
    )
    if len(data_qubits) != len(code):
        raise ValueError("data_qubits must contain one target per data qubit")
    if len(reference_qubits) != code.dimension:
        raise ValueError("reference_qubits must contain one target per logical qubit")

    # entangle the first code.dimension data qubits with references
    circuit = stim.Circuit()
    circuit.append("H", data_qubits[: code.dimension])
    circuit.append("CX", [qq for pair in zip(data_qubits, reference_qubits) for qq in pair])

    # encode the first code.dimension data qubits
    circuit.append(with_remapped_qubits(get_encoding_circuit(code), data_qubits))

    return as_noiseless_circuit(circuit)


class _StabilizerDetector(NamedTuple):
    """A stabilizer whose value is the product of the most recent outcomes of some checks."""

    key: Hashable  # the key for this detector in a DetectorRecord
    check_ids: tuple[int, ...]  # the check qubits whose measurement outcomes we multiply
    stabilizer: galois.FieldArray  # the stabilizer, as a symplectic vector


def _get_gauge_layers(
    code: codes.QuditCode, syndrome_measurement_strategy: SyndromeMeasurementStrategy
) -> tuple[tuple[int, ...], ...] | None:
    """Validated gauge measurement layers for a subsystem code, or None for a stabilizer code.

    The layers are computed once so that the circuit and its detectors use the same schedule.
    """
    if not code.is_subsystem_code:
        return None
    layers = tuple(
        tuple(sorted(layer)) for layer in syndrome_measurement_strategy.get_gauge_layers(code)
    )
    validate_gauge_layers(code, layers)
    return layers


def _get_stabilizer_detectors(
    code: codes.QuditCode,
    qubit_ids: QubitIDs,
    layers: Sequence[Sequence[int]] | None,
    basis: PauliXZ | None = None,
) -> list[_StabilizerDetector]:
    """Identify the stabilizers to annotate with detectors in a memory experiment.

    For a stabilizer code, every check is a stabilizer, and the detector for a check is keyed by its
    check qubit.  If a basis is provided, only annotate checks of that type.

    For a subsystem code, a detector tracks a product of checks from one of the given layers of the
    gauge measurement schedule, such that this product is a stabilizer.  The detector is keyed by
    the tuple of check qubits in this product.  If a basis is provided, only annotate stabilizers of
    that type.  This method prefers the stabilizer generators in code.get_stabilizer_ops() (which
    may, for example, have low weight), and adds other products of checks as necessary to generate
    all tracked stabilizers.

    Raises:
        ValueError: If the gauge measurement schedule of a subsystem code does not determine all
            stabilizers that the memory experiment should track.
    """
    if layers is None:
        detector_check_ids = (
            qubit_ids.check
            if basis is None
            else qubit_ids.checks_x
            if basis is Pauli.X
            else qubit_ids.checks_z
        )
        check_to_row = {check_id: row for row, check_id in enumerate(qubit_ids.check)}
        return [
            _StabilizerDetector(check_id, (check_id,), code.matrix[check_to_row[check_id]])
            for check_id in detector_check_ids
        ]

    num_qubits = len(code)
    gauge_ops = code.matrix.view(code.field)

    def is_tracked(ops: galois.FieldArray) -> npt.NDArray[np.bool_]:
        """Identify the rows of a symplectic matrix that have the tracked Pauli type."""
        if basis is None:
            return np.ones(len(ops), dtype=bool)
        other_support = ops[:, num_qubits:] if basis is Pauli.X else ops[:, :num_qubits]
        return ~np.any(other_support, axis=1)

    stabilizer_ops = code.get_stabilizer_ops(symplectic=True)
    stabilizer_ops = stabilizer_ops[is_tracked(stabilizer_ops)]

    # collect (layer, coefficients, stabilizer) triplets, where the coefficients of checks in a
    # layer multiply to a stabilizer
    candidates: list[tuple[Sequence[int], galois.FieldArray, galois.FieldArray]] = []
    for layer in layers:
        # row-reduce [checks | identity] to express stabilizers in the span of these checks
        layer_ops = gauge_ops[list(layer)]
        identity = code.field.Identity(len(layer))
        reduced = np.hstack([layer_ops, identity]).view(code.field).row_reduce()
        reduced = reduced[np.any(reduced[:, : 2 * num_qubits], axis=1)]
        pivots = math.first_nonzero_cols(reduced[:, : 2 * num_qubits])
        coefficients = stabilizer_ops[:, pivots] @ reduced[:, 2 * num_qubits :]
        solved = ~np.any(coefficients @ layer_ops - stabilizer_ops, axis=1)
        candidates.extend(
            (layer, coefficients_row, stabilizer)
            for coefficients_row, stabilizer in zip(coefficients[solved], stabilizer_ops[solved])
        )
    for layer in layers:
        # every product of checks in this layer that commutes with all checks is a stabilizer
        layer_ops = gauge_ops[list(layer)]
        commutators = layer_ops @ math.symplectic_conjugate(gauge_ops).T
        coefficients = commutators.T.null_space()
        stabilizers = coefficients @ layer_ops
        tracked = is_tracked(stabilizers)
        candidates.extend(
            (layer, coefficients_row, stabilizer)
            for coefficients_row, stabilizer in zip(coefficients[tracked], stabilizers[tracked])
        )

    # select a maximal set of independent stabilizers, in order of preference
    detectors = []
    if candidates:
        candidate_stabilizers = code.field([stabilizer for *_, stabilizer in candidates])
        reduced = candidate_stabilizers.T.row_reduce()
        independent = math.first_nonzero_cols(reduced[np.any(reduced, axis=1)])
        for index in independent:
            layer, coefficients_row, stabilizer = candidates[index]
            check_ids = tuple(qubit_ids.check[layer[cc]] for cc in np.flatnonzero(coefficients_row))
            detectors.append(_StabilizerDetector(check_ids, check_ids, stabilizer))

    if len(detectors) != np.linalg.matrix_rank(stabilizer_ops):
        raise ValueError(
            "The gauge measurement schedule does not determine all "
            + ("" if basis is None else f"{basis}-type ")
            + "stabilizers of the code: products of the checks within single gauge layers must"
            " generate these stabilizers (see SyndromeMeasurementStrategy.get_gauge_layers)"
        )
    return detectors


def _get_qec_cycle(
    code: codes.QuditCode,
    num_rounds: int,
    qubit_ids: QubitIDs,
    detectors: Sequence[_StabilizerDetector],
    syndrome_measurement_strategy: SyndromeMeasurementStrategy,
    layers: Sequence[Sequence[int]] | None = None,
) -> tuple[stim.Circuit, MeasurementRecord, DetectorRecord]:
    """Build a circuit of num_rounds noiseless syndrome measurements for a given code.

    Args:
        code: The code for which we are building a logical QEC cycle.
        num_rounds: The number of syndrome measurement rounds in one logical QEC cycle.
        qubit_ids: A QubitIDs object specifying the index of data and check qubits.
        detectors: The stabilizers to annotate with detectors, each of which is the product of the
            most recent measurement outcomes of some check qubits in qubit_ids.check.
        syndrome_measurement_strategy: The syndrome measurement strategy that defines how each
            round of QEC measures the parity checks of the code.
        layers: The gauge measurement layers of a subsystem code, or None for a stabilizer code.

    Returns:
        stim.Circuit: The noiseless circuit of num_rounds syndrome measurements.
        MeasurementRecord: The record of all measurements in the constructed circuit.
        DetectorRecord: The record of all detectors in the constructed circuit.
    """
    if layers is not None:
        one_round, round_measurement_record = syndrome_measurement_strategy.get_subsystem_circuit(
            code, qubit_ids, layers=layers
        )
    else:
        one_round, round_measurement_record = syndrome_measurement_strategy.get_circuit(
            code, qubit_ids
        )
    round_detector_record = {detector.key: dd for dd, detector in enumerate(detectors)}

    circuit = stim.Circuit()
    measurement_record = MeasurementRecord()
    detector_record = DetectorRecord()

    # apply first round of QEC and detectors
    circuit.append(one_round)
    measurement_record.append(round_measurement_record)
    measurement_count = circuit.num_measurements
    measurement_record.validate_num_measurements(measurement_count)
    for kk, detector in enumerate(detectors):
        targets = [measurement_record.get_target_rec(check_id) for check_id in detector.check_ids]
        circuit.append("DETECTOR", targets, (0, 0, kk))
    detector_record.append(round_detector_record)

    # apply following repeated rounds of QEC and detectors
    if num_rounds > 1:
        repeat_circuit = one_round.copy()
        measurement_record.append(round_measurement_record)
        measurement_count += repeat_circuit.num_measurements
        measurement_record.validate_num_measurements(measurement_count)
        repeat_circuit.append("SHIFT_COORDS", [], (1, 0, 0))
        for kk, detector in enumerate(detectors):
            targets = [
                measurement_record.get_target_rec(check_id, index)
                for index in (-1, -2)
                for check_id in detector.check_ids
            ]
            repeat_circuit.append("DETECTOR", targets, (0, 0, kk))
        circuit.append(stim.CircuitRepeatBlock(num_rounds - 1, repeat_circuit))

        # update the measurement and detector records to account for repetitions
        measurement_record.append(round_measurement_record, repeat=num_rounds - 2)
        detector_record.append(round_detector_record, repeat=num_rounds - 1)

    return circuit, measurement_record, detector_record
