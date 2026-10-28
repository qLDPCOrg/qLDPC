# SPDX-License-Identifier: Apache-2.0

"""Circuit construction utilities for quantum error-corrected memory experiments."""

from __future__ import annotations

import dataclasses
from collections.abc import Hashable, Sequence
from typing import NamedTuple, Self

import galois
import numpy as np
import numpy.typing as npt
import stim

from qldpc import _deprecations, codes, math
from qldpc._utils import format_docstring, get_external_caller_stacklevel
from qldpc.objects import Node, Pauli, PauliXZ, PauliXZLike, TannerGraph

from ..bookkeeping import DetectorRecord, MeasurementRecord, QubitIDs
from ..common import _restrict_to_qubits, get_pauli_product_measurements, with_remapped_qubits
from ..encoding import get_encoding_circuit
from ..noise_model import (
    DEFAULT_IMMUNE_OP_TAG,
    DEFAULT_IMMUNE_QUBIT_TAG,
    NoiseModel,
    as_noiseless_circuit,
)
from ..noise_model._operations import GATE_OP_TYPES, op_type
from .syndrome_measurement import EdgeColoring, SyndromeMeasurementStrategy

# default strategy used to schedule the two-qubit gates of a syndrome measurement circuit
_DEFAULT_STRATEGY = EdgeColoring()


####################################################################################################
# memory experiments


class MemoryExperimentParts(NamedTuple):
    """Noiseless components of a memory experiment, returned by get_memory_experiment_parts."""

    initialization: stim.Circuit
    qec_cycle: stim.Circuit
    readout: stim.Circuit
    measurement_record: MeasurementRecord
    detector_record: DetectorRecord
    qubit_ids: QubitIDs


class CheckKey(int):
    """The key of the detectors in a memory experiment that track the outcome of one check.

    A CheckKey is the index of a check qubit (an element of ``QubitIDs.check``).  It compares,
    hashes, and sorts as that integer, so records can be indexed with plain integers, as in
    ``detector_record[check_id]``, although arithmetic with a CheckKey returns a plain integer.
    Detectors that track a product of several checks are labeled by a CheckProductKey instead, which
    has the same attributes.  See help(qldpc.circuits.get_memory_experiment_parts) for the detectors
    that a key labels.

    Example::

        from qldpc import circuits, codes

        code = codes.SurfaceCode(2)
        parts = circuits.get_memory_experiment_parts(code, "Z", num_rounds=3, readout_keys=True)
        check_id = parts.qubit_ids.checks_z[0]
        round_detectors = parts.detector_record[check_id]  # one for each of the 3 rounds
        readout_detector = parts.detector_record.get_detector(circuits.ReadoutKey(check_id))

    Attributes:
        check_ids: The index of the check qubit, as a tuple with one element, like
            CheckProductKey.check_ids.
        is_relation: Whether the check is the identity operator, which makes it a relation (see
            get_memory_experiment_parts).
    """

    _is_relation: bool

    def __new__(cls, check_id: int, is_relation: bool = False) -> Self:
        """Construct the key of the detectors that track the outcome of a check."""
        key = super().__new__(cls, check_id)
        object.__setattr__(key, "_is_relation", bool(is_relation))
        return key

    @property
    def check_ids(self) -> tuple[int, ...]:
        """The index of the check qubit, as a tuple with one element."""
        return (int(self),)

    @property
    def is_relation(self) -> bool:
        """Whether the check is the identity operator."""
        return self._is_relation

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable")

    def __reduce__(self) -> tuple[type[Self], tuple[int, bool]]:
        """Copy or pickle a CheckKey together with its attributes."""
        return type(self), (int(self), self.is_relation)


@dataclasses.dataclass(frozen=True)
class CheckProductKey:
    """The key of the detectors in a memory experiment that track a product of several checks.

    Memory experiments for subsystem codes track products of the (gauge) checks in one layer of the
    gauge measurement schedule.  See help(qldpc.circuits.get_memory_experiment_parts) for the
    detectors that a key labels.  Two keys are equal if they multiply the same checks.

    Attributes:
        check_ids: The sorted indices of the check qubits (elements of ``QubitIDs.check``) whose
            outcomes the detectors multiply.  There are at least two of them.
        is_relation: Whether the product of the checks is the identity operator, which makes it a
            relation (see get_memory_experiment_parts).
    """

    check_ids: tuple[int, ...]
    is_relation: bool = dataclasses.field(default=False, compare=False)

    def __post_init__(self) -> None:
        check_ids = tuple(sorted(int(check_id) for check_id in self.check_ids))
        if len(check_ids) < 2 or len(set(check_ids)) != len(check_ids):
            raise ValueError(
                f"A CheckProductKey needs at least two distinct check qubits (got {self.check_ids})"
            )
        object.__setattr__(self, "check_ids", check_ids)
        object.__setattr__(self, "is_relation", bool(self.is_relation))


@dataclasses.dataclass(frozen=True)
class ReadoutKey:
    """The key of a readout detector, or a final stabilizer measurement, in a memory experiment.

    At the end of a memory experiment, a readout detector compares the last outcome of a check, or
    product of checks, with its value inferred from the final readout.  With ``readout_keys=True``,
    get_memory_experiment_parts keys this readout detector by ``ReadoutKey(key)``, where ``key``
    labels the detectors of the same check or checks in each round.  If the memory experiment tracks
    both X-type and Z-type logical operators, the measurement record likewise keys the final
    measurement of the corresponding stabilizer.  See
    help(qldpc.circuits.get_memory_experiment_parts).

    Since a CheckKey is equal to the index of its check qubit, ``ReadoutKey(check_id)`` with a plain
    integer is equal to ``ReadoutKey(CheckKey(check_id))``, and finds the same record entries.

    Attributes:
        key: The key of the detectors that track the same check, or product of checks, in each
            round.
    """

    key: int | CheckProductKey


@format_docstring(DEFAULT_IMMUNE_OP_TAG=DEFAULT_IMMUNE_OP_TAG)
def get_memory_experiment(
    code: codes.QuditCode | codes.ClassicalCode,
    basis: PauliXZLike | None = Pauli.X,
    num_rounds: int = 1,
    *,
    noise_model: NoiseModel | None = None,
    qubit_ids: QubitIDs | None = None,
    syndrome_measurement_strategy: SyndromeMeasurementStrategy = _DEFAULT_STRATEGY,
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

    1. Initialize all data qubits to a +1 eigenstate of the specified basis: ``|0>`` for Z, ``|+>``
       for X.
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
    so they cannot all be measured at once.  Each round of QEC instead measures one layer of
    mutually commuting checks at a time, in the order set by
    ``syndrome_measurement_strategy.get_gauge_layers(code)`` (by default, all X-type checks and then
    all Z-type checks of a CSS code).  Operations of consecutive layers on disjoint qubits may
    nonetheless share TICK-separated moments (see
    SyndromeMeasurementStrategy.get_subsystem_circuit).

    Since checks in different layers need not commute, the outcomes of individual checks may be
    random, but any product of checks in a single layer that is a stabilizer of the code has a
    deterministic value.  Detectors therefore track such products of checks.  Every distinct
    (basis-type) stabilizer generator in ``code.get_stabilizer_ops()`` that is a product of checks
    in a single layer gets a detector, even if these generators are redundant, since redundant
    detectors can help decoders (as with redundant checks of a stabilizer code).  Other products are
    added as necessary to generate all (basis-type) stabilizers of the code, and each stabilizer is
    represented by a product of few checks, which are chosen greedily.  If the checks in a layer are
    linearly dependent, products of these checks that equal the identity operator have deterministic
    values in every round, so additional detectors check these products within each round, unless
    their values follow from other detectors.

    If stabilizers are products of many checks, a single fault can flip many detectors, in which
    case the detector error model of the circuit may not be decomposable into graphlike errors (as
    required, for example, by matching decoders).  A ValueError is raised if the gauge layers do not
    partition the checks of the code into sets of mutually commuting checks, or if products of the
    checks in single layers do not generate all (basis-type) stabilizers of the code.

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
        syndrome_measurement_strategy: The syndrome measurement strategy that defines how each round
            of QEC measures the parity checks of the code.  Default: circuits.EdgeColoring().

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
        readout_keys=True,  # the records are unused, so only avoid the FutureWarning
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


@_restrict_to_qubits
def get_memory_experiment_parts(
    code: codes.QuditCode | codes.ClassicalCode,
    basis: PauliXZLike | None,
    num_rounds: int = 1,
    *,
    qubit_ids: QubitIDs | None = None,
    syndrome_measurement_strategy: SyndromeMeasurementStrategy = _DEFAULT_STRATEGY,
    readout_keys: bool | None = None,  # FUTURE CHANGE: see qldpc._deprecations
) -> MemoryExperimentParts:
    """Noiseless components of a memory experiment.

    See help(qldpc.circuits.get_memory_experiment) for additional information.

    The detector record labels the detectors that track the outcome of one check by a CheckKey,
    which is equal to the index of the check qubit, and labels the detectors that track a product of
    several checks of a subsystem code by a CheckProductKey.  Each such key labels one detector for
    each round of syndrome measurement.  In the first round, this detector checks the outcome
    against its known value, and in each later round, it compares the outcome to that of the
    previous round.  Unless the key is a relation (see below), it also has a readout detector, which
    compares the outcome of the last round to its value inferred from the final measurement of the
    data qubits, or (if basis is None) from a final measurement of the stabilizer, which the
    measurement record keys in the same way as the readout detector.  How readout detectors and
    final stabilizer measurements are keyed is determined by readout_keys.

    A relation is a product of checks that equals the identity operator.  If the checks in one layer
    of a subsystem code's gauge measurement schedule are linearly dependent, some of their products
    are relations, which have the same fixed outcome in every round.  The detectors of a relation
    (``key.is_relation == True``) therefore check each round on its own, and a relation has no
    readout detector.  These detectors flag errors that flip the outcomes of an odd number of the
    checks in a relation during one round, such as a single measurement error.  The checks of a
    stabilizer code are never relations.

    Args:
        code: The code, as for get_memory_experiment.
        basis: The type of logical operators to track, as for get_memory_experiment.
        num_rounds: The number of syndrome measurement rounds, as for get_memory_experiment.
        qubit_ids: The indices of the qubits, as for get_memory_experiment.
        syndrome_measurement_strategy: The syndrome measurement strategy, as for
            get_memory_experiment.
        readout_keys: If True, key each readout detector, and (if basis is None) each final
            stabilizer measurement, by ``ReadoutKey(key)``, where ``key`` is the key of the
            detectors that track the same check or checks in each round.  Each key then maps to
            exactly one detector for each round.  If False, append them to the entries of ``key``
            instead, so that the readout detector of a key is ``detector_record[key][-1]``, as in
            earlier releases of qLDPC.  If None (the default), behave as for False and issue a
            FutureWarning, because the default will change to the behavior of True in a future
            release.

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

    if readout_keys is None:  # FUTURE CHANGE: see qldpc._deprecations
        _deprecations.warn_future_change(
            "get_memory_experiment_parts currently appends each readout detector (and, if basis is"
            " None, each final stabilizer measurement) to the record entries of the check(s) that"
            " it tracks.  In a future release, it will key them by circuits.ReadoutKey instead."
            "  Pass readout_keys=True to use the new layout now, or readout_keys=False to keep the"
            " current layout and silence this warning."
            "  See help(qldpc.circuits.get_memory_experiment_parts).",
            stacklevel=get_external_caller_stacklevel(),
        )
        readout_keys = False

    if basis is None:
        return _get_combined_memory_simulation_parts(
            code,
            num_rounds=num_rounds,
            qubit_ids=qubit_ids,
            syndrome_measurement_strategy=syndrome_measurement_strategy,
            readout_keys=readout_keys,
        )
    return _get_basis_memory_experiment_parts(
        code,
        basis=basis,
        num_rounds=num_rounds,
        qubit_ids=qubit_ids,
        syndrome_measurement_strategy=syndrome_measurement_strategy,
        readout_keys=readout_keys,
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


@_restrict_to_qubits
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
        basis: The type of observable (Pauli.X or Pauli.Z, or equivalently a case-insensitive "X" or
            "Z" string) we wish to annotate, or None for both.
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
    logical_op_graph = TannerGraph.from_matrix(logical_ops, symplectic=True)

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


@_restrict_to_qubits
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


def _get_basis_memory_experiment_parts(
    code: codes.QuditCode,
    basis: PauliXZ,
    num_rounds: int = 1,
    *,
    qubit_ids: QubitIDs | None = None,
    syndrome_measurement_strategy: SyndromeMeasurementStrategy = _DEFAULT_STRATEGY,
    readout_keys: bool,
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
    one_round, round_measurement_record, detectors = _get_syndrome_round(
        code, qubit_ids, syndrome_measurement_strategy, basis
    )
    qec_cycle, measurement_record, detector_record = _get_qec_cycle(
        one_round, round_measurement_record, detectors, num_rounds
    )

    # measure out the data qubits
    readout = stim.Circuit()
    readout.append(f"M{basis}", data_ids)
    measurement_record.append({data_id: mm for mm, data_id in enumerate(data_ids)})
    measurement_count = qec_cycle.num_measurements + readout.num_measurements
    measurement_record.validate_num_measurements(measurement_count)

    # detectors for stabilizers that can be inferred from data qubit measurements
    readout.append("SHIFT_COORDS", [], (1, 0, 0))
    readout_detectors = [detector for detector in detectors if not detector.is_relation]
    for kk, detector in enumerate(readout_detectors):
        basis_support = detector.stabilizer.reshape(2, len(code))[basis]
        readout.append(
            "DETECTOR",
            [
                measurement_record.get_target_rec(data_ids[qq])
                for qq in np.flatnonzero(basis_support)
            ]
            + [measurement_record.get_target_rec(check_id) for check_id in detector.check_ids],
            (0, 0, kk),
        )
    detector_record.append(
        {
            _get_readout_record_key(detector.key, readout_keys): dd
            for dd, detector in enumerate(readout_detectors)
        }
    )

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
    syndrome_measurement_strategy: SyndromeMeasurementStrategy = _DEFAULT_STRATEGY,
    readout_keys: bool,
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
    one_round, round_measurement_record, detectors = _get_syndrome_round(
        code, qubit_ids, syndrome_measurement_strategy
    )
    qec_cycle, measurement_record, detector_record = _get_qec_cycle(
        one_round, round_measurement_record, detectors, num_rounds
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
    readout_detectors = [detector for detector in detectors if not detector.is_relation]
    readout = get_pauli_product_measurements(
        [detector.stabilizer for detector in readout_detectors], data_ids
    )

    # identify the final syndrome measurements that detectors compare against stabilizer readout
    last_syndrome_targets = [
        [measurement_record.get_target_rec(check_id) for check_id in detector.check_ids]
        for detector in readout_detectors
    ]

    # update the measurement record, add detectors, and update the detector record
    readout.append("SHIFT_COORDS", [], (1, 0, 0))
    readout_record_keys = [
        _get_readout_record_key(detector.key, readout_keys) for detector in readout_detectors
    ]
    measurement_record.append({key: mm for mm, key in enumerate(readout_record_keys)})
    measurement_count = qec_cycle.num_measurements + readout.num_measurements
    measurement_record.validate_num_measurements(measurement_count)
    for kk, (readout_key, syndrome_targets) in enumerate(
        zip(readout_record_keys, last_syndrome_targets)
    ):
        # shift the syndrome targets to account for the stabilizer readout
        targets = [measurement_record.get_target_rec(readout_key)] + [
            stim.target_rec(target.value - len(readout_detectors)) for target in syndrome_targets
        ]
        readout.append("DETECTOR", targets, (0, 0, kk))
    detector_record.append({key: dd for dd, key in enumerate(readout_record_keys)})

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


####################################################################################################
# syndrome measurement rounds and their detectors


class _StabilizerDetector(NamedTuple):
    """A stabilizer whose value is the product of the most recent outcomes of some checks."""

    key: CheckKey | CheckProductKey  # the key for this detector in a DetectorRecord
    stabilizer: galois.FieldArray  # the stabilizer (or identity), as a symplectic vector

    @property
    def check_ids(self) -> tuple[int, ...]:
        """The check qubits whose measurement outcomes we multiply."""
        return self.key.check_ids

    @property
    def is_relation(self) -> bool:
        """Whether the checks multiply to the identity operator."""
        return self.key.is_relation


class _LayerProduct(NamedTuple):
    """A product of some of the checks in one layer of a gauge measurement schedule."""

    layer_index: int  # the index of the layer in the schedule
    check_vector: npt.NDArray[np.uint8]  # indicator vector of the multiplied checks in the layer
    stabilizer: galois.FieldArray  # the product, as a symplectic vector


def _get_syndrome_round(
    code: codes.QuditCode,
    qubit_ids: QubitIDs,
    syndrome_measurement_strategy: SyndromeMeasurementStrategy,
    basis: PauliXZ | None = None,
) -> tuple[stim.Circuit, MeasurementRecord, list[_StabilizerDetector]]:
    """Build one noiseless round of syndrome measurement, and identify the stabilizers to detect.

    For a subsystem code, the gauge layers are computed once, so that the circuit and its detectors
    share one measurement schedule.  If a basis is provided, only stabilizers of that type get
    detectors.

    Returns:
        stim.Circuit: A circuit for one round of syndrome measurement.
        MeasurementRecord: The record of measurements in that circuit.
        list[_StabilizerDetector]: The stabilizers to annotate with detectors in every round.
    """
    if not code.is_subsystem_code:
        circuit, record = syndrome_measurement_strategy.get_circuit(code, qubit_ids)
        return circuit, record, _get_check_detectors(code, qubit_ids, basis)
    layers = tuple(
        tuple(sorted(layer)) for layer in syndrome_measurement_strategy.get_gauge_layers(code)
    )
    circuit, record = syndrome_measurement_strategy.get_subsystem_circuit(
        code, qubit_ids, layers=layers
    )
    return circuit, record, _get_subsystem_detectors(code, qubit_ids, layers, basis)


def _get_qec_cycle(
    one_round: stim.Circuit,
    round_measurement_record: MeasurementRecord,
    detectors: Sequence[_StabilizerDetector],
    num_rounds: int,
) -> tuple[stim.Circuit, MeasurementRecord, DetectorRecord]:
    """Repeat one round of syndrome measurement num_rounds times, and annotate detectors.

    Args:
        one_round: A noiseless circuit for one round of syndrome measurement.
        round_measurement_record: The record of measurements in one_round.
        detectors: The stabilizers to annotate with detectors in every round, each of which is the
            product of the most recent measurement outcomes of some check qubits.
        num_rounds: The number of syndrome measurement rounds in one logical QEC cycle.

    Returns:
        stim.Circuit: The noiseless circuit of num_rounds syndrome measurements.
        MeasurementRecord: The record of all measurements in the constructed circuit.
        DetectorRecord: The record of all detectors in the constructed circuit.
    """
    round_detector_record: dict[Hashable, int] = {
        detector.key: dd for dd, detector in enumerate(detectors)
    }

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
            # relations are deterministic in every round, so they need not compare to the last round
            targets = [
                measurement_record.get_target_rec(check_id, index)
                for index in ((-1,) if detector.is_relation else (-1, -2))
                for check_id in detector.check_ids
            ]
            repeat_circuit.append("DETECTOR", targets, (0, 0, kk))
        circuit.append(stim.CircuitRepeatBlock(num_rounds - 1, repeat_circuit))

        # update the measurement and detector records to account for repetitions
        measurement_record.append(round_measurement_record, repeat=num_rounds - 2)
        detector_record.append(round_detector_record, repeat=num_rounds - 1)

    return circuit, measurement_record, detector_record


def _get_check_detectors(
    code: codes.QuditCode, qubit_ids: QubitIDs, basis: PauliXZ | None = None
) -> list[_StabilizerDetector]:
    """Detectors for the checks of a stabilizer code, keyed by check qubit.

    If a basis is provided, only checks of that type get detectors.
    """
    if basis is None:
        detector_check_ids = qubit_ids.check
    else:
        detector_check_ids = qubit_ids.checks_x if basis is Pauli.X else qubit_ids.checks_z
    check_to_row = {check_id: row for row, check_id in enumerate(qubit_ids.check)}
    return [
        _StabilizerDetector(CheckKey(check_id), code.matrix[check_to_row[check_id]])
        for check_id in detector_check_ids
    ]


def _get_subsystem_detectors(
    code: codes.QuditCode,
    qubit_ids: QubitIDs,
    layers: Sequence[Sequence[int]],
    basis: PauliXZ | None = None,
) -> list[_StabilizerDetector]:
    """Detectors for products of the checks in single gauge layers of a subsystem code.

    See help(qldpc.circuits.get_memory_experiment) for the products that get detectors.  Each
    detector is keyed by the CheckKey or CheckProductKey of the check qubits in its product.  If a
    basis is provided, only products of that type get detectors.

    Raises:
        ValueError: If products of the checks in single layers do not generate all stabilizers of
            the code (of the given type).
    """
    stabilizer_ops = _get_distinct_stabilizers(code, basis)
    layer_relations = [_get_layer_relations(code, layer, basis) for layer in layers]

    # prefer the stabilizer generators of the code, and add other products if necessary
    products = _get_generator_products(code, layers, stabilizer_ops)
    rank = _get_rank(stabilizer_ops)
    if _get_rank(_stack_stabilizers(code, products)) < rank:
        products += _get_completing_products(code, layers, basis, products)
    if _get_rank(_stack_stabilizers(code, products)) != rank:
        raise ValueError(
            "The gauge measurement schedule does not determine all "
            + ("" if basis is None else f"{basis}-type ")
            + "stabilizers of the code: products of the checks within single gauge layers must"
            " generate these stabilizers (see SyndromeMeasurementStrategy.get_gauge_layers)"
        )

    # represent each stabilizer by a product of few checks
    detectors = []
    layer_check_vectors: list[list[npt.NDArray[np.uint8]]] = [[] for _ in layers]
    for product in products:
        relations = layer_relations[product.layer_index]
        check_vector = _reduce_weight(product.check_vector, relations)
        layer_check_vectors[product.layer_index].append(check_vector)
        check_ids = _get_check_ids(qubit_ids, layers[product.layer_index], check_vector)
        detectors.append(_StabilizerDetector(_get_detector_key(check_ids), product.stabilizer))

    # Relations that are products of the stabilizer detectors of a layer get no detectors: in the
    # first round they are products of stabilizer detectors, and in each later round they follow
    # from the previous round and the stabilizer detectors that compare consecutive rounds.
    identity_op = code.field.Zeros(code.matrix.shape[1])
    for layer, relations, check_vectors in zip(layers, layer_relations, layer_check_vectors):
        implied = _get_implied_relations(code, layer, check_vectors)
        for index in _get_span_extension(implied, code.field(relations)):
            check_ids = _get_check_ids(qubit_ids, layer, relations[index])
            key = _get_detector_key(check_ids, is_relation=True)
            detectors.append(_StabilizerDetector(key, identity_op))
    return detectors


def _get_distinct_stabilizers(code: codes.QuditCode, basis: PauliXZ | None) -> galois.FieldArray:
    """The distinct nontrivial stabilizer generators of a code (of a given type), in order."""
    stabilizer_ops = code.get_stabilizer_ops(symplectic=True)
    stabilizer_ops = stabilizer_ops[
        _has_type(stabilizer_ops, basis) & np.any(stabilizer_ops, axis=1)
    ]
    _, first_indices = np.unique(stabilizer_ops.view(np.ndarray), axis=0, return_index=True)
    return stabilizer_ops[np.sort(first_indices)]


def _has_type(ops: galois.FieldArray, basis: PauliXZ | None) -> npt.NDArray[np.bool_]:
    """Identify the rows of a symplectic matrix that are Paulis of a given type (or any if None)."""
    if basis is None:
        return np.ones(len(ops), dtype=bool)
    other_support = ops.reshape(-1, 2, ops.shape[1] // 2)[:, basis.swap_xz(), :]
    return ~np.any(other_support, axis=1)


def _get_layer_relations(
    code: codes.QuditCode, layer: Sequence[int], basis: PauliXZ | None
) -> npt.NDArray[np.uint8]:
    """A low-weight basis of products of checks (of a given type) in a layer that are the identity.

    Each product is an indicator vector over the checks of the layer.
    """
    layer_ops = code.matrix[list(layer)]
    has_type = _has_type(layer_ops, basis)
    relations = np.zeros((0, len(layer)), dtype=np.uint8)
    if np.any(has_type):
        typed_relations = layer_ops[has_type].T.null_space()
        relations = np.zeros((len(typed_relations), len(layer)), dtype=np.uint8)
        relations[:, has_type] = typed_relations.view(np.ndarray)
    return _get_low_weight_basis(relations)


def _get_generator_products(
    code: codes.QuditCode, layers: Sequence[Sequence[int]], stabilizer_ops: galois.FieldArray
) -> list[_LayerProduct]:
    """Express stabilizers as products of the checks in single layers, where possible.

    Each stabilizer is expressed with the checks of the first layer that generates it, and
    stabilizers that no single layer generates are omitted.  Products are returned in the order of
    the stabilizers.
    """
    num_columns = code.matrix.shape[1]
    products: dict[int, _LayerProduct] = {}
    for ll, layer in enumerate(layers):
        unsolved = np.array([ss for ss in range(len(stabilizer_ops)) if ss not in products], int)
        layer_ops = code.matrix[list(layer)]

        # row-reduce [checks | identity] to express stabilizers as products of these checks
        identity_matrix = code.field.Identity(len(layer))
        reduced = np.hstack([layer_ops, identity_matrix]).view(code.field).row_reduce()
        reduced = reduced[np.any(reduced[:, :num_columns], axis=1)]
        pivots = math.first_nonzero_cols(reduced[:, :num_columns])
        targets = stabilizer_ops[unsolved]
        check_vectors = targets[:, pivots] @ reduced[:, num_columns:]
        is_product = ~np.any(check_vectors @ layer_ops - targets, axis=1)
        for ss, check_vector in zip(unsolved[is_product], check_vectors[is_product]):
            check_vector = check_vector.view(np.ndarray).astype(np.uint8)
            products[int(ss)] = _LayerProduct(ll, check_vector, stabilizer_ops[ss])
    return [products[ss] for ss in sorted(products)]


def _get_completing_products(
    code: codes.QuditCode,
    layers: Sequence[Sequence[int]],
    basis: PauliXZ | None,
    products: Sequence[_LayerProduct],
) -> list[_LayerProduct]:
    """Products of the checks in single layers that extend the stabilizers of the given products.

    Every product of checks in a layer that commutes with all checks is a stabilizer.  The returned
    products are independent of each other and of the given products.
    """
    candidates: list[_LayerProduct] = []
    for ll, layer in enumerate(layers):
        layer_ops = code.matrix[list(layer)]
        commutators = layer_ops @ math.symplectic_conjugate(code.matrix).T
        central_vectors = commutators.T.null_space()
        stabilizers = central_vectors @ layer_ops
        has_type = _has_type(stabilizers, basis)
        candidates.extend(
            _LayerProduct(ll, check_vector.view(np.ndarray).astype(np.uint8), stabilizer)
            for check_vector, stabilizer in zip(central_vectors[has_type], stabilizers[has_type])
        )
    new_indices = _get_span_extension(
        _stack_stabilizers(code, products), _stack_stabilizers(code, candidates)
    )
    return [candidates[index] for index in new_indices]


def _get_implied_relations(
    code: codes.QuditCode, layer: Sequence[int], check_vectors: Sequence[npt.NDArray[np.uint8]]
) -> galois.FieldArray:
    """Relations implied by some products of checks in a layer.

    Each relation is a product of the given products that equals the identity operator, returned as
    an indicator vector over the checks of the layer.
    """
    vectors = code.field(np.array(check_vectors, dtype=np.uint8).reshape(-1, len(layer)))
    if not len(vectors):
        return vectors
    return (vectors @ code.matrix[list(layer)]).T.null_space() @ vectors


def _get_span_extension(existing: galois.FieldArray, candidates: galois.FieldArray) -> list[int]:
    """Indices of the candidate rows that, taken in order, extend the row space of existing rows."""
    if not len(candidates):
        return []
    stacked = np.vstack([existing, candidates]).view(type(candidates))
    reduced = stacked.T.row_reduce()
    pivots = math.first_nonzero_cols(reduced[np.any(reduced, axis=1)])
    return [int(pivot) - len(existing) for pivot in pivots if pivot >= len(existing)]


def _stack_stabilizers(
    code: codes.QuditCode, products: Sequence[_LayerProduct]
) -> galois.FieldArray:
    """Stack the stabilizers of some products of checks into a matrix."""
    stabilizers = np.array([product.stabilizer for product in products], dtype=int)
    return code.field(stabilizers.reshape(-1, code.matrix.shape[1]))


def _get_check_ids(
    qubit_ids: QubitIDs, layer: Sequence[int], check_vector: npt.NDArray[np.integer]
) -> tuple[int, ...]:
    """The check qubits of the checks in a layer that are multiplied in a product."""
    return tuple(qubit_ids.check[layer[int(cc)]] for cc in np.flatnonzero(check_vector))


def _get_detector_key(
    check_ids: tuple[int, ...], is_relation: bool = False
) -> CheckKey | CheckProductKey:
    """The key of the detectors that track the product of the outcomes of some checks."""
    if len(check_ids) == 1:
        return CheckKey(check_ids[0], is_relation)
    return CheckProductKey(check_ids, is_relation)


def _get_readout_record_key(key: CheckKey | CheckProductKey, readout_keys: bool) -> Hashable:
    """The record key of the readout detector, or final stabilizer measurement, of a key."""
    return ReadoutKey(key) if readout_keys else key


def _get_low_weight_basis(vectors: npt.NDArray[np.integer]) -> npt.NDArray[np.uint8]:
    """Greedily reduce the Hamming weights of a basis of binary vectors.

    Each basis vector is repeatedly replaced by its sum with another basis vector whenever doing so
    reduces its weight, which preserves the span of the basis.
    """
    basis = np.array(vectors, dtype=np.uint8)
    improved = True
    while improved:
        improved = False
        for vv in range(len(basis)):
            sums = basis ^ basis[vv]
            weights = np.count_nonzero(sums, axis=1)
            weights[vv] = basis.shape[1] + 1
            best = int(np.argmin(weights))
            if weights[best] < np.count_nonzero(basis[vv]):
                basis[vv] = sums[best]
                improved = True
    return basis


def _reduce_weight(
    vector: npt.NDArray[np.integer], others: npt.NDArray[np.integer]
) -> npt.NDArray[np.uint8]:
    """Greedily reduce the Hamming weight of a binary vector by adding other vectors to it."""
    result = np.array(vector, dtype=np.uint8)
    other_vectors = np.asarray(others, dtype=np.uint8)
    while len(other_vectors):
        sums = other_vectors ^ result
        weights = np.count_nonzero(sums, axis=1)
        best = int(np.argmin(weights))
        if weights[best] >= np.count_nonzero(result):
            break
        result = sums[best]
    return result


def _get_rank(matrix: galois.FieldArray) -> int:
    """The rank of a matrix, which may have no rows."""
    return int(np.linalg.matrix_rank(matrix)) if matrix.size else 0
