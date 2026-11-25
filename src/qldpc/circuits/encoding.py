# SPDX-License-Identifier: Apache-2.0

"""Encoding circuits, logical tableaus, and analysis of logical states."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import stim

from qldpc import _deprecations, codes, math
from qldpc._binary import get_independent_rows
from qldpc._utils import get_external_caller_stacklevel
from qldpc.objects import Pauli

from .bookkeeping import QubitIDs
from .common import _get_pauli_product_text, _restrict_to_qubits
from .transforms import with_remapped_qubits

####################################################################################################
# encoding circuits, logical tableaus, and logical states


@_restrict_to_qubits
def get_encoding_tableau(code: codes.QuditCode, *, only_zero: bool = False) -> stim.Tableau:
    """Tableau to encode physical states at its input into logical states of the given code.

    If ``only_zero is True``, this tableau maps an all-0 physical state at its input to an all-0
    logical state at its output.  Otherwise, for all ``j`` in ``{0, 1, ..., code.dimension - 1}``,
    this tableau maps weight-one ``X_j`` and ``Z_j`` operators at its input to the logical X and Z
    operators of the j-th logical qubit of the code.  Weight-one ``Z_j`` operators for
    ``j >= code.dimension`` get mapped to "Z-type" gauge operators and stabilizers, and their
    conjugate ``X_j`` get mapped to "X-type" gauge operators and destabilizers.
    """
    if only_zero:
        return stim.Tableau.from_stabilizers(
            [math.op_to_string(op) for op in code.get_stabilizer_ops(symplectic=True)]
            + [math.op_to_string(op) for op in code.get_logical_ops(Pauli.Z, symplectic=True)],
            allow_redundant=True,
            allow_underconstrained=True,
        )

    # identify logical and gauge operators
    logical_ops = code.get_logical_ops()
    gauge_ops = code.get_gauge_ops()

    # identify a minimal generating set of stabilizers and the dual destabilizers
    stab_ops = code.get_stabilizer_ops()
    if len(stab_ops) != len(code) - code.dimension - code.gauge_dimension:
        # Constructing an encoding circuit requires a minimal choice of stabilizer generators.
        # Canonicalize to remove redundant stabilizers.
        stab_ops = code.get_stabilizer_ops(canonicalized=True)
    destab_ops = code.get_destabilizer_ops()

    # construct Pauli strings to hand over to Stim
    matrices_x = [logical_ops[: code.dimension], gauge_ops[: code.gauge_dimension], destab_ops]
    matrices_z = [logical_ops[code.dimension :], gauge_ops[code.gauge_dimension :], stab_ops]
    strings_x = [math.op_to_string(op) for matrix in matrices_x for op in matrix]
    strings_z = [math.op_to_string(op) for matrix in matrices_z for op in matrix]
    return stim.Tableau.from_conjugated_generators(xs=strings_x, zs=strings_z)


@_restrict_to_qubits
def get_encoding_circuit(code: codes.QuditCode, *, only_zero: bool = False) -> stim.Circuit:
    """Circuit to encode physical states at its input into logical states of the given code.

    If ``only_zero is True``, this circuit maps an all-0 physical state at its input to an all-0
    logical state at its output.  Otherwise, for all ``j`` in ``{0, 1, ..., code.dimension - 1}``,
    this circuit maps weight-one ``X_j`` and ``Z_j`` operators at its input to the logical X and Z
    operators of the j-th logical qubit of the code.  Weight-one ``Z_j`` operators for
    ``j >= code.dimension`` get mapped to "Z-type" gauge operators and stabilizers, and their
    conjugate ``X_j`` get mapped to "X-type" gauge operators and destabilizers.

    .. warning::
        The resulting encoding circuit is not fault-tolerant.

    Args:
        code: The qubit code to encode into.
        only_zero: If True, only require the circuit to prepare the all-0 logical state from an
            all-0 input state.  Default: False.

    Returns:
        A Clifford circuit on qubits ``0, 1, ..., len(code) - 1``, without measurements.

    Raises:
        ValueError: If the code is not a qubit code.
    """
    return get_encoding_tableau(code, only_zero=only_zero).to_circuit()


@_restrict_to_qubits
def get_encoder_and_decoder(
    code: codes.QuditCode, deformation: stim.Circuit | stim.Tableau | None = None
) -> tuple[stim.Tableau, stim.Tableau]:
    """Encoder for a code, and decoder for either the same code or a deformed code.

    The encoder is the tableau of :func:`get_encoding_tableau`, which maps weight-one ``X_j`` and
    ``Z_j`` operators at its input to the logical operators, gauge operators, stabilizers, and
    destabilizers of the code.  Without a deformation, the decoder is the inverse of the encoder.
    With a deformation, the decoder is the inverse of the encoding tableau of the deformed code
    ``code.deformed(deformation, preserve_logicals=True)``, whose stabilizers are those of the code
    transformed by the deformation, and whose logical operators are those of the original code.

    Use this pair to view a physical circuit in the encoded frame: the first ``code.dimension``
    qubits of ``encoder.then(physical_tableau).then(decoder)`` carry the logical operation that the
    physical circuit implements, as in :func:`get_logical_tableau`.  Pass the physical circuit as
    the deformation if it maps the code to a different code, rather than to itself.

    Args:
        code: The qubit code to encode and decode.
        deformation: A physical circuit or tableau that deforms the code before decoding, or None
            (the default) to decode the original code.

    Returns:
        stim.Tableau: The encoder of the code.
        stim.Tableau: The decoder of the code, or of the deformed code if a deformation is given.

    Raises:
        ValueError: If the code is not a qubit code, or if the logical operators of the code are not
            valid logical operators of the deformed code.
    """
    encoder = get_encoding_tableau(code)
    if deformation is None:
        return encoder, encoder.inverse()
    deformation = deformation if isinstance(deformation, stim.Circuit) else deformation.to_circuit()
    deformed_code = code.deformed(deformation, preserve_logicals=True)
    decoder = get_encoding_tableau(deformed_code).inverse()
    return encoder, decoder


@_restrict_to_qubits
def get_logical_tableau(
    code: codes.QuditCode,
    physical_circuit_or_tableau: stim.Circuit | stim.Tableau,
    *,
    deform_code: bool = False,
) -> stim.Tableau:
    """Identify the logical tableau implemented by the physical circuit or tableau.

    If ``deform_code is True``, then the physical circuit is required to have two effects, namely
    (a) transforming a logical state of the QuditCode by a corresponding logical Clifford gate, and
    (b) changing the code that encodes the logical state to::

        code.deformed(physical_circuit, preserve_logicals=True)
    """
    physical_circuit = (
        physical_circuit_or_tableau
        if isinstance(physical_circuit_or_tableau, stim.Circuit)
        else physical_circuit_or_tableau.to_circuit()
    )
    encoder, decoder = get_encoder_and_decoder(code, physical_circuit if deform_code else None)
    return _get_logical_tableau_from_code_data(
        code.dimension, code.gauge_dimension, encoder, decoder, physical_circuit
    )


def restrict_tableau(tableau: stim.Tableau, qubits: Sequence[int]) -> stim.Tableau:
    """Restrict the given stabilizer tableau to the sub-tableau at the specified qubits.

    The selected qubits must be a closed subsystem: the tableau must not map a selected Pauli to
    support outside ``qubits``.  This function does not check that precondition; if it is violated,
    the result is simply the submatrices restricted to ``qubits``, silently discarding any support
    that spilled outside the selection.
    """
    x2x, x2z, z2x, z2z, x_signs, z_signs = tableau.to_numpy()
    return stim.Tableau.from_numpy(
        x2x=x2x[np.ix_(qubits, qubits)],
        x2z=x2z[np.ix_(qubits, qubits)],
        z2x=z2x[np.ix_(qubits, qubits)],
        z2z=z2z[np.ix_(qubits, qubits)],
        x_signs=x_signs[qubits],
        z_signs=z_signs[qubits],
    )


def get_state_stabilizers(
    state_prep_circuit: stim.Circuit, qubits: Sequence[int] | int
) -> list[stim.PauliString]:
    """Identify stabilizers of the prepared state that are supported on specified qubits.

    The circuit runs without noise, from the state ``|0...0⟩``, with every sweep bit 0, as in
    ``stim.TableauSimulator``.  Its measurement outcomes, and so the state that it prepares, may
    vary from run to run.  This function returns a basis of the Pauli strings on the specified
    qubits that stabilize the prepared state with the same sign in every run.

    The circuit's detectors count as post-selected, and its observables are ignored.  The
    detectors' combined outcomes should be deterministic in a noiseless run, as Stim assumes.  A
    stabilizer on the specified qubits that holds only because of post-selection on detectors with
    a random combined outcome has no fixed sign, since these detectors have no noiseless outcome to
    post-select on, and this function raises an error for it.  Every other stabilizer has the same
    sign in every noiseless run, and a simulation of one run finds this sign.

    To find these stabilizers, this function simulates one run of the circuit, and analyzes its
    random outcomes with Stim's detector error model, allowing gauge detectors.

    Args:
        state_prep_circuit: The circuit that prepares the state.
        qubits: The qubits on which we are looking for stabilizers.  If provided an integer, target
            the qubits indexed by range(qubits).

    Returns:
        A list of Pauli strings supported on the specified qubits.  The strings are compacted into
        the requested-qubit frame: position ``i`` refers to ``qubits[i]`` in the input circuit.

    Raises:
        ValueError: If a requested qubit is negative, repeated, or not an integer, or if qubits is a
            negative integer.
        ValueError: If the sign of a stabilizer on the requested qubits depends on detectors whose
            combined outcome is random, as for Z on qubit 1 after
            ``H 0; CX 0 1; M 0; DETECTOR rec[-1]``.
    """
    requested = range(qubits) if isinstance(qubits, int) else qubits
    state_qubits = np.asarray(requested, dtype=int)
    if (
        (isinstance(qubits, int) and qubits < 0)
        or not np.array_equal(state_qubits, requested)
        or np.any(state_qubits < 0)
        or len(np.unique(state_qubits)) < len(state_qubits)
    ):
        raise ValueError(f"Requested qubits must be distinct nonnegative integers, not {qubits}")

    # reset every qubit, including requested qubits that the circuit leaves untouched in |0>
    num_qubits = max(state_prep_circuit.num_qubits, int(state_qubits.max(initial=-1)) + 1)
    resets = stim.Circuit("R " + " ".join(map(str, range(num_qubits))))
    circuit = resets + state_prep_circuit.without_noise()

    # Simulate one run, which gives the signs of the stabilizers whose signs are fixed.  Finding the
    # fixed parities below reuses this run.
    simulator = stim.TableauSimulator()
    # TEMPORARY until stim.TableauSimulator resets an unentangled qubit in linear time: see
    # circuits/stim_workarounds.md, "Resets with random outcomes in stim.TableauSimulator".
    if circuit.num_qubits < _MIN_QUBITS_FOR_CHEAP_RESETS:
        simulator.do(circuit)
    else:
        _do_with_cheap_resets(simulator, circuit)

    # Find the parities that the circuit fixes: the pairs (D, P) of a set D of detectors and a Pauli
    # string P on the requested qubits for which the outcome of measuring P after the circuit,
    # combined with the outcomes of the detectors in D, is deterministic.  Simulation finds them
    # about 15 times faster than stim.Circuit.flow_generators on a circuit of about 8000 qubits and
    # measurements, since flows track every measurement.  _get_fixed_parities_by_simulation returns
    # some detectors, and the reduced row echelon form of the fixed parities whose D is among them,
    # in the column order [D | X part of P | Z part of P].  Each detector that it leaves out is
    # deterministic on its own, so leaving it out changes no row with a Pauli string.
    detectors, rows = _get_fixed_parities_by_simulation(circuit, simulator, state_qubits)
    num_detectors = len(detectors)
    on_detectors = np.any(rows[:, :num_detectors], axis=1)
    on_state = np.any(rows[:, num_detectors:], axis=1)
    state_parts = rows[:, num_detectors:] != 0
    num_state_qubits = len(state_qubits)

    # Raise if a stabilizer on the requested qubits needs detectors, since its sign is then
    # undetermined.  A row (D, P) with both D and P nonzero fixes the sign of P only jointly with
    # D.  Its P vanishes at the pivots of the rows without detectors, so it is not a product of
    # them.  If the detectors in D had a deterministic combined outcome, then (D, 0) would be a
    # fixed parity, and adding it would remove D.  So that outcome, which fixes the sign, is random.
    undetermined = np.flatnonzero(on_state & on_detectors)
    if len(undetermined):
        part = state_parts[undetermined[0]]
        stabilizer = stim.PauliString.from_numpy(
            xs=part[:num_state_qubits], zs=part[num_state_qubits:]
        )
        detector_part = rows[undetermined[0], :num_detectors]
        raise _UndeterminedSignError(
            stabilizer, state_qubits.tolist(), detectors[np.flatnonzero(detector_part)].tolist()
        )

    # collect the stabilizers that the circuit fixes without detectors
    stabilizers = [
        stim.PauliString.from_numpy(xs=part[:num_state_qubits], zs=part[num_state_qubits:])
        for part in state_parts[on_state & ~on_detectors]
    ]

    # Fix signs, which are the same in every run of the circuit.  Build each physical Pauli string
    # from arrays, since assigning Paulis to a stim.PauliString one qubit at a time is slow.
    for ss, stabilizer in enumerate(stabilizers):
        xs, zs = stabilizer.to_numpy()
        physical_xs = np.zeros(num_qubits, dtype=bool)
        physical_zs = np.zeros(num_qubits, dtype=bool)
        physical_xs[state_qubits] = xs
        physical_zs[state_qubits] = zs
        physical_stabilizer = stim.PauliString.from_numpy(xs=physical_xs, zs=physical_zs)
        expectation = simulator.peek_observable_expectation(physical_stabilizer)
        if expectation == 0:
            raise ValueError("The prepared state is not an eigenstate of a returned stabilizer")
        if expectation == -1:
            stabilizers[ss] = -stabilizer

    return stabilizers


@_restrict_to_qubits
def get_logical_state_stabilizers(
    code: codes.QuditCode, state_prep_circuit: stim.Circuit, qubit_ids: QubitIDs | None = None
) -> list[stim.PauliString]:
    """Identify pure logical operators that stabilize the state prepared by the provided circuit.

    The data qubits of the code are those specified by qubit_ids, and the circuit may address other
    qubits as well.

    Args:
        code: The code whose logical state is prepared by the provided state_prep_circuit.
        state_prep_circuit: A circuit that prepares a logical state of the provided code.  The
            argument order (state_prep_circuit, code) is deprecated; pass the code first.
        qubit_ids: A QubitIDs object specifying the indices of the data qubits of the code.  If
            None, the data qubits of the code are assumed to be range(len(code)).

    Returns:
        A list of Pauli strings on len(code) qubits, whose j-th qubit is the data qubit
        qubit_ids.data[j] of the provided code.

    Raises:
        ValueError: If the sign of a logical stabilizer depends on detectors whose combined outcome
            is random, as ``get_state_stabilizers`` describes.
    """
    # DEPRECATED: see qldpc._deprecations
    if isinstance(code, stim.Circuit) and isinstance(state_prep_circuit, codes.QuditCode):
        _deprecations.warn_deprecated(
            "Check argument order: it should be"
            " get_logical_state_stabilizers(code, state_prep_circuit, qubit_ids).  The order"
            " (state_prep_circuit, code, qubit_ids) is DEPRECATED and will throw an error in the"
            " future!",
            stacklevel=get_external_caller_stacklevel(),
        )
        code, state_prep_circuit = state_prep_circuit, code

    qubit_ids = QubitIDs.validated(qubit_ids or QubitIDs.from_code(code), code)
    encoder = get_encoding_circuit(code)
    circuit = state_prep_circuit + with_remapped_qubits(encoder.inverse(), qubit_ids.data)

    # Decoding puts logical qubit j on qubit_ids.data[j], so name the Paulis of an undetermined
    # stabilizer by their logical qubits.
    try:
        decoded_stabilizers = get_state_stabilizers(circuit, qubit_ids.data[: code.dimension])
    except _UndeterminedSignError as error:
        raise _UndeterminedSignError(
            error.stabilizer, list(range(code.dimension)), error.detectors, logical=True
        ) from error
    identity = stim.PauliString(num_qubits=len(code))
    return [(string * identity).after(encoder) for string in decoded_stabilizers]


class _UndeterminedSignError(ValueError):
    """The sign of a stabilizer depends on detectors whose combined outcome is random.

    The stabilizer is compacted into the frame of the given qubit labels, and has no sign.
    """

    def __init__(
        self,
        stabilizer: stim.PauliString,
        labels: list[int],
        detectors: list[int],
        *,
        logical: bool = False,
    ) -> None:
        self.stabilizer = stabilizer
        self.detectors = detectors
        support = stabilizer.pauli_indices()
        string = " ".join(f"{'_XYZ'[stabilizer[ii]]}{labels[ii]}" for ii in support)
        kind, frame = ("logical stabilizer", "logical") if logical else ("stabilizer", "circuit")
        super().__init__(
            f"The sign of the {kind} {string} (indexed by {frame} qubit) of the prepared state"
            f" depends on detectors {detectors}, whose combined outcome is random, so"
            " post-selecting on them leaves the sign undetermined; the circuit's detectors should"
            " have deterministic outcomes in a noiseless run"
        )


# TEMPORARY until stim.TableauSimulator resets an unentangled qubit in linear time: see
# circuits/stim_workarounds.md, "Resets with random outcomes in stim.TableauSimulator".
# Simulating a circuit one instruction at a time, as _do_with_cheap_resets does, costs about 1-2 µs
# per instruction, while a reset whose outcome is random costs Stim about 0.6 n² ns on n qubits.  So
# get_state_stabilizers simulates a circuit on fewer qubits than this in one call.
_MIN_QUBITS_FOR_CHEAP_RESETS = 256

# TEMPORARY until stim.TableauSimulator resets an unentangled qubit in linear time: see
# circuits/stim_workarounds.md, "Resets with random outcomes in stim.TableauSimulator".
# The Pauli whose +1 eigenstate each reset prepares.  For each Pauli, the stim.TableauSimulator
# method of a Pauli gate that anticommutes with it.  For each pair of distinct Paulis, the method of
# a Clifford gate that swaps them.
_RESET_PAULIS = {"R": "Z", "RX": "X", "RY": "Y"}
_PAULI_FLIPS = {"X": "z", "Y": "z", "Z": "x"}
_PAULI_SWAPS = {"XY": "h_xy", "YX": "h_xy", "XZ": "h", "ZX": "h", "YZ": "h_yz", "ZY": "h_yz"}


def _do_with_cheap_resets(simulator: stim.TableauSimulator, circuit: stim.Circuit) -> None:
    """Simulate a noiseless circuit as simulator.do(circuit) does, replacing costly resets by gates.

    TEMPORARY until stim.TableauSimulator resets an unentangled qubit in linear time: see
    circuits/stim_workarounds.md, "Resets with random outcomes in stim.TableauSimulator".

    A reset whose outcome is random costs Stim's tableau simulator time quadratic in the number of
    qubits, since it collapses the qubit by transposing the whole tableau.  But if the qubit is in
    an eigenstate of a single-qubit Pauli other than that of the reset, then it is unentangled with
    the other qubits, and a single-qubit Clifford that maps this eigenstate to the state that the
    reset prepares has the same effect as the reset, at a cost linear in the number of qubits.  This
    is the case for a reset of a fresh qubit in another basis, or of a qubit measured in another
    basis.

    So this function simulates the circuit one top-level instruction at a time, and replaces the
    reset of each such qubit by such a Clifford.  This changes neither the state after the reset nor
    the distribution of later measurement outcomes, so the simulator ends in the state of a run of
    the circuit, as after simulator.do(circuit).  Resets in REPEAT blocks run as they are.
    """
    peeks = {"X": simulator.peek_x, "Y": simulator.peek_y, "Z": simulator.peek_z}
    for index in range(len(circuit)):
        instruction = circuit[index]
        reset_pauli = _RESET_PAULIS.get(instruction.name)
        if reset_pauli is None:
            simulator.do(instruction)
            continue

        # Replace the reset of each qubit in an eigenstate of another Pauli.  Leave every other
        # reset to Stim, which is cheap for a qubit already in an eigenstate of the reset's Pauli.
        kept = []
        for target in instruction.targets_copy():
            qubit = target.value
            signs = {pauli: peek(qubit) for pauli, peek in peeks.items()}
            pauli = next((pauli for pauli, sign in signs.items() if sign), reset_pauli)
            if pauli == reset_pauli:
                kept.append(qubit)
                continue
            if signs[pauli] < 0:
                getattr(simulator, _PAULI_FLIPS[pauli])(qubit)
            getattr(simulator, _PAULI_SWAPS[pauli + reset_pauli])(qubit)
        if kept:
            getattr(simulator, f"reset_{reset_pauli.lower()}")(*kept)


def _get_fixed_parities_by_simulation(
    circuit: stim.Circuit, simulator: stim.TableauSimulator, state_qubits: npt.NDArray[np.int_]
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.uint8]]:
    """Find the fixed parities of a circuit by simulation.

    Fixed parities are as the comment in get_state_stabilizers defines them.  The circuit must be
    noiseless and reset every qubit first, and the simulator must have run it.  The requested
    qubits, state_qubits, must be distinct qubits of the circuit.  get_state_stabilizers ensures all
    of this.  Returns the circuit's detectors that some random outcome flips, and the reduced row
    echelon form of the fixed parities whose detectors are among them.
    """
    num_qubits = circuit.num_qubits
    num_state_qubits = len(state_qubits)

    # In a stabilizer circuit, measurement outcomes change only the signs of the final stabilizers.
    # Up to sign, these are the images T(Z_i), where T is the inverse of the simulator's inverse
    # tableau V.  The inverse of a symplectic matrix S is Ω Sᵀ Ω, where Ω = [[0, I], [I, 0]], so
    # T(Z_i) has X part z2x[:, i] and Z part x2x[:, i] of V, and V need not be inverted.
    x2x, _, z2x, *_ = simulator.current_inverse_tableau().to_numpy()

    # Row-reduce the final stabilizers with the columns of the qubits not requested first.  The rows
    # that vanish on these columns span the candidates: the stabilizers on the requested qubits, up
    # to sign.
    other_qubits = np.setdiff1d(np.arange(num_qubits), state_qubits)
    generators = np.concatenate(
        [z2x[other_qubits], x2x[other_qubits], z2x[state_qubits], x2x[state_qubits]]
    ).T
    reduced = get_independent_rows(np.ascontiguousarray(generators, dtype=np.uint8))
    num_other_columns = 2 * len(other_qubits)
    candidates = reduced[~np.any(reduced[:, :num_other_columns], axis=1), num_other_columns:]

    # Measure each candidate after the circuit, with a detector on its outcome.  Stim's detector
    # error model then lists the gauges: for each random outcome, the detectors that it flips, up
    # to Gaussian elimination.  Every error of the model is a gauge, since the circuit is
    # noiseless.  Detectors have a deterministic combined outcome exactly when every gauge flips an
    # even number of them.  Stim rejects a random observable even with gauge detectors allowed, so
    # the model leaves out the circuit's observables, which change neither its state nor its
    # measurement record.
    # TEMPORARY until stim>=1.17: see common._get_pauli_product_text.
    lines = []
    for candidate in candidates.astype(bool):
        xs = np.zeros(num_qubits, dtype=bool)
        zs = np.zeros(num_qubits, dtype=bool)
        xs[state_qubits] = candidate[:num_state_qubits]
        zs[state_qubits] = candidate[num_state_qubits:]
        pauli_string = stim.PauliString.from_numpy(xs=xs, zs=zs)
        lines.append(f"MPP {_get_pauli_product_text(pauli_string)}\nDETECTOR rec[-1]")
    measurements = stim.Circuit("\n".join(lines))
    dem = (_without_observables(circuit) + measurements).detector_error_model(
        allow_gauge_detectors=True
    )
    gauges = [
        [
            target.val
            for target in instruction.targets_copy()
            if isinstance(target, stim.DemTarget) and target.is_relative_detector_id()
        ]
        for instruction in dem.flattened()
        if isinstance(instruction, stim.DemInstruction) and instruction.type == "error"
    ]

    # Keep the circuit's detectors that some gauge flips.  Each other detector is deterministic on
    # its own, so the matrix below needs no row or column for any of them, and most detectors of
    # a large circuit, such as flag detectors, are of this kind.  Then index the kept detectors and
    # the candidates' detectors.
    num_detectors = circuit.num_detectors
    num_candidates = len(candidates)
    flipped = np.zeros(num_detectors + num_candidates, dtype=bool)
    for gauge in gauges:
        flipped[gauge] = True
    detectors = np.flatnonzero(flipped[:num_detectors])
    num_kept = len(detectors)
    indices = np.zeros(num_detectors + num_candidates, dtype=int)
    indices[detectors] = np.arange(num_kept)
    indices[num_detectors:] = num_kept + np.arange(num_candidates)

    # Give each kept detector and candidate a row: the gauges that flip it, then the fixed parity
    # (D, P) that it stands for.  Row reduction with the gauge columns first leaves the rows that
    # no gauge flips, which span the fixed parities, in reduced row echelon form.
    num_gauges = len(gauges)
    matrix = np.zeros(
        (num_kept + num_candidates, num_gauges + num_kept + 2 * num_state_qubits), dtype=np.uint8
    )
    for gg, gauge in enumerate(gauges):
        matrix[indices[gauge], gg] = 1
    matrix[:num_kept, num_gauges : num_gauges + num_kept] = np.eye(num_kept, dtype=np.uint8)
    matrix[num_kept:, num_gauges + num_kept :] = candidates
    rows = get_independent_rows(matrix)
    fixed = ~np.any(rows[:, :num_gauges], axis=1)
    return detectors, rows[fixed, num_gauges:]


def _without_observables(circuit: stim.Circuit) -> stim.Circuit:
    """A given circuit without its OBSERVABLE_INCLUDE instructions, also in REPEAT blocks.

    If it has no observables, this returns the given circuit itself, not a copy.
    """
    if not circuit.num_observables:
        return circuit
    stripped = stim.Circuit()
    for instruction in circuit:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            body = _without_observables(instruction.body_copy())
            stripped.append(stim.CircuitRepeatBlock(instruction.repeat_count, body))
        elif instruction.name != "OBSERVABLE_INCLUDE":
            stripped.append(instruction)
    return stripped


def _get_logical_tableau_from_code_data(
    dimension: int,  # number of logical qubits of a QuditCode
    gauge_dimension: int,  # number of gauge qubits of a QuditCode
    encoder: stim.Tableau,
    decoder: stim.Tableau,
    physical_circuit: stim.Circuit,
    skip_validation: bool = False,
) -> stim.Tableau:
    """Identify the logical tableau implemented by the physical circuit."""
    if len(encoder) != len(decoder) or len(encoder) < dimension + gauge_dimension:
        raise ValueError("Encoder and decoder have incompatible dimensions")
    identity_phys = stim.Circuit(f"I {len(encoder) - 1}")
    physical_tableau = (physical_circuit + identity_phys).to_tableau()

    # Validate the decoded physical action before restricting it.  Restriction can itself fail for
    # an invalid action, obscuring the useful public error below.
    decoded_tableau = encoder.then(physical_tableau).then(decoder)
    if not skip_validation:
        # identify sectors that address logical, gauge, and stabilizer qubits
        sector_l = slice(dimension)
        sector_g = slice(dimension, dimension + gauge_dimension)
        sector_s = slice(dimension + gauge_dimension, len(encoder))
        x2x, x2z, z2x, z2z, _, z_signs = decoded_tableau.to_numpy()

        # sanity check: stabilizers, logicals, and gauge operators should not pick up destabilizers
        ops_acquired_destabilizers = (
            np.any(z2x[:, sector_s])
            or np.any(x2x[sector_l, sector_s])
            or np.any(x2x[sector_g, sector_s])
        )

        # sanity check: gauge operators should not pick up logical factors
        gauges_acquired_logicals = (
            np.any(x2x[sector_g, sector_l])
            or np.any(x2z[sector_g, sector_l])
            or np.any(z2x[sector_g, sector_l])
            or np.any(z2z[sector_g, sector_l])
        )

        # sanity check: no stabilizers get flipped
        stabilizers_flipped = np.any(z_signs[sector_s])

        if ops_acquired_destabilizers or gauges_acquired_logicals or stabilizers_flipped:
            raise ValueError("The provided physical circuit does not implement a logical operation")

    # compute the "upper left" block of the decoded tableau that acts on all logical qubits
    logical_tableau = restrict_tableau(decoded_tableau, range(dimension))
    return logical_tableau
