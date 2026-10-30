# SPDX-License-Identifier: Apache-2.0

"""Exact certification of bounded fault propagation in state-preparation circuits."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass

import galois
import numpy as np
import numpy.typing as npt
import stim

from qldpc import math
from qldpc.codes.distance._bounded_weight_search import MinWeightBound

from .._post_selection import get_flag_detectors, get_selected_detectors, is_integer
from ..bookkeeping import DetectorRecord
from ..common import get_pauli_product_measurements
from ..encoding import get_state_stabilizers
from ._fault_search import find_smallest_violation

####################################################################################################
# bounded fault propagation certification


def certify_bounded_fault_propagation(
    state_prep_circuit: stim.Circuit,
    *,
    add_flags: bool = False,
    post_select: bool | Collection[int] | None = None,
    max_weight: int | None = None,
    css_weight: bool | None = None,
    output_qubits: Sequence[int] | None = None,
    memory_limit: int = 2**30,  # 1 GiB
) -> BoundedFaultPropagationCertificate:
    """Certify bounded fault propagation in a noisy Clifford state-preparation circuit.

    Fault-tolerant state preparation requires that a few faults cannot leave an error on the
    prepared state that is too heavy for later error correction.  This function checks a sufficient
    criterion, bounded fault propagation (Definition 3 in Section 4.2 of `arXiv:0904.2557
    <https://arxiv.org/abs/0904.2557>`_), and returns a certificate of the result.

    Faults are sets of error mechanisms, which are the error instructions of the detector error
    model (DEM) of the circuit after adding any flag detectors and one noiseless detector for each
    stabilizer of the prepared state.  The certificate stores this DEM as ``detector_error_model``,
    and the circuit with those detectors as ``diagnostic_circuit``.  Stim merges physical errors
    that flip the same detectors and observables into one error mechanism, which does not change the
    result.  A fault is represented by a binary vector over the error mechanisms, its fault weight
    is its number of error mechanisms, and a primitive fault is a fault of fault weight one.

    A fault leaves a Pauli error, its output error, on the output qubits that hold the prepared
    state.  The Pauli weight of a Pauli error is either its physical weight (the number of qubits
    that it acts on) or its CSS weight ``max(|x|, |z|)``, where ``x`` and ``z`` are its X and Z
    parts.  Two Pauli errors are equivalent if they differ by a stabilizer of the prepared state,
    and the residual weight of a fault is the smallest Pauli weight of a Pauli error equivalent to
    its output error.  Fault propagation is bounded if every fault that passes post-selection
    satisfies the bound that its residual weight is at most its fault weight.

    Without post-selection, it suffices to check that each primitive fault alone has residual weight
    at most one, because Pauli errors propagate linearly through a Clifford circuit and Pauli weight
    is subadditive.  With post-selection, primitive faults that each fire a detector can pass
    together, so this function also considers the irreducible accepted faults, which are the faults
    that pass post-selection and contain no other nonempty fault that passes.

    The certificate covers only the faults of this model: operations without noise are perfect,
    error instructions with probability zero are not faults, and alternatives that exclude each
    other, such as the cases of an ``ELSE_CORRELATED_ERROR`` chain, count as separate error
    mechanisms that can occur together.

    Args:
        state_prep_circuit: Noisy Stim Clifford circuit starting in Stim's all-zero state.

    Keyword args:
        add_flags: Whether to add a flag detector for each measurement that no detector or
            observable addresses, as in ``get_state_prep_diagnostic_circuit``.  By default,
            ``post_select`` then selects these flag detectors.  Default: False.
        post_select: The detectors to post-select on, among the circuit's own detectors followed by
            any added flag detectors: None for exactly the added flag detectors, True for all of
            them, False for none, or a collection of their indices (negative indices count back from
            the last detector).  A fault passes post-selection if none of the selected detectors
            fire, where a detector fires if an odd number of the fault's error mechanisms flip it.
            Default: None, which post-selects on the added flag detectors, if any, but never on the
            circuit's own detectors.
        max_weight: The largest fault weight to certify, a positive integer.  Default: None, for no
            limit.  It does not change the result without post-selection.  With post-selection, the
            search time can grow exponentially with the fault weight, so a limit bounds it.
        css_weight: True to choose CSS weight as the Pauli weight, or False to choose physical
            weight.  Default: CSS weight if the prepared state is a CSS state in the physical Pauli
            basis (meaning its stabilizer group is generated by pure-X and pure-Z operators), and
            physical weight otherwise.  A CSS code can prepare a non-CSS state, such as a logical Y
            eigenstate; pass ``css_weight=True`` for such a state if CSS weight is appropriate.
        output_qubits: Distinct indices of the output qubits, at least one, in the order that the
            certificate uses for Pauli strings.  Default: all qubits that the circuit acts on and
            never measures.
        memory_limit: The approximate number of bytes that the search may use for its lookup tables,
            a positive integer.  A smaller limit can make the search slower, but never changes the
            result.  Default: ``2**30`` (1 GiB).

    Returns:
        A certificate of bounded fault propagation, which includes a counterexample if applicable.

    Raises:
        TypeError: If ``max_weight`` or ``memory_limit`` is not an integer, which a bool is not
            here, if ``css_weight`` is not a bool or None, or if ``output_qubits`` or
            ``post_select`` has an invalid type, such as a string or a non-integer index.
        ValueError: If ``max_weight`` or ``memory_limit`` is not positive, if ``output_qubits`` is
            empty, repeats a qubit, or has an index out of range, if ``post_select`` has an index
            out of range, if the output qubits do not end in one definite stabilizer state, or if
            Stim cannot convert the circuit to a DEM.
    """
    if max_weight is not None:
        if not is_integer(max_weight):
            raise TypeError("max_weight must be an integer or None")
        if max_weight < 1:
            raise ValueError("max_weight must be positive")
        max_weight = int(max_weight)
    if css_weight is not None and not isinstance(css_weight, bool):
        raise TypeError("css_weight must be a bool or None")
    if not is_integer(memory_limit):
        raise TypeError("memory_limit must be an integer")
    if memory_limit < 1:
        raise ValueError("memory_limit must be positive")
    if output_qubits is None:
        output_qubits = _get_unmeasured_qubits(state_prep_circuit)
        if not output_qubits:
            raise ValueError("The circuit has no unmeasured qubits to use as output_qubits")
    qubits = _validate_output_qubits(output_qubits, state_prep_circuit.num_qubits)

    # Flag detectors come after the detectors that the circuit already has.
    detector_record = DetectorRecord({"prep": range(state_prep_circuit.num_detectors)})
    flag_detectors = get_flag_detectors(state_prep_circuit) if add_flags else stim.Circuit()
    detector_record.append({"flags": range(flag_detectors.num_detectors)})
    selected = get_selected_detectors(post_select, detector_record)
    circuit = state_prep_circuit + flag_detectors

    stabilizers = get_state_stabilizers(circuit, qubits)
    if len(stabilizers) != len(qubits):
        raise ValueError("Selected outputs must have a definite pure, complete stabilizer state")
    matrix = np.array([math.string_to_op(stabilizer) for stabilizer in stabilizers], dtype=np.uint8)
    css_weight = _is_css_state(matrix) if css_weight is None else css_weight
    diagnostic_circuit, dem, flags, syndromes = _get_primitive_fault_signatures(
        circuit, stabilizers, qubits, selected
    )
    violation = find_smallest_violation(
        flags, syndromes, matrix, max_weight, css_weight, int(memory_limit)
    )
    return BoundedFaultPropagationCertificate(
        output_qubits=qubits,
        state_stabilizers=tuple(stabilizers),
        diagnostic_circuit=diagnostic_circuit,
        detector_error_model=dem,
        post_selection_detectors=selected,
        max_weight=max_weight,
        css_weight=css_weight,
        counterexample=None if violation is None else _get_counterexample(*violation, len(qubits)),
    )


@dataclass(frozen=True)
class BoundedFaultPropagationCertificate:
    """The result of certifying bounded fault propagation in a state-preparation circuit.

    Equivalence, fault weight, Pauli weight, residual weight, and bounded fault propagation are as
    defined in ``certify_bounded_fault_propagation``.

    Attributes are in dependency order: each refers only to the ones before it.

    Attributes:
        output_qubits: The output qubit indices.  The ``i``-th Pauli of every Pauli string in this
            certificate acts on qubit ``output_qubits[i]``.
        state_stabilizers: Stim Pauli strings that generate the stabilizer group of the prepared
            state, one per output qubit.
        diagnostic_circuit: The circuit that the certificate analyzes: the certified circuit,
            followed by any flag detectors and noiseless measurements of the state stabilizers with
            one detector each.  Its detectors are the certified circuit's own, then any added flag
            detectors, then one for each state stabilizer, in order.  Pass it to Stim's
            ``explain_detector_error_model_errors`` to locate the faults of a counterexample.
        detector_error_model: The detector error model of ``diagnostic_circuit``, flattened and
            undecomposed.  Its error instructions are the error mechanisms.
        post_selection_detectors: The post-selection detectors, as indices in increasing order among
            the detectors of ``detector_error_model``.
        max_weight: The largest fault weight that the certificate covers, or None for no limit.
        css_weight: Whether the certificate chose CSS weight (True) or physical weight (False) as
            the Pauli weight.
        counterexample: The counterexample if ``certified`` is False, and None otherwise.
    """

    output_qubits: tuple[int, ...]
    state_stabilizers: tuple[stim.PauliString, ...]
    diagnostic_circuit: stim.Circuit
    detector_error_model: stim.DetectorErrorModel
    post_selection_detectors: tuple[int, ...]
    max_weight: int | None
    css_weight: bool
    counterexample: BoundedFaultPropagationCounterexample | None

    @property
    def certified(self) -> bool:
        """Whether fault propagation is bounded, within the limit ``max_weight``."""
        return self.counterexample is None

    def __str__(self) -> str:
        """Summarize the certificate, and locate the faults of a counterexample in the circuit.

        Each error mechanism of a counterexample is traced to one circuit error that produces it,
        given by its Pauli error, noise instruction, qubits, instruction index, and tick.
        Instruction indices start at 0 and index the instructions of the certified circuit, which
        are also the first instructions of ``diagnostic_circuit``; inside a REPEAT block, an index
        counts within the block and the printout names the iteration (from 0) and the block's own
        index.
        """
        limit = "any fault weight" if self.max_weight is None else f"fault weight {self.max_weight}"
        metric = "CSS weight" if self.css_weight else "physical weight"
        num_selected = len(self.post_selection_detectors)
        selection = (
            f"post-selecting on {_count(num_selected, 'detector')}"
            if num_selected
            else "no post-selection"
        )
        settings = f"({metric}, {selection})"
        counterexample = self.counterexample
        if counterexample is None:
            return f"Bounded fault propagation certified up to {limit} {settings}"

        weight = counterexample.fault_weight
        lines = [f"Bounded fault propagation violated at fault weight {weight} {settings}"]
        passes = ", which passes post-selection" if num_selected else ""
        lines.append(f"  fault of {_count(weight, 'error mechanism')}{passes}:")
        errors = [inst for inst in self.detector_error_model if inst.type == "error"]
        for index in counterexample.fault_indices:
            lines.append(f"    {_describe_error_mechanism(self.diagnostic_circuit, errors[index])}")
        pauli = counterexample.output_pauli
        x_support = [self.output_qubits[qq] for qq in pauli.pauli_indices("XY")]
        z_support = [self.output_qubits[qq] for qq in pauli.pauli_indices("YZ")]
        lines.append(f"  output error, up to stabilizers: X on {x_support}, Z on {z_support}")
        bound = counterexample.residual_weight_lower_bound
        lines.append(f"  residual weight >= {bound} > fault weight {weight}")
        return "\n".join(lines)


@dataclass(frozen=True)
class BoundedFaultPropagationCounterexample:
    """A fault that passes post-selection and violates the bound.

    Its residual weight exceeds its fault weight.  No fault that passes post-selection and has a
    smaller fault weight violates the bound.  Equivalence, fault weight, Pauli weight, output error,
    residual weight, and the bound are as defined in ``certify_bounded_fault_propagation``.

    Attributes are in dependency order: each refers only to the ones before it.

    Attributes:
        fault_indices: The indices, among the error instructions of the certificate's
            ``detector_error_model``, of the primitive faults that make up the fault, in increasing
            order.
        output_pauli: A Pauli error that is equivalent to the fault's output error, on the output
            qubits in the order of the certificate's ``output_qubits``.  It need not be the output
            error itself, and its Pauli weight need not be the residual weight.
        output_syndrome: The syndrome of ``output_pauli``, and so of the output error: entry ``j``
            is 1 if the error anticommutes with the certificate's ``state_stabilizers[j]``.
        residual_weight_lower_bound: A lower bound on the fault's residual weight, which is the
            smallest Pauli weight of an error equivalent to ``output_pauli``; it exceeds
            ``fault_weight``.
    """

    fault_indices: tuple[int, ...]
    output_pauli: stim.PauliString
    output_syndrome: tuple[int, ...]
    residual_weight_lower_bound: int

    @property
    def fault_weight(self) -> int:
        """The fault weight of this counterexample's fault."""
        return len(self.fault_indices)


def _get_counterexample(
    indices: tuple[int, ...], syndrome: int, bound: MinWeightBound, num_qubits: int
) -> BoundedFaultPropagationCounterexample:
    """Describe a violating fault by its DEM indices and an error on the output qubits."""
    n = num_qubits
    pauli = math.op_to_string(np.array([(bound.representative >> jj) & 1 for jj in range(2 * n)]))
    return BoundedFaultPropagationCounterexample(
        fault_indices=tuple(sorted(indices)),
        output_pauli=pauli,
        output_syndrome=tuple((syndrome >> jj) & 1 for jj in range(n)),
        residual_weight_lower_bound=bound.lower_bound,
    )


def _count(number: int, noun: str) -> str:
    """Count a noun, as in "1 detector" or "2 detectors"."""
    return f"{number} {noun}" if number == 1 else f"{number} {noun}s"


def _describe_error_mechanism(circuit: stim.Circuit, error: stim.DemInstruction) -> str:
    """Describe one circuit location of an error mechanism, or the mechanism if Stim finds none."""
    explained = circuit.explain_detector_error_model_errors(
        dem_filter=stim.DetectorErrorModel(str(error)), reduce_to_one_representative_error=True
    )
    locations = [loc for explanation in explained for loc in explanation.circuit_error_locations]
    if not locations:
        return str(error)
    location = locations[0]
    flips = [
        f"{tt.gate_target.pauli_type}{tt.gate_target.value}"
        for tt in location.flipped_pauli_product
    ]
    if location.flipped_measurement is not None:
        flips.append(f"flip of measurement {location.flipped_measurement.record_index}")
    targets = location.instruction_targets
    qubits = [tt.gate_target.value for tt in targets.targets_in_range]

    # Stack frames drill down from the top-level instruction into nested REPEAT blocks.
    frames = location.stack_frames
    position = f"instruction {frames[-1].instruction_offset}"
    for parent, child in zip(reversed(frames[:-1]), reversed(frames[1:])):
        position += (
            f" in iteration {child.iteration_index}"
            f" of REPEAT instruction {parent.instruction_offset}"
        )
    return (
        f"{' '.join(flips)} from {targets.gate} on qubits {qubits}"
        f" at {position}, tick {location.tick_offset}"
    )


def _get_primitive_fault_signatures(
    circuit: stim.Circuit,
    stabilizers: Sequence[stim.PauliString],
    qubits: Sequence[int],
    selected: Sequence[int],
) -> tuple[stim.Circuit, stim.DetectorErrorModel, list[int], list[int]]:
    """Get the certificate's circuit and DEM, and the flags and syndrome of each primitive fault.

    The flags of a primitive fault have bit ``j`` for the ``j``-th selected detector, and its
    syndrome has bit ``j`` for stabilizer ``j``.  Both lists follow the order of the DEM's error
    instructions.
    """
    diagnostics = get_pauli_product_measurements(stabilizers, qubits=qubits)
    for jj in range(len(stabilizers)):
        diagnostics.append("DETECTOR", [stim.target_rec(jj - len(stabilizers))])
    diagnostic_circuit = circuit + diagnostics

    # Stim raises an error here for a nondeterministic detector or observable.  It needs the option
    # for correlated error chains, and treats their components as independent error mechanisms,
    # which only approximates their probabilities, and no probabilities are used here.
    dem = diagnostic_circuit.detector_error_model(approximate_disjoint_errors=True).flattened()

    flag_bit = {detector: jj for jj, detector in enumerate(selected)}
    flags: list[int] = []
    syndromes: list[int] = []
    for instruction in dem:
        if instruction.type != "error":
            continue
        # Detector ``circuit.num_detectors + j`` measures stabilizer ``j``.
        flag = syndrome = 0
        for target in instruction.targets_copy():
            if not target.is_relative_detector_id():
                continue
            if target.val in flag_bit:
                flag ^= 1 << flag_bit[target.val]
            if target.val >= circuit.num_detectors:
                syndrome ^= 1 << (target.val - circuit.num_detectors)
        flags.append(flag)
        syndromes.append(syndrome)
    return diagnostic_circuit, dem, flags, syndromes


def _get_unmeasured_qubits(circuit: stim.Circuit) -> tuple[int, ...]:
    """Qubits that some instruction acts on and no measurement targets, in increasing order."""
    touched: set[int] = set()
    measured: set[int] = set()
    for instruction in circuit.flattened():
        name = instruction.name
        # The targets of MPAD are padding bits, not qubits.
        if name in ("DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "MPAD"):
            continue
        qubits = {tt.qubit_value for tt in instruction.targets_copy() if tt.qubit_value is not None}
        touched |= qubits
        # Heralded noise reports whether an error occurred, and it does not measure the qubit.
        if stim.gate_data(name).produces_measurements and not name.startswith("HERALDED_"):
            measured |= qubits
    return tuple(sorted(touched - measured))


def _validate_output_qubits(output_qubits: Sequence[int], num_qubits: int) -> tuple[int, ...]:
    """Return the output qubits, which must be distinct in-range integers, at least one."""
    try:
        qubits = tuple(output_qubits)
    except TypeError as error:
        raise TypeError("output_qubits must be a sequence of integer qubit indices") from error
    if not all(is_integer(jj) for jj in qubits):
        raise TypeError("output_qubits must contain integer qubit indices")
    if (
        not qubits
        or any(not 0 <= jj < num_qubits for jj in qubits)
        or len(set(qubits)) != len(qubits)
    ):
        raise ValueError("output_qubits must contain distinct in-range qubit indices, at least one")
    return tuple(map(int, qubits))


def _is_css_state(stabilizers: npt.NDArray[np.uint8]) -> bool:
    """Test whether a complete rank-n stabilizer space is CSS in the physical basis."""
    n = len(stabilizers)
    # Let W be the stabilizer space, and W_X and W_Z the X and Z parts of its rows.  The pure-X and
    # pure-Z elements of W have dimensions n-rank(W_Z) and n-rank(W_X) and meet only in 0, so they
    # span W exactly when rank(W_X)+rank(W_Z)=n, whatever generators are used.
    ranks = [
        np.linalg.matrix_rank(galois.GF2(sector))
        for sector in (stabilizers[:, :n], stabilizers[:, n:])
    ]
    return sum(ranks) == n
