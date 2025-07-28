"""Implementation of different noise models for Stim quantum circuits.

This module provides a comprehensive framework for adding noise models
to quantum circuits built with Stim.

The main components are:
- NoiseRule: Defines how to add noise to individual operations
- NoiseModel: Controls the application of noise across entire circuits
- Built-in noise models: si1000 (superconducting) and uniform_depolarizing
- Automatic TICK insertion with idling errors

Examples of basic usage with a predefined noise model:

    import stim
    from qldpc.stim.noise_model import NoiseModel

    # Create a simple circuit
    circuit = stim.Circuit()
    circuit.append('H', [0])
    circuit.append('CX', [0, 1])

    # Apply superconducting-inspired noise
    noise_model = NoiseModel.si1000(0.001)
    noisy_circuit = noise_model.noisy_circuit(circuit)

    # Create a custom noise model
    custom_model = NoiseModel.custom(
         idle_depolarization=0.0001,
         clifford_1q_depolarization=0.001,
         clifford_2q_depolarization=0.01,
         measure_flip_z=0.02
     )
    noisy_circuit = custom_model.noisy_circuit(circuit)


Important note:
---------------

This file has been extended from https://github.com/tqec/tqec/blob/main/src/tqec/utils/noise_model.py
which itself was taken from https://zenodo.org/records/7487893
and is under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/legalcode).
It is part of the code from the paper

    Gidney, C. (2022). Data for "Inplace Access to the Surface Code Y Basis".
    https://doi.org/10.5281/zenodo.7487893
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterator, Set

import stim

CLIFFORD_1Q = "C1"
CLIFFORD_2Q = "C2"
ANNOTATION = "info"
MPP = "MPP"
MEASURE_RESET_1Q = "MR1"
JUST_MEASURE_1Q = "M1"
JUST_RESET_1Q = "R1"
NOISE = "!?"

OP_TYPES = {
    "I": CLIFFORD_1Q,
    "X": CLIFFORD_1Q,
    "Y": CLIFFORD_1Q,
    "Z": CLIFFORD_1Q,
    "C_XYZ": CLIFFORD_1Q,
    "C_ZYX": CLIFFORD_1Q,
    "H": CLIFFORD_1Q,
    "H_XY": CLIFFORD_1Q,
    "H_XZ": CLIFFORD_1Q,
    "H_YZ": CLIFFORD_1Q,
    "S": CLIFFORD_1Q,
    "SQRT_X": CLIFFORD_1Q,
    "SQRT_X_DAG": CLIFFORD_1Q,
    "SQRT_Y": CLIFFORD_1Q,
    "SQRT_Y_DAG": CLIFFORD_1Q,
    "SQRT_Z": CLIFFORD_1Q,
    "SQRT_Z_DAG": CLIFFORD_1Q,
    "S_DAG": CLIFFORD_1Q,
    "CNOT": CLIFFORD_2Q,
    "CX": CLIFFORD_2Q,
    "CY": CLIFFORD_2Q,
    "CZ": CLIFFORD_2Q,
    "ISWAP": CLIFFORD_2Q,
    "ISWAP_DAG": CLIFFORD_2Q,
    "SQRT_XX": CLIFFORD_2Q,
    "SQRT_XX_DAG": CLIFFORD_2Q,
    "SQRT_YY": CLIFFORD_2Q,
    "SQRT_YY_DAG": CLIFFORD_2Q,
    "SQRT_ZZ": CLIFFORD_2Q,
    "SQRT_ZZ_DAG": CLIFFORD_2Q,
    "SWAP": CLIFFORD_2Q,
    "XCX": CLIFFORD_2Q,
    "XCY": CLIFFORD_2Q,
    "XCZ": CLIFFORD_2Q,
    "YCX": CLIFFORD_2Q,
    "YCY": CLIFFORD_2Q,
    "YCZ": CLIFFORD_2Q,
    "ZCX": CLIFFORD_2Q,
    "ZCY": CLIFFORD_2Q,
    "ZCZ": CLIFFORD_2Q,
    "MPP": MPP,
    "MR": MEASURE_RESET_1Q,
    "MRX": MEASURE_RESET_1Q,
    "MRY": MEASURE_RESET_1Q,
    "MRZ": MEASURE_RESET_1Q,
    "M": JUST_MEASURE_1Q,
    "MX": JUST_MEASURE_1Q,
    "MY": JUST_MEASURE_1Q,
    "MZ": JUST_MEASURE_1Q,
    "R": JUST_RESET_1Q,
    "RX": JUST_RESET_1Q,
    "RY": JUST_RESET_1Q,
    "RZ": JUST_RESET_1Q,
    "DETECTOR": ANNOTATION,
    "OBSERVABLE_INCLUDE": ANNOTATION,
    "QUBIT_COORDS": ANNOTATION,
    "SHIFT_COORDS": ANNOTATION,
    "TICK": ANNOTATION,
    "E": ANNOTATION,
    "DEPOLARIZE1": NOISE,
    "DEPOLARIZE2": NOISE,
    "PAULI_CHANNEL_1": NOISE,
    "PAULI_CHANNEL_2": NOISE,
    "X_ERROR": NOISE,
    "Y_ERROR": NOISE,
    "Z_ERROR": NOISE,
    # Not supported.
    # 'CORRELATED_ERROR': NOISE,
    # 'E': NOISE,
    # 'ELSE_CORRELATED_ERROR',
}
OP_MEASURE_BASES = {
    "M": "Z",
    "MX": "X",
    "MY": "Y",
    "MZ": "Z",
    "MPP": "",
}
COLLAPSING_OPS = {
    op
    for op, t in OP_TYPES.items()
    if t == JUST_RESET_1Q or t == JUST_MEASURE_1Q or t == MEASURE_RESET_1Q or t == MPP
}


class NoiseRule:
    """Describes how to add noise to an operation.

    This class encapsulates the noise channels and measurement error probabilities
    that should be applied to a particular type of quantum operation.
    """

    def __init__(self, *, after: dict[str, float] = {}, flip_result: float = 0):
        """Initializes a noise rule with specified error channels.

        Args:
            after: A dictionary mapping noise channel names to their probability
                arguments. For example, {"DEPOLARIZE2": 0.01, "X_ERROR": 0.02}
                will add two-qubit depolarization with parameter 0.01 and also
                add 2% bit flip noise. These noise channels occur after all other
                operations in the moment and are applied to the same targets as
                the relevant operation.
            flip_result: The probability that a measurement result should be
                reported incorrectly. Only valid when applied to operations that
                produce measurement results. Defaults to 0.

        Raises:
            ValueError: If flip_result is not between 0 and 1 (inclusive), or if
                any probability in after is not between 0 and 1 (inclusive), or
                if any key in after is not a valid noise channel name.
        """
        if not (0 <= flip_result <= 1):
            raise ValueError(f"not (0 <= {flip_result=} <= 1)")
        for k, p in after.items():
            if OP_TYPES[k] != NOISE:
                raise ValueError(f"not a noise channel: {k} from {after=}")
            if not (0 <= p <= 1):
                raise ValueError(f"not (0 <= {p} <= 1) from {after=}")
        self.after = after
        self.flip_result = flip_result

    def append_noisy_version_of(
        self,
        *,
        split_op: stim.CircuitInstruction,
        out_during_moment: stim.Circuit,
        after_moments: defaultdict[tuple[str, float], stim.Circuit],
        immune_qubits: Set[int],
    ) -> None:
        """Appends a noisy version of the given operation to the circuit.

        This method applies the noise rule to a quantum operation, adding the
        operation itself along with any associated noise channels to the appropriate
        circuit sections.

        Args:
            split_op: The quantum operation to add noise to.
            out_during_moment: The circuit to append the operation to during the
                current moment.
            after_moments: A dictionary mapping noise channel types and parameters
                to circuits that should be executed after the main operations.
            immune_qubits: A set of qubit indices that should not have noise
                applied to them.
        """
        targets = split_op.targets_copy()
        if immune_qubits and any(
            (t.is_qubit_target or t.is_x_target or t.is_y_target or t.is_z_target)
            and t.value in immune_qubits
            for t in targets
        ):
            out_during_moment.append(split_op)
            return

        args = split_op.gate_args_copy()
        if self.flip_result:
            t = OP_TYPES[split_op.name]
            assert t == MPP or t == JUST_MEASURE_1Q or t == MEASURE_RESET_1Q
            assert len(args) == 0
            args = [self.flip_result]

        out_during_moment.append(split_op.name, targets, args)
        raw_targets = [t.value for t in targets if not t.is_combiner]
        for op_name, arg in self.after.items():
            after_moments[(op_name, arg)].append(op_name, raw_targets, arg)


class NoiseModel:
    """A model that defines how to add noise to quantum circuits.

    This class provides a framework for adding various types of
    noise to quantum circuits, including idle depolarization, gate errors,
    and measurement errors. It supports both built-in noise models and
    custom user-defined noise configurations.

    Attributes:
        idle_depolarization: Probability of depolarization for idle qubits.
        additional_depolarization_waiting_for_m_or_r: Additional depolarization
            probability for qubits waiting during measurement or reset operations.
        gate_rules: Dictionary mapping gate names to their noise rules.
        measure_rules: Dictionary mapping measurement bases to their noise rules.
        any_clifford_1q_rule: Default noise rule for all single-qubit Clifford gates.
        any_clifford_2q_rule: Default noise rule for all two-qubit Clifford gates.
    """

    def __init__(
        self,
        idle_depolarization: float,
        additional_depolarization_waiting_for_m_or_r: float = 0,
        gate_rules: dict[str, NoiseRule] | None = None,
        measure_rules: dict[str, NoiseRule] | None = None,
        any_clifford_1q_rule: NoiseRule | None = None,
        any_clifford_2q_rule: NoiseRule | None = None,
    ):
        """Initializes a noise model with specified parameters.

        Args:
            idle_depolarization: Probability of depolarization for qubits that
                are not being operated on during a moment.
            additional_depolarization_waiting_for_m_or_r: Additional depolarization
                probability applied to qubits that are waiting while other qubits
                undergo measurement or reset operations.
            gate_rules: Optional dictionary mapping specific gate names to their
                noise rules. If provided, these take precedence over the default
                Clifford rules.
            measure_rules: Optional dictionary mapping measurement basis strings
                (e.g., "X", "Y", "Z", "XX", "YY", "ZZ") to their noise rules.
            any_clifford_1q_rule: Optional default noise rule to apply to all
                single-qubit Clifford gates that don't have specific rules.
            any_clifford_2q_rule: Optional default noise rule to apply to all
                two-qubit Clifford gates that don't have specific rules.
        """
        self.idle_depolarization = idle_depolarization
        self.additional_depolarization_waiting_for_m_or_r = (
            additional_depolarization_waiting_for_m_or_r
        )
        self.gate_rules = gate_rules
        self.measure_rules = measure_rules
        self.any_clifford_1q_rule = any_clifford_1q_rule
        self.any_clifford_2q_rule = any_clifford_2q_rule

    @staticmethod
    def custom(
        idle_depolarization: float = 0.0,
        additional_depolarization_waiting_for_m_or_r: float = 0.0,
        clifford_1q_depolarization: float = 0.0,
        clifford_2q_depolarization: float = 0.0,
        measure_flip_x: float = 0.0,
        measure_flip_y: float = 0.0,
        measure_flip_z: float = 0.0,
        measure_flip_xx: float = 0.0,
        measure_flip_yy: float = 0.0,
        measure_flip_zz: float = 0.0,
        reset_x: float = 0.0,
        reset_y: float = 0.0,
        reset_z: float = 0.0,
    ) -> NoiseModel:
        """Creates a custom noise model with user-specified error rates.

        This method allows fine-grained control over all types of noise that can
        be applied to quantum circuits, including idle errors, gate errors,
        measurement errors, and reset errors.

        Args:
            idle_depolarization: Probability of depolarization for qubits that
                are idle (not being operated on) during a moment. Defaults to 0.0.
            additional_depolarization_waiting_for_m_or_r: Additional depolarization
                probability for qubits waiting while other qubits undergo
                measurement or reset operations. Defaults to 0.0.
            clifford_1q_depolarization: Depolarization probability for single-qubit
                Clifford gates. Defaults to 0.0.
            clifford_2q_depolarization: Depolarization probability for two-qubit
                Clifford gates. Defaults to 0.0.
            measure_flip_x: Probability of measurement result bit flip for X-basis
                measurements. Defaults to 0.0.
            measure_flip_y: Probability of measurement result bit flip for Y-basis
                measurements. Defaults to 0.0.
            measure_flip_z: Probability of measurement result bit flip for Z-basis
                measurements. Defaults to 0.0.
            measure_flip_xx: Probability of measurement result bit flip for XX-basis
                joint measurements. Defaults to 0.0.
            measure_flip_yy: Probability of measurement result bit flip for YY-basis
                joint measurements. Defaults to 0.0.
            measure_flip_zz: Probability of measurement result bit flip for ZZ-basis
                joint measurements. Defaults to 0.0.
            reset_x: Probability of bit flip error after RX (reset to |+⟩) operations.
                Defaults to 0.0.
            reset_y: Probability of bit flip error after RY (reset to |i⟩) operations.
                Defaults to 0.0.
            reset_z: Probability of bit flip error after R/RZ (reset to |0⟩) operations.
                Defaults to 0.0.

        Returns:
            A NoiseModel instance configured with the specified error rates.
        """
        return NoiseModel(
            idle_depolarization=idle_depolarization,
            additional_depolarization_waiting_for_m_or_r=additional_depolarization_waiting_for_m_or_r,
            any_clifford_1q_rule=(
                NoiseRule(after={"DEPOLARIZE1": clifford_1q_depolarization})
                if clifford_1q_depolarization
                else None
            ),
            any_clifford_2q_rule=(
                NoiseRule(after={"DEPOLARIZE2": clifford_2q_depolarization})
                if clifford_2q_depolarization
                else None
            ),
            measure_rules={
                "X": NoiseRule(flip_result=measure_flip_x),
                "Y": NoiseRule(flip_result=measure_flip_y),
                "Z": NoiseRule(flip_result=measure_flip_z),
                "XX": NoiseRule(flip_result=measure_flip_xx),
                "YY": NoiseRule(flip_result=measure_flip_yy),
                "ZZ": NoiseRule(flip_result=measure_flip_zz),
            },
            gate_rules={
                "RX": (NoiseRule(after={"Z_ERROR": reset_x}) if reset_x else NoiseRule()),
                "RY": (NoiseRule(after={"X_ERROR": reset_y}) if reset_y else NoiseRule()),
                "R": (NoiseRule(after={"X_ERROR": reset_z}) if reset_z else NoiseRule()),
            },
        )

    @staticmethod
    def si1000(p: float) -> NoiseModel:
        """Creates a superconducting-inspired noise model.

        As defined in "A Fault-Tolerant Honeycomb Memory":
        https://arxiv.org/abs/2108.10457

        Note: Small tweak from the paper - the measurement result is
        probabilistically flipped instead of the input qubit.

        Args:
            p: The base error probability parameter.

        Returns:
            A NoiseModel instance configured with superconducting-inspired
            error rates.
        """
        return NoiseModel.custom(
            idle_depolarization=p / 10,
            additional_depolarization_waiting_for_m_or_r=2 * p,
            clifford_1q_depolarization=p / 10,
            clifford_2q_depolarization=p,
            measure_flip_z=p * 5,
            reset_z=p * 2,
        )

    @staticmethod
    def uniform_depolarizing(p: float, idling_error: bool = True) -> NoiseModel:
        """Creates a near-standard circuit depolarizing noise model.

        Everything has the same parameter p. Single qubit clifford gates
        get single qubit depolarization. Two qubit clifford gates get
        single qubit depolarization. Dissipative gates have their result
        probabilistically bit flipped (or phase flipped if appropriate).

        Non-demolition measurement is treated a bit unusually in that it
        is the result that is flipped instead of the input qubit.

        Args:
            p: Depolarizing noise parameter.
            idling_error: If False, disables idle depolarization noise.

        Returns:
            A NoiseModel instance configured with uniform depolarizing noise.
        """
        return NoiseModel.custom(
            idle_depolarization=p if idling_error else 0.0,
            clifford_1q_depolarization=p,
            clifford_2q_depolarization=p,
            measure_flip_x=p,
            measure_flip_y=p,
            measure_flip_z=p,
            measure_flip_xx=p,
            measure_flip_yy=p,
            measure_flip_zz=p,
            reset_x=p,
            reset_y=p,
            reset_z=p,
        )

    def _noise_rule_for_split_operation(
        self, *, split_op: stim.CircuitInstruction
    ) -> NoiseRule | None:
        """Determines the noise rule to apply to a specific operation.

        Args:
            split_op: The circuit instruction to find a noise rule for.

        Returns:
            The NoiseRule to apply to this operation, or None if the operation
            occurs in the classical control system and should not have noise
            applied.

        Raises:
            ValueError: If no noise rule is specified for the given operation.
        """
        if occurs_in_classical_control_system(split_op):
            return None

        if self.gate_rules is not None:
            rule = self.gate_rules.get(split_op.name)
            if rule is not None:
                return rule

        t = OP_TYPES[split_op.name]

        if self.any_clifford_1q_rule is not None and t == CLIFFORD_1Q:
            return self.any_clifford_1q_rule
        if self.any_clifford_2q_rule is not None and t == CLIFFORD_2Q:
            return self.any_clifford_2q_rule
        if self.measure_rules is not None:
            measure_basis = _measure_basis(split_op=split_op)
            assert measure_basis is not None
            rule = self.measure_rules.get(measure_basis)
            if rule is not None:
                return rule

        raise ValueError(f"No noise (or lack of noise) specified for {split_op=}.")

    def _append_idle_error(
        self,
        *,
        moment_split_ops: list[stim.CircuitInstruction],
        out: stim.Circuit,
        system_qubits: Set[int],
        immune_qubits: Set[int],
    ) -> None:
        """Appends idle errors to the circuit for qubits not being operated on.

        This method identifies which qubits are idle during a moment and applies
        depolarization noise to them according to the noise model parameters.

        Args:
            moment_split_ops: List of operations happening during this moment.
            out: The circuit to append idle error operations to.
            system_qubits: Set of all qubits in the system that can experience
                idle errors.
            immune_qubits: Set of qubit indices that should not have noise
                applied to them.

        Raises:
            ValueError: If qubits are operated on multiple times within the
                same moment without a TICK in between.
        """
        collapse_qubits: list[int] = []
        clifford_qubits: list[int] = []
        for split_op in moment_split_ops:
            if occurs_in_classical_control_system(split_op):
                continue
            if split_op.name in COLLAPSING_OPS:
                qubits_out = collapse_qubits
            else:
                qubits_out = clifford_qubits
            for target in split_op.targets_copy():
                if not target.is_combiner:
                    qubits_out.append(target.value)

        # Safety check for operation collisions.
        usage_counts = Counter(collapse_qubits + clifford_qubits)
        qubits_used_multiple_times = {q for q, c in usage_counts.items() if c != 1}
        if qubits_used_multiple_times:
            moment = stim.Circuit()
            for op in moment_split_ops:
                moment.append(op)
            raise ValueError(
                f"Qubits were operated on multiple times without a TICK in between:\n"
                f"multiple uses: {sorted(qubits_used_multiple_times)}\n"
                f"moment:\n"
                f"{moment}"
            )

        collapse_qubits_set = set(collapse_qubits)
        clifford_qubits_set = set(clifford_qubits)
        idle = sorted(system_qubits - collapse_qubits_set - clifford_qubits_set - immune_qubits)
        if idle and self.idle_depolarization:
            out.append("DEPOLARIZE1", idle, self.idle_depolarization)

        waiting_for_mr = sorted(system_qubits - collapse_qubits_set - immune_qubits)
        if (
            collapse_qubits_set
            and waiting_for_mr
            and self.additional_depolarization_waiting_for_m_or_r
        ):
            out.append("DEPOLARIZE1", idle, self.additional_depolarization_waiting_for_m_or_r)

    def _append_noisy_moment(
        self,
        *,
        moment_split_ops: list[stim.CircuitInstruction],
        out: stim.Circuit,
        system_qubits: Set[int],
        immune_qubits: Set[int],
    ) -> None:
        """Appends a noisy version of a moment to the output circuit.

        This method processes all operations in a moment, applies their respective
        noise rules, and adds the resulting noisy operations to the output circuit.

        Args:
            moment_split_ops: List of operations happening during this moment.
            out: The circuit to append the noisy operations to.
            system_qubits: Set of all qubits in the system that can experience
                idle errors.
            immune_qubits: Set of qubit indices that should not have noise
                applied to them.
        """
        after: defaultdict[tuple[str, float], stim.Circuit] = defaultdict(stim.Circuit)
        for split_op in moment_split_ops:
            rule = self._noise_rule_for_split_operation(split_op=split_op)
            if rule is None:
                out.append(split_op)
            else:
                rule.append_noisy_version_of(
                    split_op=split_op,
                    out_during_moment=out,
                    after_moments=after,
                    immune_qubits=immune_qubits,
                )
        for k in sorted(after.keys()):
            out += after[k]

        self._append_idle_error(
            moment_split_ops=moment_split_ops,
            out=out,
            system_qubits=system_qubits,
            immune_qubits=immune_qubits,
        )

    def _preprocess_circuit_with_auto_ticks(self, circuit: stim.Circuit) -> stim.Circuit:
        """Preprocesses a circuit to automatically insert TICKs when qubits are reused.

        This ensures that no qubit is operated on multiple times within a single moment,
        which allows the existing idling error logic to work correctly.

        Args:
            circuit: The input circuit to preprocess.

        Returns:
            A new circuit with TICKs automatically inserted to prevent qubit reuse
            within the same moment.
        """
        result = stim.Circuit()
        used_qubits: set[int] = set()

        for op in circuit:
            if isinstance(op, stim.CircuitRepeatBlock):
                # Process repeat blocks recursively
                if used_qubits:
                    result.append("TICK")
                    used_qubits.clear()
                processed_body = self._preprocess_circuit_with_auto_ticks(op.body_copy())
                result.append(
                    stim.CircuitRepeatBlock(repeat_count=op.repeat_count, body=processed_body)
                )
                continue

            if op.name == "TICK":
                # Explicit TICK - clear used qubits
                result.append(op)
                used_qubits.clear()
                continue

            # For preprocessing, we need to force splitting of multi-target operations
            # to detect qubit reuse properly. Use a dummy immune_qubits set with -1
            # to force splitting of 2-qubit operations
            split_ops = list(_split_targets_if_needed(op, immune_qubits={-1}))

            for split_op in split_ops:
                # Check if this split operation would reuse any qubits
                op_qubits = set()
                if not occurs_in_classical_control_system(split_op):
                    for target in split_op.targets_copy():
                        if not target.is_combiner:
                            op_qubits.add(target.value)

                # If there's qubit reuse, insert a TICK first
                if op_qubits & used_qubits:
                    result.append("TICK")
                    used_qubits.clear()

                # Add the operation and update used qubits
                result.append(split_op)
                used_qubits.update(op_qubits)

        return result

    def noisy_circuit(
        self,
        circuit: stim.Circuit,
        *,
        system_qubits: set[int] | None = None,
        immune_qubits: set[int] | None = None,
        automatic_ticks: bool = True,
    ) -> stim.Circuit:
        """Returns a noisy version of the given circuit.

        This method applies the noise model to transform a clean quantum circuit
        into one that includes realistic noise effects. The circuit is first
        preprocessed to automatically insert TICKs when needed to prevent qubit
        reuse conflicts, then noise is applied according to the configured rules.

        Args:
            circuit: The circuit to layer noise over.
            system_qubits: All qubits used by the circuit. These are the qubits
                eligible for idling noise. If None, defaults to all qubits from
                0 to circuit.num_qubits-1.
            immune_qubits: Qubits to not apply noise to, even if they are
                operated on. If None, defaults to an empty set.
            automatic_ticks: If True, automatically inserts TICK operations
                to prevent qubit reuse conflicts. If False, assumes the circuit
                is already preprocessed and does not insert TICKs.

        Returns:
            The noisy version of the circuit with all specified noise channels
            applied.
        """
        if system_qubits is None:
            system_qubits = set(range(circuit.num_qubits))
        if immune_qubits is None:
            immune_qubits = set()

        # Preprocess the circuit to automatically insert TICKs when qubits are reused
        if automatic_ticks:
            if immune_qubits:
                raise ValueError("Automatic TICK insertion does not support immune qubits.")
            circuit = self._preprocess_circuit_with_auto_ticks(circuit)

        result = stim.Circuit()

        first = True
        for moment_split_ops in _iter_split_op_moments(circuit, immune_qubits=immune_qubits):
            if first:
                first = False
            elif result and isinstance(result[-1], stim.CircuitRepeatBlock):
                pass
            else:
                result.append("TICK")
            if isinstance(moment_split_ops, stim.CircuitRepeatBlock):
                noisy_body = self.noisy_circuit(
                    moment_split_ops.body_copy(),
                    system_qubits=system_qubits,
                    immune_qubits=immune_qubits,
                )
                noisy_body.append("TICK")
                result.append(
                    stim.CircuitRepeatBlock(
                        repeat_count=moment_split_ops.repeat_count, body=noisy_body
                    )
                )
            else:
                self._append_noisy_moment(
                    moment_split_ops=moment_split_ops,
                    out=result,
                    system_qubits=system_qubits,
                    immune_qubits=immune_qubits,
                )

        return result


def occurs_in_classical_control_system(op: stim.CircuitInstruction) -> bool:
    """Determines if an operation is an annotation or a classical control system update.

    Args:
        op: The circuit instruction to check.

    Returns:
        True if the operation occurs in the classical control system and should
        not have quantum noise applied to it, False otherwise.
    """
    t = OP_TYPES[op.name]
    if t == ANNOTATION:
        return True
    if t == CLIFFORD_2Q:
        targets = op.targets_copy()
        for k in range(0, len(targets), 2):
            a = targets[k]
            b = targets[k + 1]
            classical_0 = a.is_measurement_record_target or a.is_sweep_bit_target
            classical_1 = b.is_measurement_record_target or b.is_sweep_bit_target
            if not (classical_0 or classical_1):
                return False
        return True
    return False


def _split_targets_if_needed(
    op: stim.CircuitInstruction, immune_qubits: Set[int]
) -> Iterator[stim.CircuitInstruction]:
    """Splits operations into pieces as needed.

    This function splits operations like MPP into each product, and separates
    classical control operations from quantum operations.

    Args:
        op: The circuit instruction to potentially split.
        immune_qubits: Set of qubit indices that should not have noise applied.

    Yields:
        Circuit instructions, potentially split into smaller pieces.
    """
    t = OP_TYPES[op.name]
    if t == CLIFFORD_2Q:
        yield from _split_targets_if_needed_clifford_2q(op, immune_qubits)
    elif t == MPP:
        yield from _split_targets_if_needed_m_basis(op, immune_qubits)
    elif t in [NOISE, ANNOTATION]:
        yield op
    else:
        yield from _split_targets_if_needed_clifford_1q(op, immune_qubits)


def _split_targets_if_needed_clifford_1q(
    op: stim.CircuitInstruction, immune_qubits: Set[int]
) -> Iterator[stim.CircuitInstruction]:
    """Splits single-qubit Clifford operations when immune qubits are present.

    Args:
        op: The single-qubit Clifford operation to potentially split.
        immune_qubits: Set of qubit indices that should not have noise applied.

    Yields:
        Circuit instructions, either the original operation or split into
        individual single-target operations.
    """
    if immune_qubits:
        args = op.gate_args_copy()
        for t in op.targets_copy():
            yield stim.CircuitInstruction(op.name, [t], args)
    else:
        yield op


def _split_targets_if_needed_clifford_2q(
    op: stim.CircuitInstruction, immune_qubits: Set[int]
) -> Iterator[stim.CircuitInstruction]:
    """Splits two-qubit Clifford operations into individual gate pairs.

    This function separates classical control system operations from quantum
    operations happening on the quantum computer.

    Args:
        op: The two-qubit Clifford operation to potentially split.
        immune_qubits: Set of qubit indices that should not have noise applied.

    Yields:
        Circuit instructions, either the original operation or split into
        individual two-qubit gate operations.
    """
    assert OP_TYPES[op.name] == CLIFFORD_2Q
    targets = op.targets_copy()
    if immune_qubits or any(t.is_measurement_record_target for t in targets):
        args = op.gate_args_copy()
        for k in range(0, len(targets), 2):
            yield stim.CircuitInstruction(op.name, targets[k : k + 2], args)
    else:
        yield op


def _split_targets_if_needed_m_basis(
    op: stim.CircuitInstruction, immune_qubits: Set[int]
) -> Iterator[stim.CircuitInstruction]:
    """Splits an MPP operation into one operation for each Pauli product it measures.

    Args:
        op: The MPP operation to split.
        immune_qubits: Set of qubit indices that should not have noise applied
            (unused in this function but included for interface consistency).

    Yields:
        Circuit instructions, one for each Pauli product measurement.
    """
    targets = op.targets_copy()
    args = op.gate_args_copy()
    k = 0
    start = k
    while k < len(targets):
        if k + 1 == len(targets) or not targets[k + 1].is_combiner:
            yield stim.CircuitInstruction(op.name, targets[start : k + 1], args)
            k += 1
            start = k
        else:
            k += 2
    assert k == len(targets)


def _iter_split_op_moments(
    circuit: stim.Circuit, *, immune_qubits: Set[int]
) -> Iterator[stim.CircuitRepeatBlock | list[stim.CircuitInstruction]]:
    """Splits a circuit into moments and some operations into pieces.

    Classical control system operations like CX rec[-1] 0 are split from quantum
    operations like CX 1 0. MPP operations are split into one operation per
    Pauli product.

    Args:
        circuit: The circuit to split into moments.
        immune_qubits: Set of qubit indices that should not have noise applied.

    Yields:
        Lists of operations corresponding to one moment in the circuit, with any
        problematic operations like MPPs split into pieces, or CircuitRepeatBlock
        instances for repeat blocks.

    Note:
        A moment is the time between two TICKs.
    """
    cur_moment: list[stim.CircuitInstruction] = []

    for op in circuit:
        if isinstance(op, stim.CircuitRepeatBlock):
            if cur_moment:
                yield cur_moment
                cur_moment = []
            yield op
        elif op.name == "TICK":
            yield cur_moment
            cur_moment = []
        else:
            cur_moment.extend(_split_targets_if_needed(op, immune_qubits=immune_qubits))
    if cur_moment:
        yield cur_moment


def _measure_basis(*, split_op: stim.CircuitInstruction) -> str | None:
    """Converts an operation into a string describing the Pauli product basis it measures.

    This function determines what basis a measurement operation measures in, which
    is used to determine the appropriate noise rules to apply.

    Args:
        split_op: The circuit instruction to analyze.

    Returns:
        None if this is not a measurement operation (or not exclusively a measurement).
        str: Pauli product string that the operation measures (e.g. "XX" or "Y").

    Examples:
        - MZ operation returns "Z"
        - MX operation returns "X"
        - MPP X0*Y1 operation returns "XY"

    Raises:
        NotImplementedError: If the operation contains target types that are not
            supported.
    """
    result = OP_MEASURE_BASES.get(split_op.name)
    targets = split_op.targets_copy()
    if result == "":
        for k in range(0, len(targets), 2):
            t = targets[k]
            if t.is_x_target:
                result += "X"
            elif t.is_y_target:
                result += "Y"
            elif t.is_z_target:
                result += "Z"
            else:
                raise NotImplementedError(f"{targets=}")
    return result
