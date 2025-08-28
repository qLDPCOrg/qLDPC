"""Implementation of noise models for Stim circuits

The main components of this module are:
- NoiseRule: Defines how to add noise to individual operations.
- NoiseModel: Defines how noise is added to circuits.
- Built-in noise models: DepolarizingNoiseModel and the superconducting-inspired SI1000NoiseModel.

Examples of basic usage with a predefined noise model:

    import stim
    from qldpc.circuits.noise_model import DepolarizingNoiseModel, NoiseModel, SI1000NoiseModel

    # Create a simple circuit
    circuit = stim.Circuit("H 0 \n CX 0 1")

    # Apply simple depolarizing noise
    noise_model = DepolarizingNoiseModel(0.001)
    noisy_circuit = noise_model.noisy_circuit(circuit)

    # Apply superconducting-inspired noise
    noise_model = SI1000NoiseModel(0.001)
    noisy_circuit = noise_model.noisy_circuit(circuit)

    # Create a custom noise model
    custom_model = NoiseModel(
        clifford_1q_error=3e-5,
        clifford_2q_error=1e-3,
        readout_error=1e-3,
        reset_error=1e-3,
        idle_error=2e-4,
    )
    noisy_circuit = custom_model.noisy_circuit(circuit)


Important note:
---------------

This file was taken and modified from
    https://github.com/tqec/tqec/blob/main/src/tqec/utils/noise_model.py
which itself was taken from
    https://zenodo.org/records/7487893
and licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/legalcode).

The original code was written for the paper at "Inplace Access to the Surface Code Y Basis", at
    https://quantum-journal.org/papers/q-2024-04-08-1310.
"""

from __future__ import annotations

import collections
from collections.abc import Collection, Iterable, Iterator

import stim

CLIFFORD_1Q = "C1"
CLIFFORD_2Q = "C2"
JUST_MEASURE_1Q = "M1"
JUST_MEASURE_2Q = "M2"
JUST_MEASURE_PP = "MPP"
JUST_RESET_1Q = "R1"
MEASURE_RESET_1Q = "MR1"
ANNOTATION = "info"
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
    "M": JUST_MEASURE_1Q,
    "MX": JUST_MEASURE_1Q,
    "MY": JUST_MEASURE_1Q,
    "MZ": JUST_MEASURE_1Q,
    "MXX": JUST_MEASURE_2Q,
    "MYY": JUST_MEASURE_2Q,
    "MZZ": JUST_MEASURE_2Q,
    "MPP": JUST_MEASURE_PP,
    "R": JUST_RESET_1Q,
    "RX": JUST_RESET_1Q,
    "RY": JUST_RESET_1Q,
    "RZ": JUST_RESET_1Q,
    "MR": MEASURE_RESET_1Q,
    "MRX": MEASURE_RESET_1Q,
    "MRY": MEASURE_RESET_1Q,
    "MRZ": MEASURE_RESET_1Q,
    "DETECTOR": ANNOTATION,
    "OBSERVABLE_INCLUDE": ANNOTATION,
    "QUBIT_COORDS": ANNOTATION,
    "SHIFT_COORDS": ANNOTATION,
    "TICK": ANNOTATION,
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
    # 'ELSE_CORRELATED_ERROR': NOISE,
}
JUST_MEASURE_OPS = {
    op
    for op, op_type in OP_TYPES.items()
    if op_type == JUST_MEASURE_1Q or op_type == JUST_MEASURE_2Q or op_type == JUST_MEASURE_PP
}
JUST_RESET_OPS = {op for op, op_type in OP_TYPES.items() if op_type == JUST_RESET_1Q}
MEASURE_AND_RESET_OPS = {op for op, op_type in OP_TYPES.items() if op_type == MEASURE_RESET_1Q}
COLLAPSING_OPS = JUST_MEASURE_OPS | JUST_RESET_OPS | MEASURE_AND_RESET_OPS


DEFAULT_IMMUNE_OP_TAG = "__IMMUNE_TO_NOISE__"


class NoiseRule:
    """Describes how to add noise to an operation.

    This class encapsulates the noise channels and measurement error probabilities that should be
    applied to a particular type of quantum operation.
    """

    def __init__(
        self,
        *,
        after: dict[str, float | Iterable[float]] = {},
        readout_error: float = 0,
        reset_error: float = 0,
    ):
        """Initializes a noise rule with specified error channels.

        Args:
            after: A dictionary mapping noise channel names to their probability arguments.  For
                example, {"DEPOLARIZE2": 0.01, "PAULI_CHANNEL_1": [0.02, 0, 0]} will add two-qubit
                depolarization with parameter 0.01, followed by 2% bit-flip noise.  These noise
                channels occur after all other operations in the moment and are applied to the same
                targets as the relevant operation.
            readout_error: The probability that a measurement result is reported incorrectly.  Only
                allowed for operations that produce measurement results.
            reset_error: The probability that a qubit is reset to the wrong state.  Only allowed for
                operations that reset qubits.

        Raises:
            ValueError: If any noise channel name is not recognized or if any net probability of an
                error is not between 0 and 1 (inclusive).
        """
        self.readout_error = readout_error
        if not (0 <= readout_error <= 1):
            raise ValueError(f"{readout_error=} is not between 0 and 1")

        self.reset_error = reset_error
        if not (0 <= reset_error <= 1):
            raise ValueError(f"{reset_error=} is not between 0 and 1")

        self.after = {
            op: tuple(prob_or_probs) if isinstance(prob_or_probs, Iterable) else (prob_or_probs,)
            for op, prob_or_probs in after.items()
        }
        for op, probs in self.after.items():
            if OP_TYPES[op] != NOISE:
                raise ValueError(f"Invalid or unrecognized noise channel {op} in {after=}")
            if not (0 <= sum(probs) <= 1):
                raise ValueError(
                    f"The net probability of an error is not between 0 and 1 in {after=}"
                )

    def __bool__(self) -> bool:
        """Is this noise rule nontrivial?"""
        return bool(self.after) or bool(self.readout_error) or bool(self.reset_error)

    def noisy_operation(
        self, op: stim.CircuitInstruction, *, immune_qubits: set[int] = set()
    ) -> tuple[stim.CircuitInstruction, stim.Circuit]:
        """Apply this noise rule to the given operation.

        Args:
            op: The operation to add noise to.

        Returns:
            stim.CircuitInstruction: The given operation possibly modified to account for noise.
            stim.Circuit: Noise operations that should follow the given operation.
        """
        targets = op.targets_copy()
        if immune_qubits and any(
            (
                target.is_qubit_target
                or target.is_x_target
                or target.is_y_target
                or target.is_z_target
            )
            and target.value in immune_qubits
            for target in targets
        ):
            return op, stim.Circuit()

        args = op.gate_args_copy()
        if self.readout_error:
            assert op.name in JUST_MEASURE_OPS or op.name in MEASURE_AND_RESET_OPS
            if not args:
                args = [self.readout_error]
            else:
                assert len(args) == 1
                # combine bit-flip probabilities
                args = [1 - (1 - self.readout_error) * (1 - args[0])]

        noisy_op = stim.CircuitInstruction(op.name, targets, args, tag=op.tag)
        noise_after = stim.Circuit()

        qubit_targets = [target.value for target in targets if not target.is_combiner]
        if self.reset_error:
            assert op.name in JUST_RESET_OPS or op.name in MEASURE_AND_RESET_OPS
            error_name = ("X" if _get_standardized_name(op)[-1] != "X" else "Z") + "_ERROR"
            error_op = stim.CircuitInstruction(error_name, qubit_targets, [self.reset_error])
            noise_after.append(error_op)

        for op_name, args in self.after.items():
            error_op = stim.CircuitInstruction(op_name, qubit_targets, args)
            noise_after.append(error_op)

        return noisy_op, noise_after


class NoiseModel:
    """A model that defines how to add noise to quantum circuits.

    This class provides a framework for adding various types of noise to quantum circuits, including
    gate errors, readout errors, reset errors, and idling errors.  Classically controlled operations
    are assumed to NOT occur, so the corresponding qubits pick up idling errors, if applicable.
    """

    def __init__(
        self,
        clifford_1q_error: NoiseRule | float | None = None,
        clifford_2q_error: NoiseRule | float | None = None,
        readout_error: float | None = None,
        reset_error: float | None = None,
        *,
        idle_error: float | None = None,
        additional_error_waiting_for_m_or_r: float | None = None,
        rules: dict[str, NoiseRule] | None = None,
    ):
        """Initializes a noise model with specified parameters.

        Args:
            clifford_1q_error: Default noise rule or depolarization probability for one-qubit unitary
                Clifford gates.
            clifford_2q_error: Default noise rule or depolarization probability for two-qubit unitary
                Clifford gates.
            readout_error: Default probability of flipping measurement results.
            reset_error: Default probability of resetting qubits to the wrong state.
            idle_error: Probability of depolarization for each idling qubit in any given moment.
            additional_error_waiting_for_m_or_r: Additional depolarization probability applied to
                qubits that are waiting while other qubits undergo measurement or reset operations.
            rules: Dictionary mapping specific gate names to their noise rules.  Overrides all other
                rules for unitary, measurement, and reset gates.
        """
        if not (isinstance(clifford_1q_error, NoiseRule) or clifford_1q_error is None):
            clifford_1q_error = NoiseRule(after={"DEPOLARIZE1": clifford_1q_error})
        if not (isinstance(clifford_2q_error, NoiseRule) or clifford_2q_error is None):
            clifford_2q_error = NoiseRule(after={"DEPOLARIZE2": clifford_2q_error})

        self.rules = rules
        self.clifford_1q_error = clifford_1q_error
        self.clifford_2q_error = clifford_2q_error
        self.readout_error = readout_error or 0
        self.reset_error = reset_error or 0
        self.idle_error = idle_error
        self.additional_error_waiting_for_m_or_r = additional_error_waiting_for_m_or_r

    def __bool__(self) -> bool:
        """Is this noise model nontrivial?"""
        return (
            bool(self.rules)
            or bool(self.clifford_1q_error)
            or bool(self.clifford_2q_error)
            or bool(self.readout_error)
            or bool(self.reset_error)
            or bool(self.idle_error)
            or bool(self.additional_error_waiting_for_m_or_r)
        )

    def get_noise_rule(self, op: stim.CircuitInstruction) -> NoiseRule | None:
        """Determines the noise rule to apply to a specific operation.

        Args:
            op: The circuit instruction to find a noise rule for.

        Returns:
            The NoiseRule to apply for the given operation, or None for no noise.
        """
        if OP_TYPES[op.name] == ANNOTATION or _involves_classical_bits(op):
            return None

        if self.rules is not None:
            rule = self.rules.get(_get_standardized_name(op)) or self.rules.get(
                op.name
            )  # allows for an MPP rule, but first checks for rules such as MXY
            if rule is not None:
                return rule

        op_type = OP_TYPES[op.name]
        if self.clifford_1q_error is not None and op_type == CLIFFORD_1Q:
            return self.clifford_1q_error
        if self.clifford_2q_error is not None and op_type == CLIFFORD_2Q:
            return self.clifford_2q_error

        if self.readout_error and op.name in JUST_MEASURE_OPS:
            return NoiseRule(readout_error=self.readout_error)
        if self.reset_error and op.name in JUST_RESET_OPS:
            return NoiseRule(reset_error=self.reset_error)
        if (self.readout_error or self.reset_error) and op.name in MEASURE_AND_RESET_OPS:
            return NoiseRule(readout_error=self.readout_error, reset_error=self.reset_error)

        return None

    def noisy_circuit(
        self,
        circuit: stim.Circuit,
        *,
        system_qubits: Collection[int] | None = None,
        immune_qubits: Collection[int] | None = None,
        immune_op_tag: str = DEFAULT_IMMUNE_OP_TAG,
        insert_ticks: bool = True,
    ) -> stim.Circuit:
        f"""Construct a noisy version of the given circuit.

        This method first uses TICKs to split the input circuit into moments of operations that can
        be applied in parallel, thereby preventing qubit reuse conflicts.  Noise is then applied to
        each operation according to the rules of this NoiseModel.

        Args:
            circuit: The circuit to apply noise to.
            system_qubits: All qubits that are used by the circuit or are otherwise allowed to
                accumulate idling errors.  Defaults to set(range(circuit.num_qubits)).
            immune_qubits: All qubits that are declared immune to noise, even if they are operated
                on.  If None, defaults to the empty set.
            immune_op_tag: If an operation contains this string in its tag, that operation is
                noiseless.  Default: "{DEFAULT_IMMUNE_OP_TAG}".
            insert_ticks: If True, automatically inserts TICK operations to prevent qubit reuse
                conflicts.  If False, assumes that this preprocessing is not necessary.

        Returns:
            stim.Circuit: A noisy version of the input circuit.
        """
        system_qubits = set(system_qubits or range(circuit.num_qubits))
        immune_qubits = set(immune_qubits or [])

        if insert_ticks:
            # split moments with TICKs to prevent qubit reuse conflicts
            if immune_qubits:
                raise ValueError("Automatic TICK insertion does not support immune qubits.")
            circuit = _split_moments_with_ticks(circuit)

        noisy_circuit = stim.Circuit()

        first_moment = True
        for moment_or_repeat_block in _iter_moments_and_repeat_blocks(circuit, immune_qubits):
            if first_moment:
                first_moment = False
            elif not isinstance(noisy_circuit[-1], stim.CircuitRepeatBlock):
                noisy_circuit.append("TICK")

            if isinstance(moment_or_repeat_block, stim.CircuitRepeatBlock):
                if immune_op_tag in moment_or_repeat_block.tag:
                    noisy_circuit.append(moment_or_repeat_block)
                else:
                    noisy_body = self.noisy_circuit(
                        moment_or_repeat_block.body_copy(),
                        system_qubits=system_qubits,
                        immune_qubits=immune_qubits,
                    )
                    noisy_body.append("TICK")
                    noisy_circuit.append(
                        stim.CircuitRepeatBlock(
                            repeat_count=moment_or_repeat_block.repeat_count,
                            body=noisy_body,
                            tag=moment_or_repeat_block.tag,
                        )
                    )
            else:
                self._inplace_append_noisy_moment(
                    circuit=noisy_circuit,
                    moment=moment_or_repeat_block,
                    system_qubits=system_qubits,
                    immune_qubits=immune_qubits,
                    immune_op_tag=immune_op_tag,
                )

        return noisy_circuit

    def _inplace_append_noisy_moment(
        self,
        *,
        circuit: stim.Circuit,
        moment: Collection[stim.CircuitInstruction],
        system_qubits: set[int],
        immune_qubits: set[int],
        immune_op_tag: str,
    ) -> None:
        """Apps noise to a moment and appends it to a circuit (in-place).

        This method processes all operations in a moment, applies their respective noise rules, and
        adds the resulting noisy operations to the output circuit.

        Args:
            circuit: The circuit to append the noisy operations to.
            moment: Collection of operations happening during the moment in question.
            system_qubits: Set of all qubits in the system that may experience idle errors.
            immune_qubits: Set of all qubits that should not have noise applied to them.
            immune_op_tag: If an operation contains this string in its tag, that operation is
                noiseless.
        """
        noise_after_moment = stim.Circuit()
        for op in moment:
            if immune_op_tag in op.tag or (rule := self.get_noise_rule(op)) is None:
                circuit.append(op)
            else:
                noisy_op, after = rule.noisy_operation(op, immune_qubits=immune_qubits)
                circuit.append(noisy_op)
                noise_after_moment += after

        circuit += noise_after_moment

        moment_was_noisy = any(immune_op_tag not in op.tag for op in moment)
        if moment_was_noisy and self.idle_error or self.additional_error_waiting_for_m_or_r:
            self._inplace_append_idle_errors(
                circuit=circuit,
                moment=moment,
                system_qubits=system_qubits,
                immune_qubits=immune_qubits,
            )

    def _inplace_append_idle_errors(
        self,
        *,
        circuit: stim.Circuit,
        moment: Collection[stim.CircuitInstruction],
        system_qubits: set[int],
        immune_qubits: set[int],
    ) -> None:
        """Append idling errors from the given moment to the given circuit.

        This method identifies which qubits are idle during a moment and applies depolarization noise
        to them according to the noise model parameters.

        Args:
            circuit: The circuit to append idle error operations to.
            moment: The collection of operations happening in the final moment of the circuit.
            system_qubits: Set of all qubits in the system that can experience idle errors.
            immune_qubits: Set of qubit indices that should not have noise applied to them.

        Raises:
            ValueError: If qubits are operated on multiple times within the same moment without a
                TICK in between.
        """
        collapsed_qubits: list[int] = []
        operation_qubits: list[int] = []
        classically_controlled_qubits: list[int] = []
        for op in moment:
            if OP_TYPES[op.name] == ANNOTATION:
                continue

            target_qubits = [
                target.qubit_value for target in op.targets_copy() if target.qubit_value is not None
            ]
            if op.name in COLLAPSING_OPS:
                qubits = collapsed_qubits
            elif _involves_classical_bits(op):
                qubits = classically_controlled_qubits
            else:
                qubits = operation_qubits
            qubits.extend(target_qubits)

        # Safety check for operation collisions.
        usage_counts = collections.Counter(
            collapsed_qubits + operation_qubits + classically_controlled_qubits
        )
        qubits_used_multiple_times = {qubit for qubit, count in usage_counts.items() if count != 1}
        if qubits_used_multiple_times:
            raise ValueError(
                f"Qubits were operated on multiple times without a TICK in between:\n"
                f"multiple uses: {sorted(qubits_used_multiple_times)}\n"
                f"moment:\n{moment}"
            )

        non_collapse_qubits = system_qubits - immune_qubits - set(collapsed_qubits)
        idle_qubits = sorted(non_collapse_qubits - set(operation_qubits))

        if self.idle_error and idle_qubits:
            circuit.append("DEPOLARIZE1", idle_qubits, self.idle_error)
        if self.additional_error_waiting_for_m_or_r and collapsed_qubits and non_collapse_qubits:
            circuit.append(
                "DEPOLARIZE1", non_collapse_qubits, self.additional_error_waiting_for_m_or_r
            )


class DepolarizingNoiseModel(NoiseModel):
    """Creates a near-standard circuit depolarizing noise model.

    All operations has the same error parameter p:
    - One-qubit Clifford gates get one-qubit depolarization.
    - Two-qubit Clifford gates get two-qubit depolarization.
    - Measurements have their outcomes probabilistically flipped.
    - Reset gates probabalistically reset qubits to the wrong (orthogonal) state.
    - If applicable, every idling qubit in a given moment gets depolarized.
    """

    def __init__(self, p: float, *, include_idling_error: bool = False) -> None:
        """Instantiate a depolarizing noise model."""
        self.p = p
        self.include_idling_error = include_idling_error
        super().__init__(
            clifford_1q_error=p,
            clifford_2q_error=p,
            readout_error=p,
            reset_error=p,
            idle_error=p if include_idling_error else False,
        )


class SI1000NoiseModel(NoiseModel):
    """A superconducting-inspired noise model defined in "A Fault-Tolerant Honeycomb Memory"

    This noise model is defined by a two-qubit gate infidelity that determines all error rates.

    See https://arxiv.org/abs/2108.10457.
    """

    def __init__(self, p: float) -> None:
        """Instantiate a superconducting-inspired noise model."""
        self.p = p
        super().__init__(
            clifford_1q_error=p / 10,
            clifford_2q_error=p,
            readout_error=p * 5,
            reset_error=p * 2,
            idle_error=p / 10,
            additional_error_waiting_for_m_or_r=2 * p,
        )


def _get_standardized_name(op: stim.CircuitInstruction) -> str:
    """Stardardized name of a circuit instruction.

    The primary function of this method is to disambiguate the basis of measurement and reset gates.

    Args:
        op:_name The name of the circuit instruction that we need to standardize.

    Returns:
        str: The standardized name.
    """
    op_name = op.name
    if op_name == "M" or op_name == "R" or op_name == "MR":
        return op_name + "Z"

    if op_name == "MPP":
        name = "M"
        for target in op.targets_copy()[::2]:
            if target.is_x_target:
                name += "X"
            elif target.is_y_target:
                name += "Y"
            else:
                assert target.is_z_target
                name += "Z"
        return name

    return op_name


def _split_moments_with_ticks(circuit: stim.Circuit) -> stim.Circuit:
    """Insert TICKs into a circuit to split stim.CircuitInstruction that reuse qubits.

    This preprocessing ensures that errors are applied correctly to a stim.CircuitInstruction that
    reuses qubits.

    Args:
        circuit: The input circuit to preprocess.

    Returns:
        stim.Circuit: A circuit with TICKs added to prevent instructions from reusing qubits.
    """
    result = stim.Circuit()
    used_qubits: set[int] = set()

    for op in circuit:
        if isinstance(op, stim.CircuitRepeatBlock):
            # Process repeat blocks recursively
            if used_qubits:
                result.append("TICK")
                used_qubits.clear()
            processed_body = _split_moments_with_ticks(op.body_copy())
            result.append(
                stim.CircuitRepeatBlock(
                    repeat_count=op.repeat_count, body=processed_body, tag=op.tag
                )
            )
            continue

        if op.name == "TICK":
            # Explicit TICK - clear used qubits
            result.append("TICK")
            used_qubits.clear()
            continue

        """
        For preprocessing, we need to force splitting of multi-target operations to detect qubit
        reuse properly.  Use a dummy immune_qubits set with -1 to force splitting of 2-qubit
        operations.
        """
        split_ops = list(_split_targets_if_needed(op, immune_qubits={-1}))

        for split_op in split_ops:
            # Check if this split operation would reuse any qubits
            op_qubits = set()
            if OP_TYPES[split_op.name] != ANNOTATION:
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


def _involves_classical_bits(op: stim.CircuitInstruction) -> bool:
    """Determines if an operation involves classical bits.

    Args:
        op: The circuit instruction to check.

    Returns:
        True if the operation involves classical control bits.  False otherwise.
    """
    return any(
        target.is_measurement_record_target or target.is_sweep_bit_target
        for target in op.targets_copy()
    )


def _split_targets_if_needed(
    op: stim.CircuitInstruction, immune_qubits: set[int]
) -> Iterator[stim.CircuitInstruction]:
    """Splits operations into pieces as needed.

    This function splits operations like MPP into each product, and separates classical control
    operations from quantum operations.

    Args:
        op: The circuit instruction to potentially split.
        immune_qubits: Set of qubits that are immune to noise.

    Yields:
        Circuit instructions, potentially split into smaller pieces.
    """
    op_type = OP_TYPES[op.name]
    if op_type == CLIFFORD_2Q:
        yield from _split_targets_clifford_2q(op, immune_qubits)
    elif op_type == JUST_MEASURE_PP:
        yield from _split_targets_mpp(op, immune_qubits)
    elif op_type in [NOISE, ANNOTATION]:
        yield op
    else:
        yield from _split_targets_clifford_1q(op, immune_qubits)


def _split_targets_clifford_1q(
    op: stim.CircuitInstruction, immune_qubits: set[int]
) -> Iterator[stim.CircuitInstruction]:
    """Splits single-qubit Clifford operations when immune qubits are present.

    Args:
        op: The single-qubit Clifford operation to split.
        immune_qubits: Set of qubits that are immune to noise.

    Yields:
        Circuit instructions split into individual single-target operations.
    """
    if immune_qubits:
        args = op.gate_args_copy()
        for target in op.targets_copy():
            yield stim.CircuitInstruction(op.name, [target], args, tag=op.tag)
    else:
        yield op


def _split_targets_clifford_2q(
    op: stim.CircuitInstruction, immune_qubits: set[int]
) -> Iterator[stim.CircuitInstruction]:
    """Splits two-qubit Clifford operations into individual gate pairs.

    This function separates classical control system operations from quantum operations happening on
    the quantum computer.

    Args:
        op: The two-qubit Clifford operation to split.
        immune_qubits: Set of qubits that are immune to noise.

    Yields:
        Circuit instructions split into individual two-qubit gate operations.
    """
    assert OP_TYPES[op.name] == CLIFFORD_2Q
    targets = op.targets_copy()
    if immune_qubits or any(target.is_measurement_record_target for target in targets):
        args = op.gate_args_copy()
        for k in range(0, len(targets), 2):
            yield stim.CircuitInstruction(op.name, targets[k : k + 2], args, tag=op.tag)
    else:
        yield op


def _split_targets_mpp(
    op: stim.CircuitInstruction, immune_qubits: set[int]
) -> Iterator[stim.CircuitInstruction]:
    """Splits an MPP operation into one operation for each Pauli product it measures.

    Args:
        op: The MPP operation to split.
        immune_qubits: Set of qubit indices that should not have noise applied
            (unused in this function but included for interface consistency).

    Yields:
        Circuit instructions, one for each Pauli product measurement.
    """
    assert OP_TYPES[op.name] == JUST_MEASURE_PP
    targets = op.targets_copy()
    args = op.gate_args_copy()
    start = end = 0
    while end < len(targets):
        if end + 1 == len(targets) or not targets[end + 1].is_combiner:
            yield stim.CircuitInstruction(op.name, targets[start : end + 1], args, tag=op.tag)
            end += 1
            start = end
        else:
            end += 2
    assert end == len(targets)


def _iter_moments_and_repeat_blocks(
    circuit: stim.Circuit, immune_qubits: set[int]
) -> Iterator[stim.CircuitRepeatBlock | list[stim.CircuitInstruction]]:
    """Splits a circuit into moments and some operations into pieces.

    Classical control system operations like CX rec[-1] 0 are split from quantum operations like
    CX 1 0.  MPP operations are split into one operation per Pauli product.

    Args:
        circuit: The circuit to split into moments.
        immune_qubits: Set of qubits that are immune to noise.

    Yields:
        Lists of operations corresponding to one moment in the circuit, with any problematic
        operations like MPPs split into pieces, or CircuitRepeatBlock instances for repeat blocks.

    Note:
        A moment is the time between two TICKs.
    """
    current_moment: list[stim.CircuitInstruction] = []

    for op in circuit:
        if isinstance(op, stim.CircuitRepeatBlock):
            if current_moment:
                yield current_moment
                current_moment = []
            yield op
        elif op.name == "TICK":
            if current_moment:
                yield current_moment
                current_moment = []
        else:
            current_moment.extend(_split_targets_if_needed(op, immune_qubits))
    if current_moment:
        yield current_moment
