from __future__ import annotations

import abc

import stim

from qldpc.objects import Pauli, PauliXZ


class NoiseModel(abc.ABC):
    """
    Abstract base class for circuit-level noise models.

    Noise models wrap the application of stim operations to inject errors
    according to specific noise characteristics. Subclasses must define
    the error behavior for single-qubit gates, two-qubit gates, measurement,
    and reset operations.
    """

    @abc.abstractmethod
    def apply_1q_gates(self, circ: stim.Circuit, gate: str, qubits: list[int]) -> None:
        """
        Applies single-qubit gates and associated noise to the circuit.

        Args:
            circ: The stim.Circuit to apply gates and errors to.
            gate: The name of the single-qubit gate being applied (e.g., "H", "X").
            qubits: A list of qubit IDs the gate is applied to.
        """
        pass

    @abc.abstractmethod
    def apply_2q_gates(
        self, circ: stim.Circuit, gate: str, qubit_pairs: list[tuple[int, int]]
    ) -> None:
        """
        Applies two-qubit gates and associated noise to the circuit.

        Args:
            circ: The stim.Circuit to apply gates and errors to.
            gate: The name of the two-qubit gate being applied (e.g., "CX", "CZ").
            qubit_pairs: A list of qubit pairs involved in the two-qubit gate.
                         For gates like CX, the first qubit is the control and
                         the second is the target.
        """
        pass

    @abc.abstractmethod
    def apply_meas(self, circ: stim.Circuit, basis: PauliXZ, qubits: list[int]) -> None:
        """
        Applies measurement operations and associated noise to the circuit.

        Args:
            circ: The stim.Circuit to apply measurements and errors to.
            basis: The measurement basis (PauliXZ.X or PauliXZ.Z).
            qubits: A list of qubit IDs being measured.
        """
        pass

    @abc.abstractmethod
    def apply_reset(self, circ: stim.Circuit, basis: PauliXZ, qubits: list[int]) -> None:
        """
        Applies reset operations and associated noise to the circuit.

        Args:
            circ: The stim.Circuit to apply resets and errors to.
            basis: The basis to reset in (PauliXZ.X or PauliXZ.Z).
            qubits: A list of qubit IDs being reset.
        """
        pass


class CircuitNoiseModel(NoiseModel):
    """
    A circuit-level noise model with customizable error probabilities
    for 1-qubit gates, 2-qubit gates, measurement, and reset operations.

    Attributes:
        p1q: Error probability for 1-qubit gates (e.g., depolarizing).
        p2q: Error probability for 2-qubit gates (e.g., depolarizing).
        p_meas: Error probability for measurement outcomes (e.g., flip error).
        p_reset: Error probability for reset operations (e.g., flip error after reset).
    """

    def __init__(
        self,
        p1q: float,
        p2q: float,
        p_meas: float,
        p_reset: float,
    ) -> None:
        """
        Initializes the CircuitNoiseModel with specific error probabilities.

        Args:
            p1q: Error probability for single-qubit gates. Must be between 0 and 1.
            p2q: Error probability for two-qubit gates. Must be between 0 and 1.
            p_meas: Error probability for measurement outcomes. Must be between 0 and 1.
            p_reset: Error probability for reset operations. Must be between 0 and 1.

        Raises:
            ValueError: If any error probability is not between 0 and 1.
        """
        if not all(0.0 <= p <= 1.0 for p in [p1q, p2q, p_meas, p_reset]):
            raise ValueError("Error probabilities must be between 0 and 1.")

        self.p1q = p1q
        self.p2q = p2q
        self.p_meas = p_meas
        self.p_reset = p_reset

    def apply_1q_gates(self, circ: stim.Circuit, gate: str, qubits: list[int]) -> None:
        """
        Applies single-qubit gates followed by single-qubit depolarizing errors
        with probability self.p1q.

        Args:
            circ: The stim.Circuit to apply gates and errors to.
            gate: The name of the single-qubit gate being applied.
            qubits: A list of qubit IDs the gate is applied to.
        """
        circ.append(gate, qubits)
        circ.append("TICK")
        # Apply depolarizing error after the gate
        if self.p1q > 0:
            circ.append("DEPOLARIZE1", qubits, self.p1q)
        circ.append("TICK")

    def apply_2q_gates(
        self, circ: stim.Circuit, gate: str, qubit_pairs: list[tuple[int, int]]
    ) -> None:
        """
        Applies two-qubit gates followed by two-qubit depolarizing errors
        with probability self.p2q.

        Args:
            circ: The stim.Circuit to apply gates and errors to.
            gate: The name of the two-qubit gate being applied.
            qubit_pairs: A list of qubit pairs involved in the two-qubit gate.
        """
        flattened_args = [q for pair in qubit_pairs for q in pair]
        circ.append(gate, flattened_args)
        circ.append("TICK")
        # Apply depolarizing error after the gate
        if self.p2q > 0:
            circ.append("DEPOLARIZE2", flattened_args, self.p2q)
        circ.append("TICK")

    def apply_meas(self, circ: stim.Circuit, basis: PauliXZ, qubits: list[int]) -> None:
        """
        Applies measurement operation with an error probability self.p_meas.
        The error is a flip error before the measurement, dependent on the basis.

        Args:
            circ: The stim.Circuit to apply measurements and errors to.
            basis: The measurement basis (PauliXZ.X or PauliXZ.Z).
            qubits: A list of qubit IDs being measured.

        Raises:
            ValueError: If an invalid basis is provided.
        """
        # Apply a flip error before measurement
        if self.p_meas > 0:
            if basis is Pauli.Z:
                # X error before Z measurement causes a flip
                circ.append("X_ERROR", qubits, self.p_meas)
            elif basis is Pauli.X:
                # Z error before X measurement causes a flip
                circ.append("Z_ERROR", qubits, self.p_meas)
            else:
                raise ValueError(f"Invalid basis for measurement: {basis}")
            circ.append("TICK")

        # Apply the measurement
        if basis is Pauli.Z:
            meas_gate = "M"
        elif basis is Pauli.X:
            meas_gate = "MX"
        else:
            # This should ideally not be reached if the error check passed, but included for safety
            raise ValueError(f"Invalid basis for measurement: {basis}")
        circ.append(meas_gate, qubits)
        circ.append("TICK")

    def apply_reset(self, circ: stim.Circuit, basis: PauliXZ, qubits: list[int]) -> None:
        """
        Applies reset operation with an error probability self.p_reset.
        The error is a flip error after the reset, dependent on the basis.

        Args:
            circ: The stim.Circuit to apply resets and errors to.
            basis: The basis to reset in (PauliXZ.X or PauliXZ.Z).
            qubits: A list of qubit IDs being reset.

        Raises:
            ValueError: If an invalid basis is provided.
        """
        # Determine the appropriate reset gate and the error gate that causes a flip
        if basis is Pauli.Z:
            reset_gate = "R"
            error_gate = "X_ERROR"  # X error flips |0> to |1> after Z reset
        elif basis is Pauli.X:
            reset_gate = "RX"
            error_gate = "Z_ERROR"  # Z error flips |+> to |-> after X reset
        else:
            raise ValueError(f"Invalid basis for reset: {basis}")

        circ.append(reset_gate, qubits)
        circ.append("TICK")

        # Apply a flip error after reset
        if self.p_reset > 0:
            circ.append(error_gate, qubits, self.p_reset)
            circ.append("TICK")


class UniformNoiseModel(CircuitNoiseModel):
    """
    A uniform noise model where all gate, reset, and measurement errors
    have the same probability p.

    This class inherits from CircuitNoiseModel and sets all error probabilities
    (p1q, p2q, p_meas, p_reset) to the same value p.
    """

    def __init__(self, p: float) -> None:
        """
        Initializes the UniformNoiseModel with a single error probability p.

        Args:
            p: The uniform error probability for all operations.
            Must be between 0 and 1.

        Raises:
            ValueError: If the probability p is not between 0 and 1.
        """
        # Call the parent class constructor with the same probability for all error types
        super().__init__(p1q=p, p2q=p, p_meas=p, p_reset=p)
