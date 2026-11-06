# SPDX-License-Identifier: Apache-2.0

"""Classification of Stim instructions for noise models, and the names that noise rules match.

See :mod:`qldpc.circuits.noise_model` for attribution and license information.
"""

from __future__ import annotations

import functools
import re

import stim

# a rule key that names a Pauli product of an MPP, SPP, or SPP_DAG gate, as in "MXYZ" or "SXY_DAG"
_BASIS_SUFFIXED_RULE_KEY = re.compile(r"([MS])([XYZ]+)(_DAG)?")

####################################################################################################
# instruction classification

CLIFFORD_1Q = "C1"
CLIFFORD_2Q = "C2"
CLIFFORD_PP = "CPP"
JUST_MEASURE_1Q = "M1"
JUST_MEASURE_2Q = "M2"
JUST_MEASURE_PP = "MPP"
JUST_RESET_1Q = "R1"
MEASURE_RESET_1Q = "MR1"
ANNOTATION = "info"
NOISE = "!?"
UNSUPPORTED = "unsupported"

# the op types of gates, as opposed to noise channels, annotations, and unsupported instructions.
# Noise rules, including those from a rule_func, apply to gates.
GATE_OP_TYPES = frozenset(
    {
        CLIFFORD_1Q,
        CLIFFORD_2Q,
        CLIFFORD_PP,
        JUST_MEASURE_1Q,
        JUST_MEASURE_2Q,
        JUST_MEASURE_PP,
        JUST_RESET_1Q,
        MEASURE_RESET_1Q,
    }
)


@functools.cache
def op_type(op_name: str) -> str:
    """Return the qLDPC op-type category for a gate name (canonical or alias)."""
    try:
        data = stim.gate_data(op_name)
    except IndexError:
        return UNSUPPORTED
    if data.flows is None:
        # Pure noise (X_ERROR, DEPOLARIZE1, HERALDED_ERASE, ...) or annotation (TICK, DETECTOR,
        # MPAD, QUBIT_COORDS, SHIFT_COORDS, OBSERVABLE_INCLUDE).  These are the gates that lack a
        # tableau-based flow specification.
        return NOISE if data.is_noisy_gate else ANNOTATION
    if data.is_reset:
        assert data.is_single_qubit_gate
        if data.produces_measurements:
            return MEASURE_RESET_1Q
        return JUST_RESET_1Q
    if data.produces_measurements:
        if data.is_single_qubit_gate:
            return JUST_MEASURE_1Q
        if data.is_two_qubit_gate:
            return JUST_MEASURE_2Q
        assert data.takes_pauli_targets
        return JUST_MEASURE_PP
    if data.is_unitary:
        if data.is_single_qubit_gate:
            return CLIFFORD_1Q
        if data.is_two_qubit_gate:
            return CLIFFORD_2Q
        assert data.takes_pauli_targets
        return CLIFFORD_PP
    raise NotImplementedError(  # pragma: no cover
        f"qLDPC does not know how to classify the gate {op_name!r}."
        f"  Please open an issue at https://github.com/qLDPCOrg/qLDPC/issues."
    )


# every name of every Stim gate, including aliases
ALL_STIM_GATE_NAMES = frozenset(alias for gd in stim.gate_data().values() for alias in gd.aliases)

JUST_MEASURE_OPS = frozenset(
    name
    for name in ALL_STIM_GATE_NAMES
    if op_type(name) in (JUST_MEASURE_1Q, JUST_MEASURE_2Q, JUST_MEASURE_PP)
)
JUST_RESET_OPS = frozenset(name for name in ALL_STIM_GATE_NAMES if op_type(name) == JUST_RESET_1Q)
# gates that measure and then reset, such as MR
MEASURE_AND_RESET_OPS = frozenset(
    name for name in ALL_STIM_GATE_NAMES if op_type(name) == MEASURE_RESET_1Q
)
# all measurements and resets, including gates that measure and then reset
MEASURE_OR_RESET_OPS = JUST_MEASURE_OPS | JUST_RESET_OPS | MEASURE_AND_RESET_OPS


@functools.lru_cache(maxsize=4096)
def involves_classical_bits(op: stim.CircuitInstruction) -> bool:
    """Determines if an operation involves classical bits.

    Noise models and schedulers check each operation several times, and circuits repeat
    instructions, so recent results are cached.  Instructions are hashed by value.

    Args:
        op: The circuit instruction to check.

    Returns:
        True if the operation involves classical control bits.  False otherwise.
    """
    return any(
        target.is_measurement_record_target or target.is_sweep_bit_target
        for target in op.targets_copy()
    )


####################################################################################################
# names and keys of noise rules


def get_gate_aliases(op: stim.CircuitInstruction) -> tuple[str, ...]:
    """The names by which the keys of ``NoiseModel.rules`` match an operation, most specific first.

    These are Stim's aliases of the gate, which start with the key that ``canonical_rule_key`` gives
    it.  A Pauli product of an MPP, SPP, or SPP_DAG gate is first named by its Paulis, as in
    ``"MXYZ"`` for ``MPP X*Y*Z`` or ``"SXY_DAG"`` for ``SPP_DAG X*Y``, except that ``MPP Z`` is
    named ``"M"``.
    """
    # other gates are named by Stim's aliases only
    aliases = _stim_aliases(op.name)
    if op.name not in ("MPP", "SPP", "SPP_DAG"):
        return aliases

    # a Pauli product must be a single product, alternating Paulis and combiners
    prefix, suffix = ("S", "_DAG") if op.name == "SPP_DAG" else (op.name[0], "")
    targets = op.targets_copy()
    if not all(target.is_combiner for target in targets[1::2]):
        raise ValueError(
            f"{op.name} must be split into a single Pauli product before alias enumeration"
        )

    # name the product by its Paulis, as in "MXYZ" or "SXY_DAG", before stim's names
    basis = ""
    for target in targets[::2]:
        if target.is_x_target:
            basis += "X"
        elif target.is_y_target:
            basis += "Y"
        else:
            assert target.is_z_target
            basis += "Z"
    basis_name = prefix + basis + suffix
    return ("M" if basis_name == "MZ" else basis_name, *aliases)


def canonical_rule_key(op_name: str) -> str:
    """Canonicalize and validate a user-provided noise-rule key."""
    if not isinstance(op_name, str):
        raise TypeError(f"Noise rule keys must be strings, got {type(op_name).__name__}")
    op_name = op_name.upper()
    try:
        gate_data = stim.gate_data(op_name)
    except IndexError:
        # a name that stim does not know must name a Pauli product, as in "MXYZ" or "SXY_DAG"
        match = _BASIS_SUFFIXED_RULE_KEY.fullmatch(op_name)
        if match is None:
            raise ValueError(f"Unrecognized noise rule key {op_name!r}") from None
        prefix, _basis, dag = match.groups()
        if prefix == "M" and dag:
            raise ValueError(f"Invalid noise rule key {op_name!r}: MPP has no _DAG form")
        return op_name
    else:
        # a name that stim knows becomes its canonical name, which must name a gate
        canonical = gate_data.aliases[0]
        if op_type(canonical) not in GATE_OP_TYPES:
            raise ValueError(f"Noise rules cannot target {op_name!r}")
        return canonical


def known_gate_shape(op_name: str) -> tuple[int, bool, bool] | None:
    """The number of qubits of the gates that a rule key names, and whether they measure or reset.

    Keys that name gates on any number of qubits, such as ``"MPP"``, and keys that name no gate,
    have no fixed shape, and give None.
    """
    if op_type(op_name) in (
        CLIFFORD_1Q,
        CLIFFORD_2Q,
        JUST_MEASURE_1Q,
        JUST_MEASURE_2Q,
        JUST_RESET_1Q,
        MEASURE_RESET_1Q,
    ):
        gate_data = stim.gate_data(op_name)
        arity = 2 if gate_data.is_two_qubit_gate else 1
        return arity, gate_data.produces_measurements, gate_data.is_reset
    # a key that names a Pauli product, as in "MXYZ" or "SXY_DAG", has a qubit per Pauli
    match = _BASIS_SUFFIXED_RULE_KEY.fullmatch(op_name)
    if match is None:
        return None
    prefix, basis, dag = match.groups()
    gate_data = stim.gate_data(f"{prefix}PP{dag or ''}")
    return len(basis), gate_data.produces_measurements, gate_data.is_reset


@functools.cache
def _stim_aliases(op_name: str) -> tuple[str, ...]:
    """Cached stim aliases of a gate name."""
    return tuple(stim.gate_data(op_name).aliases)
