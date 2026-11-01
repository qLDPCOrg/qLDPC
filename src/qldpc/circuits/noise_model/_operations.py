# SPDX-License-Identifier: Apache-2.0

"""Classification of Stim instructions for noise models, and the names that noise rules match.

See :mod:`qldpc.circuits.noise_model` for attribution and license information.
"""

from __future__ import annotations

import functools
import re

import stim

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

# The op-type categories that name a genuine noisy gate (as opposed to pure-noise, annotation, or
# unsupported instructions).  These are the operations a user-provided ``rule_func`` is consulted
# for, once decomposed into individual gate applications.
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


# All canonical + alias names stim knows about, plus their aliases (so lookups by any name work).
ALL_STIM_GATE_NAMES = frozenset(alias for gd in stim.gate_data().values() for alias in gd.aliases)

JUST_MEASURE_OPS = frozenset(
    name
    for name in ALL_STIM_GATE_NAMES
    if op_type(name) in (JUST_MEASURE_1Q, JUST_MEASURE_2Q, JUST_MEASURE_PP)
)
JUST_RESET_OPS = frozenset(name for name in ALL_STIM_GATE_NAMES if op_type(name) == JUST_RESET_1Q)
MEASURE_AND_RESET_OPS = frozenset(
    name for name in ALL_STIM_GATE_NAMES if op_type(name) == MEASURE_RESET_1Q
)
COLLAPSING_OPS = JUST_MEASURE_OPS | JUST_RESET_OPS | MEASURE_AND_RESET_OPS


def involves_classical_bits(op: stim.CircuitInstruction) -> bool:
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


####################################################################################################
# names and keys of noise rules


def get_gate_aliases(op: stim.CircuitInstruction) -> tuple[str, ...]:
    """Return the names by which ``op`` can be matched in ``rules`` and ``rule_func``.

    For a single-product MPP / SPP / SPP_DAG op, the basis-suffixed name (e.g. ``"MXYZ"`` for ``MPP
    X*Y*Z``, ``"SXY_DAG"`` for ``SPP_DAG X*Y``) is yielded first — it's more specific than the raw
    ``"MPP"`` / ``"SPP"`` / ``"SPP_DAG"`` — followed by the corresponding stim aliases.  For every
    other gate, this is just stim's alias list, which begins with the canonical name.  Single-qubit
    ``MPP Z`` uses the canonical measurement name ``"M"``, making it equivalent to ``M`` / ``MZ``.
    """
    aliases = _stim_aliases(op.name)
    if op.name not in ("MPP", "SPP", "SPP_DAG"):
        return aliases
    prefix, suffix = ("S", "_DAG") if op.name == "SPP_DAG" else (op.name[0], "")
    targets = op.targets_copy()
    if not all(target.is_combiner for target in targets[1::2]):
        raise ValueError(
            f"{op.name} must be split into a single Pauli product before alias enumeration"
        )
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
        match = _BASIS_SUFFIXED_RULE_KEY.fullmatch(op_name)
        if match is None:
            raise ValueError(f"Unrecognized noise rule key {op_name!r}") from None
        prefix, _basis, dag = match.groups()
        if prefix == "M" and dag:
            raise ValueError(f"Invalid noise rule key {op_name!r}: MPP has no _DAG form")
        stim.gate_data(f"{prefix}PP{dag or ''}")
        return op_name
    else:
        canonical = gate_data.aliases[0]
        if op_type(canonical) not in GATE_OP_TYPES:
            raise ValueError(f"Noise rules cannot target {op_name!r}")
        return canonical


def known_gate_shape(op_name: str) -> tuple[int, bool, bool] | None:
    """Return ``(arity, can_measure, can_reset)`` for a rule key with statically-known shape.

    Applies to fixed-arity stim gates (e.g. ``"H"``, ``"CX"``, ``"M"``, ``"R"``, ``"MR"``) and to
    basis-suffixed rule keys (``"MXYZ"`` for ``MPP X*Y*Z``, ``"SXY_DAG"`` for ``SPP_DAG X*Y``).
    Returns ``None`` for variable-arity names (bare ``"MPP"`` / ``"SPP"`` / ``"SPP_DAG"``) or for
    names not recognized as either kind.
    """
    this_op_type = op_type(op_name)
    if this_op_type in (CLIFFORD_1Q, JUST_MEASURE_1Q, JUST_RESET_1Q, MEASURE_RESET_1Q):
        gate_data = stim.gate_data(op_name)
        return 1, gate_data.produces_measurements, gate_data.is_reset
    if this_op_type in (CLIFFORD_2Q, JUST_MEASURE_2Q):
        gate_data = stim.gate_data(op_name)
        return 2, gate_data.produces_measurements, gate_data.is_reset
    # Basis-suffixed rule keys parse as `[MS](XYZ)+(_DAG)?` — look up the underlying PP gate
    # (``MPP`` / ``SPP`` / ``SPP_DAG``) for the measurement/reset flags.
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


_BASIS_SUFFIXED_RULE_KEY = re.compile(r"([MS])([XYZ]+)(?:(_DAG))?")
