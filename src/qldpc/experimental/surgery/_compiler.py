"""Private compiler-facing records and lowering for lattice surgery."""

from __future__ import annotations

import dataclasses
from collections.abc import Collection, Mapping

import numpy as np
import stim

from qldpc.circuits.bookkeeping import QubitIDs
from qldpc.circuits.noise_model import NoiseModel
from qldpc.codes.common import CSSCode
from qldpc.objects import Pauli, PauliXZ

from .bridge import build_bridge
from .circuit import _build_joint_ppm_circuit, keep_only_observable, logical_state_init
from .gadget import build_gadget


class _SurgeryBackendUnsupportedError(RuntimeError):
    """A valid logical request that the selected surgery construction cannot realize."""


@dataclasses.dataclass(frozen=True)
class _LogicalBlock:
    """A CSS code occupying one explicitly identified logical block."""

    identifier: str
    code: CSSCode


@dataclasses.dataclass(frozen=True)
class _LogicalPauliRef:
    """One logical Pauli factor in a compiler request."""

    block_id: str
    logical_index: int
    basis: Pauli


@dataclasses.dataclass(frozen=True)
class _PauliPairMeasurement:
    """A named same-basis logical Pauli-pair measurement."""

    result_key: str
    factors: tuple[_LogicalPauliRef, _LogicalPauliRef]


@dataclasses.dataclass(frozen=True)
class _WebsterBackendDetails:
    """Reproducibility metadata for the initial fallback backend."""

    name: str
    bridge_width: int
    cellulate_max_len: int | None
    boosted: bool


@dataclasses.dataclass(frozen=True)
class _SurgeryResourceCounts:
    """Exact static counts for one emitted surgery circuit."""

    source_data_qubits: int
    gadget_ancilla_qubits: int
    adapter_data_qubits: int
    syndrome_ancilla_qubits: int
    peak_physical_qubits: int
    syndrome_rounds: int
    logical_outcomes: int


@dataclasses.dataclass(frozen=True)
class _CompiledPairMeasurement:
    """One logical request lowered to the Webster/universal-adapter backend."""

    request: _PauliPairMeasurement
    circuit: stim.Circuit
    diagnostic_circuit: stim.Circuit
    merged_code: CSSCode
    result_observables: tuple[tuple[str, int], ...]
    diagnostic_observable_indices: tuple[int, ...]
    block_data_qubits: tuple[tuple[str, tuple[int, ...]], ...]
    resolved_operators: tuple[tuple[int, ...], tuple[int, ...]]
    backend: _WebsterBackendDetails
    resources: _SurgeryResourceCounts


def _as_pauli_xz(basis: Pauli) -> PauliXZ:
    """Validate and narrow a Pauli value to an X/Z basis."""
    if basis is Pauli.X:
        return Pauli.X
    if basis is Pauli.Z:
        return Pauli.Z
    raise ValueError(f"logical Pauli basis must be Pauli.X or Pauli.Z, got {basis!r}")


def _validate_blocks(blocks: Collection[_LogicalBlock]) -> dict[str, _LogicalBlock]:
    """Validate and index the two logical blocks in one pair-measurement request."""
    block_list = tuple(blocks)
    if len(block_list) != 2:
        raise ValueError(
            f"an inter-block pair measurement requires exactly 2 blocks, got {len(block_list)}"
        )

    block_by_id: dict[str, _LogicalBlock] = {}
    for block in block_list:
        if not isinstance(block, _LogicalBlock):
            raise TypeError(f"blocks must contain _LogicalBlock values, got {type(block).__name__}")
        if not isinstance(block.identifier, str):
            raise TypeError("logical block identifiers must be strings")
        if not block.identifier.strip():
            raise ValueError("logical block identifiers must not be empty")
        if block.identifier in block_by_id:
            raise ValueError(f"duplicate logical block identifier: {block.identifier!r}")
        if not isinstance(block.code, CSSCode):
            raise TypeError(
                f"logical block {block.identifier!r} must contain a CSSCode, "
                f"got {type(block.code).__name__}"
            )
        if block.code.field.order != 2:
            raise ValueError(
                f"logical block {block.identifier!r} must contain a binary CSS code, "
                f"got GF({block.code.field.order})"
            )
        if block.code.is_subsystem_code:
            raise ValueError(
                f"logical block {block.identifier!r} must contain a stabilizer CSS code, "
                "not a subsystem code"
            )
        block_by_id[block.identifier] = block
    return block_by_id


def _validate_request(
    request: _PauliPairMeasurement,
    block_by_id: Mapping[str, _LogicalBlock],
) -> tuple[tuple[_LogicalPauliRef, _LogicalBlock], tuple[_LogicalPauliRef, _LogicalBlock]]:
    """Validate a pair request and resolve its factors to logical blocks."""
    if not isinstance(request, _PauliPairMeasurement):
        raise TypeError(f"request must be a _PauliPairMeasurement, got {type(request).__name__}")
    if not isinstance(request.result_key, str):
        raise TypeError("measurement result_key must be a string")
    if not request.result_key.strip():
        raise ValueError("measurement result_key must not be empty")
    if not isinstance(request.factors, tuple) or len(request.factors) != 2:
        raise ValueError("a Pauli-pair measurement requires exactly 2 factors")

    resolved: list[tuple[_LogicalPauliRef, _LogicalBlock]] = []
    for factor in request.factors:
        if not isinstance(factor, _LogicalPauliRef):
            raise TypeError(
                f"measurement factors must be _LogicalPauliRef values, got {type(factor).__name__}"
            )
        if not isinstance(factor.block_id, str):
            raise TypeError("logical Pauli block identifiers must be strings")
        if factor.block_id not in block_by_id:
            raise ValueError(f"measurement references unknown logical block {factor.block_id!r}")
        _as_pauli_xz(factor.basis)
        if not isinstance(factor.logical_index, int) or isinstance(factor.logical_index, bool):
            raise TypeError("logical indices must be integers")
        block = block_by_id[factor.block_id]
        if not 0 <= factor.logical_index < block.code.dimension:
            raise IndexError(
                f"logical_index={factor.logical_index} is out of range for block "
                f"{factor.block_id!r} with dimension={block.code.dimension}"
            )
        resolved.append((factor, block))

    (factor_l, block_l), (factor_r, block_r) = resolved
    if factor_l.block_id == factor_r.block_id:
        raise ValueError("an inter-block pair measurement requires two distinct block identifiers")
    if factor_l.basis is not factor_r.basis:
        raise ValueError(
            "the Webster fallback requires same-basis factors, got "
            f"{factor_l.basis!r} and {factor_r.basis!r}"
        )
    return (factor_l, block_l), (factor_r, block_r)


def _resolve_initial_states(
    initial_states: Mapping[str, str] | None,
    resolved: tuple[
        tuple[_LogicalPauliRef, _LogicalBlock],
        tuple[_LogicalPauliRef, _LogicalBlock],
    ],
) -> tuple[str, str] | None:
    """Resolve logical state labels to per-physical-qubit initialization strings."""
    if initial_states is None:
        return None
    if not isinstance(initial_states, Mapping):
        raise TypeError("initial_states must be a mapping from block identifiers to state labels")

    expected_ids = {factor.block_id for factor, _ in resolved}
    provided_ids = set(initial_states)
    if provided_ids != expected_ids:
        missing = sorted(expected_ids - provided_ids)
        extra = sorted(provided_ids - expected_ids, key=repr)
        raise ValueError(
            f"initial_states must name exactly the measured blocks; missing={missing}, extra={extra}"
        )

    data_init: list[str] = []
    for factor, block in resolved:
        state = initial_states[factor.block_id]
        if not isinstance(state, str):
            raise TypeError(f"initial state for block {factor.block_id!r} must be a string")
        if state not in ("0", "1", "+", "-"):
            raise ValueError(
                f"initial state for block {factor.block_id!r} must be one of '0', '1', '+', '-'"
            )
        data_init.append(logical_state_init(block.code, state, log_idx=factor.logical_index))
    return data_init[0], data_init[1]


def _compile_interblock_pair_measurement(
    blocks: Collection[_LogicalBlock],
    request: _PauliPairMeasurement,
    *,
    rounds: int,
    noise_model: NoiseModel | None = None,
    initial_states: Mapping[str, str] | None = None,
) -> _CompiledPairMeasurement:
    """Lower one explicit inter-block Pauli-pair request through the fallback backend."""
    if not isinstance(rounds, int) or isinstance(rounds, bool):
        raise TypeError("rounds must be an integer")
    if rounds < 1:
        raise ValueError(f"rounds must be >= 1, got {rounds}")

    block_by_id = _validate_blocks(blocks)
    resolved_l, resolved_r = _validate_request(request, block_by_id)
    resolved = (resolved_l, resolved_r)
    factor_l, block_l = resolved_l
    factor_r, block_r = resolved_r
    basis = _as_pauli_xz(factor_l.basis)
    assert _as_pauli_xz(factor_r.basis) is basis
    data_init = _resolve_initial_states(initial_states, resolved)

    operators = (
        np.asarray(block_l.code.get_logical_ops(basis)[factor_l.logical_index], dtype=np.uint8),
        np.asarray(block_r.code.get_logical_ops(basis)[factor_r.logical_index], dtype=np.uint8),
    )
    supports = tuple(
        tuple(int(index) for index in np.flatnonzero(operator)) for operator in operators
    )
    for factor, support in zip(request.factors, supports):
        if len(support) < 2:
            raise _SurgeryBackendUnsupportedError(
                f"the Webster universal adapter requires support weight >= 2, but "
                f"{factor.block_id!r} logical {factor.logical_index} has weight {len(support)}"
            )

    gadget_l = build_gadget(block_l.code, operators[0], basis=basis)
    gadget_r = build_gadget(block_r.code, operators[1], basis=basis)
    try:
        bridge = build_bridge(gadget_l, gadget_r)
    except (RuntimeError, ValueError) as error:
        raise _SurgeryBackendUnsupportedError(
            f"the Webster universal adapter cannot realize result {request.result_key!r}"
        ) from error

    diagnostic_circuit, merged_code = _build_joint_ppm_circuit(
        gadget_l,
        gadget_r,
        bridge,
        intercode=True,
        rounds=rounds,
        noise_model=noise_model,
        data_init=data_init,
    )
    circuit = keep_only_observable(diagnostic_circuit, 0)

    qubit_ids = QubitIDs.from_code(merged_code)
    num_data_l = block_l.code.num_qudits
    num_data_r = block_r.code.num_qudits
    block_data_qubits = (
        (factor_l.block_id, tuple(qubit_ids.data[:num_data_l])),
        (
            factor_r.block_id,
            tuple(qubit_ids.data[num_data_l : num_data_l + num_data_r]),
        ),
    )
    gadget_ancilla_qubits = bridge.g_l_aug.incidence.shape[0] + bridge.g_r_aug.incidence.shape[0]
    resources = _SurgeryResourceCounts(
        source_data_qubits=num_data_l + num_data_r,
        gadget_ancilla_qubits=gadget_ancilla_qubits,
        adapter_data_qubits=bridge.width,
        syndrome_ancilla_qubits=merged_code.num_checks,
        peak_physical_qubits=diagnostic_circuit.num_qubits,
        syndrome_rounds=rounds,
        logical_outcomes=1,
    )
    assert (
        resources.source_data_qubits
        + resources.gadget_ancilla_qubits
        + resources.adapter_data_qubits
        == merged_code.num_qudits
    )
    assert (
        merged_code.num_qudits + resources.syndrome_ancilla_qubits == resources.peak_physical_qubits
    )

    resolved_operators = tuple(
        tuple(int(value) for value in operator.tolist()) for operator in operators
    )
    return _CompiledPairMeasurement(
        request=request,
        circuit=circuit,
        diagnostic_circuit=diagnostic_circuit,
        merged_code=merged_code,
        result_observables=((request.result_key, 0),),
        diagnostic_observable_indices=(1,),
        block_data_qubits=block_data_qubits,
        resolved_operators=(resolved_operators[0], resolved_operators[1]),
        backend=_WebsterBackendDetails(
            name="webster-universal-adapter",
            bridge_width=bridge.width,
            cellulate_max_len=None,
            boosted=False,
        ),
        resources=resources,
    )
