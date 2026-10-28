# SPDX-License-Identifier: Apache-2.0

"""Unit tests for channels.py."""

import pytest
import stim

from qldpc import circuits
from qldpc.circuits.conftest import circuits_are_equivalent


def test_pauli_channel_class() -> None:
    """PauliChannel construction, validation, and helpers."""

    ch = circuits.PauliChannel({"XYZ": 0.01, "ZZZ": 0.02})
    assert ch.num_qubits == 3
    assert dict(ch.probabilities) == {"XYZ": 0.01, "ZZZ": 0.02}
    assert bool(ch)
    assert ch == circuits.PauliChannel({"XYZ": 0.01, "ZZZ": 0.02})
    assert ch != circuits.PauliChannel({"XYZ": 0.01})
    assert ch != "foo"  # __eq__ returns NotImplemented for non-PauliChannel operands
    assert repr(ch).startswith("PauliChannel(")

    # zero-prob channel is falsy but still constructible
    assert not bool(circuits.PauliChannel({"X": 0.0}))

    # depolarizing(n, p): 4**n - 1 entries, probabilities sum to p, first key in lex order is "IX"
    dp = circuits.PauliChannel.depolarizing(2, 0.15)
    assert dp.num_qubits == 2
    assert len(dp.probabilities) == 15
    assert abs(sum(dp.probabilities.values()) - 0.15) < 1e-12
    assert next(iter(dp.probabilities.keys())) == "IX"

    # empty channel is the trivial 0-qubit channel
    empty = circuits.PauliChannel({})
    assert empty.num_qubits == 0
    assert dict(empty.probabilities) == {}
    assert not bool(empty)

    # validation errors
    with pytest.raises(ValueError, match="Identity string"):
        circuits.PauliChannel({"": 0.1})
    with pytest.raises(ValueError, match="must have length"):
        circuits.PauliChannel({"XY": 0.1, "X": 0.1})
    with pytest.raises(ValueError, match="invalid characters"):
        circuits.PauliChannel({"W": 0.1})
    with pytest.raises(ValueError, match="Identity string"):
        circuits.PauliChannel({"II": 0.1})
    with pytest.raises(ValueError, match="not in \\[0, 1\\]"):
        circuits.PauliChannel({"X": 1.5})
    with pytest.raises(ValueError, match="Sum of Pauli channel"):
        circuits.PauliChannel({"X": 0.7, "Y": 0.7})
    with pytest.raises(ValueError, match="num_qubits=0"):
        circuits.PauliChannel.depolarizing(0, 0.1)
    with pytest.raises(ValueError, match="not in \\[0, 1\\]"):
        circuits.PauliChannel.depolarizing(2, 1.5)
    with pytest.raises(ValueError, match="disagrees with Pauli string length"):
        circuits.PauliChannel({"XY": 0.1}, num_qubits=3)
    with pytest.raises(ValueError, match="must be >= 0"):
        circuits.PauliChannel({}, num_qubits=-1)

    # An empty channel with a nontrivial arity is a distinct object and reproduces via repr.
    empty_3q = circuits.PauliChannel({}, num_qubits=3)
    assert empty_3q.num_qubits == 3
    assert repr(empty_3q) == "PauliChannel({}, num_qubits=3)"
    assert empty_3q != circuits.PauliChannel({})


def test_pauli_channel_drops_zeros() -> None:
    """PauliChannel drops zero-prob entries but preserves arity."""
    ch = circuits.PauliChannel({"XI": 0.0, "IX": 0.05, "XX": 0.05})
    assert list(ch.probabilities.keys()) == ["IX", "XX"]
    # An all-zero PauliChannel drops its entries but keeps the arity derived from the strings.
    all_zero = circuits.PauliChannel({"XY": 0.0})
    assert all_zero.num_qubits == 2 and dict(all_zero.probabilities) == {} and not bool(all_zero)


def test_pauli_channel_hashable() -> None:
    """PauliChannel is hashable (equal channels hash equal), regardless of insertion order."""
    ch1 = circuits.PauliChannel({"XI": 0.1, "IX": 0.2})
    ch2 = circuits.PauliChannel({"IX": 0.2, "XI": 0.1})  # same channel, different insertion order
    assert ch1 == ch2 and hash(ch1) == hash(ch2)
    # Usable as a dict key / set member
    assert {ch1: "a", ch2: "b"} == {ch1: "b"}


def test_pauli_channel_bias_and_probability_boundaries() -> None:
    """Tiny bias is not collapsed, and Stim-invalid sums are rejected."""
    biased = circuits.PauliChannel({"X": 1e-11, "Y": 5e-10, "Z": 9e-10})
    assert "PAULI_CHANNEL_1" in str(biased.to_circuit())
    with pytest.raises(ValueError, match="Sum of Pauli"):
        circuits.PauliChannel({"X": 0.5, "Y": 0.5 + 2e-15})


def test_pauli_channel_conditioned_on() -> None:
    """PauliChannel.conditioned_on returns the identity-on-immune-qubits sub-channel."""

    channel = circuits.PauliChannel({"XYZ": 0.01, "XIZ": 0.02, "IZI": 0.03, "XII": 0.04})

    # Marginalize position 1 (middle): keep strings with I at pos 1.  Arity is preserved, so the
    # surviving strings still have an I at position 1.
    assert channel.conditioned_on([1]) == circuits.PauliChannel({"XIZ": 0.02, "XII": 0.04})

    # Marginalize positions 0 and 2: only "IZI" has I at both.
    assert channel.conditioned_on([0, 2]) == circuits.PauliChannel({"IZI": 0.03})

    # Empty index list is a no-op (returns an equal channel).
    assert channel.conditioned_on([]) == channel

    # Marginalizing every position yields an empty channel on the same number of qubits.
    empty_3q = channel.conditioned_on([0, 1, 2])
    assert empty_3q == circuits.PauliChannel({}, num_qubits=3)
    assert empty_3q.num_qubits == 3
    assert not bool(empty_3q)

    # If nothing survives, the result is empty but arity is preserved.
    all_non_id = circuits.PauliChannel({"XY": 0.1, "YX": 0.1})
    assert all_non_id.conditioned_on([0]) == circuits.PauliChannel({}, num_qubits=2)

    # Out-of-range indices raise.
    with pytest.raises(ValueError, match="not in"):
        channel.conditioned_on([3])
    with pytest.raises(ValueError, match="not in"):
        channel.conditioned_on([-1])


def test_pauli_channel_to_circuit() -> None:
    """PauliChannel.to_circuit emits noise, optionally appending to a given circuit and qubits."""

    channel = circuits.PauliChannel({"X": 0.1, "Z": 0.2})

    # With no arguments, a fresh circuit is created acting on range(num_qubits).
    expected = stim.Circuit()
    expected.append("PAULI_CHANNEL_1", [0], [0.1, 0.0, 0.2])
    assert channel.to_circuit() == expected

    # Explicit qubits are used as the targets.
    assert channel.to_circuit(qubits=[3]) == stim.Circuit("PAULI_CHANNEL_1(0.1, 0, 0.2) 3")

    # An existing circuit is appended to and returned (same object).
    circuit = stim.Circuit("H 0")
    returned = channel.to_circuit(append_to=circuit, qubits=[2])
    assert returned is circuit
    assert circuit == stim.Circuit("H 0\nPAULI_CHANNEL_1(0.1, 0, 0.2) 2")

    # A tag is applied to the emitted instruction.
    tagged = channel.to_circuit(qubits=[0], tag="foo")
    assert tagged == stim.Circuit("PAULI_CHANNEL_1[foo](0.1, 0, 0.2) 0")

    # simplify=False always emits a CORRELATED_ERROR / ELSE_CORRELATED_ERROR chain instead of a
    # native PAULI_CHANNEL form, even for a 1-qubit channel that would otherwise collapse.
    expected = stim.Circuit()
    expected.append("CORRELATED_ERROR", [stim.target_x(0)], [0.1])
    expected.append("ELSE_CORRELATED_ERROR", [stim.target_z(0)], [0.2 / 0.9])
    assert channel.to_circuit(simplify=False) == expected

    # A two-qubit depolarizing channel collapses to a native DEPOLARIZE2.
    depol2 = circuits.PauliChannel.depolarizing(2, 0.15)
    assert depol2.to_circuit() == stim.Circuit("DEPOLARIZE2(0.15) 0 1")

    # An empty channel emits nothing, returning the (possibly provided) circuit unchanged.
    assert circuits.PauliChannel({}, num_qubits=3).to_circuit() == stim.Circuit()
    prefilled = stim.Circuit("H 0")
    assert circuits.PauliChannel({}).to_circuit(append_to=prefilled) is prefilled
    assert prefilled == stim.Circuit("H 0")

    # Raise an error when provided qubit count that disagrees with the channel arity.
    with pytest.raises(ValueError, match="Provided 2 qubits for a 1-qubit channel"):
        channel.to_circuit(qubits=[0, 1])


def test_pauli_channel_1_shortcut() -> None:
    """One-qubit Pauli channels are emitted with Stim's most specific instruction."""
    assert circuits.PauliChannel.depolarizing(1, 0.3).to_circuit() == stim.Circuit(
        "DEPOLARIZE1(0.3) 0"
    )
    assert circuits.PauliChannel({"Z": 0.1}).to_circuit() == stim.Circuit("Z_ERROR(0.1) 0")


def test_high_probability_pauli_channels_remain_dem_analyzable() -> None:
    """Avoid Stim's over-mixing rejection when simplifying uniform channels."""
    channel_1 = circuits.PauliChannel.depolarizing(1, 0.8)
    circuit_1 = channel_1.to_circuit() + stim.Circuit("M 0\nDETECTOR rec[-1]")
    assert "PAULI_CHANNEL_1" in str(channel_1.to_circuit())
    circuit_1.detector_error_model(approximate_disjoint_errors=True)

    channel_2 = circuits.PauliChannel.depolarizing(2, 0.95)
    circuit_2 = channel_2.to_circuit() + stim.Circuit("M 0 1\nDETECTOR rec[-1]")
    assert "PAULI_CHANNEL_2" in str(channel_2.to_circuit())
    circuit_2.detector_error_model(approximate_disjoint_errors=True)


def test_pauli_channel_float_drift_clamped() -> None:
    """Chain emission clamps and short-circuits when FP drift makes remaining <= prob."""
    # 3-qubit channel with keys chosen so lex order gives probs {0.1, 0.2, 0.3, 0.4}.  Sequential
    # subtraction leaves remaining = 0.39999999999999997 < 0.4 at the final step, so the last
    # ELSE_CORRELATED_ERROR must clamp to 1.0 (rather than emit a value > 1.0) and stop.
    channel = circuits.PauliChannel({"XXX": 0.1, "XXY": 0.2, "XXZ": 0.3, "YYY": 0.4})
    expected = stim.Circuit("""
        CORRELATED_ERROR(0.1) X0 X1 X2
        ELSE_CORRELATED_ERROR(0.2222222222222222) X0 X1 Y2
        ELSE_CORRELATED_ERROR(0.42857142857142855) X0 X1 Z2
        ELSE_CORRELATED_ERROR(1.0) Y0 Y1 Y2
    """)
    assert circuits_are_equivalent(expected, channel.to_circuit())


def test_pauli_channel_sequence() -> None:
    """PauliChannelSequence emits independent errors and skips PauliChannel's normalization.

    Only sequence-specific behavior is checked here; the storage / validation / dunders shared with
    PauliChannel via AbstractPauliChannel are covered by ``test_pauli_channel_class``.
    """

    # One independent CORRELATED_ERROR per entry (no ELSE chain / PAULI_CHANNEL simplification) and,
    # unlike PauliChannel, the probabilities need not sum to <= 1.
    seq = circuits.PauliChannelSequence({"XX": 0.7, "ZZ": 0.7})
    assert seq.to_circuit() == stim.Circuit(
        "CORRELATED_ERROR(0.7) X0 X1\nCORRELATED_ERROR(0.7) Z0 Z1"
    )

    # A PauliChannel and a PauliChannelSequence with identical probabilities are unequal.
    assert circuits.PauliChannelSequence({"X": 0.1}) != circuits.PauliChannel({"X": 0.1})
