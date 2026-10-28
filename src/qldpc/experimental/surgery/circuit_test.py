# SPDX-License-Identifier: Apache-2.0

"""Tests for src/qldpc/experimental/surgery/circuit.py (single + joint PPM)."""

from __future__ import annotations

import collections

import numpy as np
import pytest
import stim

from qldpc import codes
from qldpc.objects import Pauli, PauliXZ

from .conftest import (
    _webster_x_bar_operator,
    build_generalised_bicycle_code,
    load_webster_seed_set,
)


def test_single_ppm_uses_stored_merged_checks_as_source_of_truth() -> None:
    """Public merged-check fields cannot be silently replaced by reconstructed matrices."""
    import dataclasses

    from qldpc.experimental.surgery.circuit import build_single_ppm_circuit
    from qldpc.experimental.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    logical = np.asarray(code.get_logical_ops(Pauli.X)[0], dtype=np.uint8)
    gadget = build_gadget(code, logical, basis=Pauli.X)
    modified = dataclasses.replace(gadget, HX_merged=gadget.HX_merged[:-1])

    with pytest.raises(ValueError, match=r"logical loss 0"):
        build_single_ppm_circuit(modified, rounds=1)


def test_explicit_intracode_structure_uses_code_equality_not_identity() -> None:
    """One shared data register may be described by equivalent code objects."""
    from qldpc.experimental.surgery import build_bridge, build_gadget
    from qldpc.experimental.surgery.circuit import _stitch_intracode

    code_l = codes.SteaneCode()
    code_r = codes.SteaneCode()
    logical_l = np.asarray(code_l.get_logical_ops(Pauli.X)[0], dtype=np.uint8)
    logical_r = np.asarray(code_r.get_logical_ops(Pauli.X)[0], dtype=np.uint8)
    gadget_l = build_gadget(code_l, logical_l, basis=Pauli.X)
    gadget_r = build_gadget(code_r, logical_r, basis=Pauli.X)

    merged = _stitch_intracode(gadget_l, gadget_r, build_bridge(gadget_l, gadget_r))
    shared_gadget_r = build_gadget(code_l, logical_l, basis=Pauli.X)
    shared_merged = _stitch_intracode(
        gadget_l, shared_gadget_r, build_bridge(gadget_l, shared_gadget_r)
    )
    assert np.array_equal(merged.matrix_x, shared_merged.matrix_x)
    assert np.array_equal(merged.matrix_z, shared_merged.matrix_z)

    permutation = np.array([1, 0, 2, 3, 4, 5, 6])
    permuted_code = codes.CSSCode(
        np.asarray(code_l.matrix_x)[:, permutation],
        np.asarray(code_l.matrix_z)[:, permutation],
    )
    permuted_logical = np.asarray(code_l.get_logical_ops(Pauli.X)[0], dtype=np.uint8)[permutation]
    permuted_gadget = build_gadget(permuted_code, permuted_logical, basis=Pauli.X)
    assert permuted_code.field is code_l.field
    assert permuted_code.matrix_x.shape == code_l.matrix_x.shape
    assert permuted_code.num_qudits == code_l.num_qudits
    assert permuted_code != code_l
    with pytest.raises(ValueError, match="structurally identical"):
        _stitch_intracode(gadget_l, permuted_gadget, build_bridge(gadget_l, permuted_gadget))


def test_single_ppm_rejects_a_reducible_logical_until_boosted() -> None:
    """A circuit cannot silently fix both factors of a requested logical product."""
    from qldpc.experimental.surgery import boost_gadget
    from qldpc.experimental.surgery.circuit import build_single_ppm_circuit
    from qldpc.experimental.surgery.gadget import build_gadget

    left, right = codes.SteaneCode(), codes.SteaneCode()
    code = codes.CSSCode.stack([left, right])
    logical_product = np.concatenate(
        [
            np.asarray(left.get_logical_ops(Pauli.X)[0]).astype(np.uint8),
            np.asarray(right.get_logical_ops(Pauli.X)[0]).astype(np.uint8),
        ]
    )
    bare = build_gadget(code, logical_product, basis=Pauli.X)
    with pytest.raises(ValueError, match=r"logical loss 2"):
        build_single_ppm_circuit(bare, rounds=1)

    boosted = boost_gadget(bare, method="combinatorial", target=1.0, seed=0)
    assert isinstance(build_single_ppm_circuit(boosted, rounds=1), stim.Circuit)


def test_single_ppm_accepts_a_logical_times_a_stabilizer() -> None:
    """Equivalent representatives do not trigger the reducible-logical guard."""
    from qldpc.experimental.surgery.circuit import build_single_ppm_circuit
    from qldpc.experimental.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    logical = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    stabilizer = np.asarray(code.matrix_x[0]).astype(np.uint8)
    gadget = build_gadget(code, logical ^ stabilizer, basis=Pauli.X)
    assert isinstance(build_single_ppm_circuit(gadget, rounds=1), stim.Circuit)


@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
def test_build_single_ppm_circuit_noiseless_no_detector_fires(basis: PauliXZ) -> None:
    """Noiseless: NO detector fires (including the new final detectors).

    The total detector count must equal: round-1 reliable + (rounds-1)*all_checks + final reliable.
    Under noiseless conditions all of them must remain silent.
    """
    from qldpc.experimental.surgery.circuit import build_single_ppm_circuit
    from qldpc.experimental.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    op = code.get_logical_ops(Pauli.X)[0] if basis is Pauli.X else code.get_logical_ops(Pauli.Z)[0]
    op_arr = np.asarray(op).astype(np.uint8)
    g = build_gadget(code, op_arr, basis=basis)
    circuit = build_single_ppm_circuit(g, rounds=3, noise_model=None)
    sampler = circuit.compile_detector_sampler()
    dets, _ = sampler.sample(shots=64, separate_observables=True)
    assert not dets.any(), (
        f"basis={basis}: {dets.sum()} detector fires noiselessly across {dets.shape[0]} shots"
    )


@pytest.mark.parametrize("layout", ["single", "joint"])
def test_ppm_logical_error_rate_grows_with_p(layout: str, pytestconfig: pytest.Config) -> None:
    """Circuit noise flips the decoded PPM outcome far more often at p=3e-2 than at p=1e-3.

    The decoded logical error rate is about 1-4% at p=1e-3 and 25-50% at p=3e-2 for a Steane X̄ (or
    X̄X̄) measurement with one round, so the thresholds below leave wide margins at 500 shots.
    """
    from qldpc import decoders
    from qldpc.circuits import DepolarizingNoiseModel
    from qldpc.experimental.surgery import build_bridge, build_gadget
    from qldpc.experimental.surgery.circuit import (
        build_joint_ppm_circuit,
        build_single_ppm_circuit,
    )

    seed = pytestconfig.getoption("randomly_seed")
    code = codes.SteaneCode()
    logical = np.asarray(code.get_logical_ops(Pauli.X)[0], dtype=np.uint8)
    g_l = build_gadget(code, logical, basis=Pauli.X)
    g_r = build_gadget(codes.SteaneCode(), logical, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)

    raw_rates, decoded_rates = [], []
    for p in (1e-3, 3e-2):
        noise_model = DepolarizingNoiseModel(p)
        if layout == "single":
            circuit = build_single_ppm_circuit(g_l, rounds=1, noise_model=noise_model)
        else:
            circuit, _ = build_joint_ppm_circuit(
                g_l, g_r, bridge, rounds=1, noise_model=noise_model
            )
        sampler = circuit.compile_detector_sampler(seed=seed)
        detectors, observables = sampler.sample(500, separate_observables=True)
        decoder = decoders.bp_osd().build_observable_decoder(circuit.detector_error_model())
        assert isinstance(decoder, decoders.BatchObservableDecoder)
        predictions = np.asarray(decoder.decode_observables_batch(detectors))
        raw_rates.append(observables[:, 0].mean())
        decoded_rates.append((predictions[:, 0] != observables[:, 0]).mean())

    assert raw_rates[1] > 0.2, f"noise rarely reaches the PPM outcome: {raw_rates}"
    assert decoded_rates[0] < 0.1, f"decoded rates: {decoded_rates}"
    assert decoded_rates[1] > 0.2, f"decoded rates: {decoded_rates}"


def test_single_ppm_decoding_corrects_errors(pytestconfig: pytest.Config) -> None:
    """With repeated rounds, decoding lowers the logical error rate of a single PPM.

    At p=1e-3 with three rounds, a Steane X̄ measurement flips its raw outcome in about 3% of shots,
    while the decoded outcome is wrong in under 1% of shots.
    """
    from qldpc import decoders
    from qldpc.circuits import DepolarizingNoiseModel
    from qldpc.experimental.surgery import build_gadget
    from qldpc.experimental.surgery.circuit import build_single_ppm_circuit

    code = codes.SteaneCode()
    logical = np.asarray(code.get_logical_ops(Pauli.X)[0], dtype=np.uint8)
    gadget = build_gadget(code, logical, basis=Pauli.X)
    circuit = build_single_ppm_circuit(gadget, rounds=3, noise_model=DepolarizingNoiseModel(1e-3))
    sampler = circuit.compile_detector_sampler(seed=pytestconfig.getoption("randomly_seed"))
    detectors, observables = sampler.sample(2000, separate_observables=True)
    decoder = decoders.bp_osd().build_observable_decoder(circuit.detector_error_model())
    assert isinstance(decoder, decoders.BatchObservableDecoder)
    predictions = np.asarray(decoder.decode_observables_batch(detectors))
    raw_rate = observables[:, 0].mean()
    decoded_rate = (predictions[:, 0] != observables[:, 0]).mean()
    assert decoded_rate < 0.015 < raw_rate, f"raw={raw_rate}, decoded={decoded_rate}"


@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
def test_stitch_intercode_both_bases_commute_and_singletons_excluded(basis: PauliXZ) -> None:
    """Inter-code Steane × Steane: commuting checks, k_l + k_r - 1, and only the joint fixed."""
    import galois

    from qldpc.experimental.surgery.bridge import build_bridge
    from qldpc.experimental.surgery.circuit import _stitch_intercode
    from qldpc.experimental.surgery.gadget import build_gadget

    GF2 = galois.GF2
    code = codes.SteaneCode()
    if basis is Pauli.X:
        x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    else:
        x = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g_l = build_gadget(code, x, basis=basis)
    g_r = build_gadget(codes.SteaneCode(), x, basis=basis)
    bridge = build_bridge(g_l, g_r)
    merged = _stitch_intercode(g_l, g_r, bridge)
    HX = np.asarray(merged.matrix_x).astype(np.int_)
    HZ = np.asarray(merged.matrix_z).astype(np.int_)
    product = (HX @ HZ.T) % 2
    assert np.array_equal(product, np.zeros_like(product))
    assert merged.dimension == 2 * code.dimension - 1
    # The joint stabilizer (x_l, x_r, 0, ...) is in the rowspan of the check matrix that contains it
    # (HX for basis=X, HZ for Z), whereas the singletons (x_l, 0, ...) and (0, x_r, ...) are not.
    H_joint = HX if basis is Pauli.X else HZ
    n_l = code.num_qudits
    base_rank = np.linalg.matrix_rank(GF2(H_joint.tolist()))
    joint = np.zeros(H_joint.shape[1], dtype=np.int_)
    joint[: 2 * n_l] = np.concatenate([x, x])
    augmented = np.vstack([H_joint, joint.reshape(1, -1)])
    assert np.linalg.matrix_rank(GF2(augmented.tolist())) == base_rank
    for which in ("left", "right"):
        single = np.zeros(H_joint.shape[1], dtype=np.int_)
        if which == "left":
            single[:n_l] = x
        else:
            single[n_l : 2 * n_l] = x
        augmented = np.vstack([H_joint, single.reshape(1, -1)])
        assert np.linalg.matrix_rank(GF2(augmented.tolist())) == base_rank + 1, which


def test_joint_ppm_of_different_codes() -> None:
    """Inter-code Webster GB code 0 × Steane joint PPM has dimension k_l + k_r - 1 = 10.

    The other inter-code tests join two copies of one code, so they cannot tell the left code's
    sizes from the right code's.
    """
    from qldpc.experimental.surgery.bridge import build_bridge
    from qldpc.experimental.surgery.circuit import build_joint_ppm_circuit
    from qldpc.experimental.surgery.gadget import build_gadget

    data = load_webster_seed_set(0)
    webster = build_generalised_bicycle_code(data["l"], data["A"], data["B"])
    z_webster = _webster_x_bar_operator(data, "Z_bar_1", pauli_type="Z")
    steane = codes.SteaneCode()
    z_steane = np.asarray(steane.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g_l = build_gadget(webster, z_webster, basis=Pauli.Z)
    g_r = build_gadget(steane, z_steane, basis=Pauli.Z)
    _, joint_code = build_joint_ppm_circuit(
        g_l, g_r, build_bridge(g_l, g_r), rounds=1, noise_model=None
    )
    assert joint_code.dimension == webster.dimension + steane.dimension - 1


@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
def test_stitch_intracode_both_bases_commute(basis: PauliXZ) -> None:
    """Intra-code commutation for both bases.  Use a Webster code with 2 distinct logicals.

    Steane intra-code (k=1) yields the degenerate joint X̄·X̄ = I case.
    """
    from qldpc.experimental.surgery.bridge import build_bridge
    from qldpc.experimental.surgery.circuit import _stitch_intracode
    from qldpc.experimental.surgery.gadget import (
        build_gadget,
    )

    data = load_webster_seed_set(0)
    code = build_generalised_bicycle_code(data["l"], data["A"], data["B"])
    if basis is Pauli.X:
        x1 = _webster_x_bar_operator(data, "X_bar_1")
        x2 = _webster_x_bar_operator(data, "X_bar_k2p1")
    else:
        from .conftest import _webster_z_bar_operator

        x1 = _webster_z_bar_operator(data, "Z_bar_1")
        x2 = _webster_z_bar_operator(data, "Z_bar_k2p1")
    g_l = build_gadget(code, x1, basis=basis)
    g_r = build_gadget(code, x2, basis=basis)
    bridge = build_bridge(g_l, g_r)
    merged = _stitch_intracode(g_l, g_r, bridge)
    HX = np.asarray(merged.matrix_x).astype(np.int_)
    HZ = np.asarray(merged.matrix_z).astype(np.int_)
    product = (HX @ HZ.T) % 2
    assert np.array_equal(product, np.zeros_like(product))
    assert merged.dimension == code.dimension - 1


def test_build_joint_ppm_circuit_meas_check_ids_no_UB() -> None:
    """build_joint_ppm_circuit's noiseless first sample has zero detectors firing."""
    from qldpc.experimental.surgery.bridge import build_bridge
    from qldpc.experimental.surgery.circuit import build_joint_ppm_circuit
    from qldpc.experimental.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g_l = build_gadget(code, x, basis=Pauli.X)
    g_r = build_gadget(codes.SteaneCode(), x, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)
    circuit, _merged = build_joint_ppm_circuit(g_l, g_r, bridge, rounds=2)
    # noiseless: all detectors must NOT fire on first sample
    sampler = circuit.compile_detector_sampler()
    dets, _ = sampler.sample(8, separate_observables=True)
    assert dets.sum() == 0


def test_joint_ppm_rejects_a_bridge_for_different_same_shape_gadgets() -> None:
    """Shape-compatible gadgets cannot substitute for the operators stored in a bridge."""
    from qldpc.experimental.surgery.bridge import build_bridge
    from qldpc.experimental.surgery.circuit import build_joint_ppm_circuit
    from qldpc.experimental.surgery.gadget import build_gadget

    data = load_webster_seed_set(0)
    code = build_generalised_bicycle_code(data["l"], data["A"], data["B"])
    g_a = build_gadget(code, _webster_x_bar_operator(data, "X_bar_1"), basis=Pauli.X)
    g_b = build_gadget(code, _webster_x_bar_operator(data, "X_bar_k2p1"), basis=Pauli.X)
    bridge = build_bridge(g_a, g_a)

    with pytest.raises(ValueError, match=r"bridge does not match supplied g_r"):
        build_joint_ppm_circuit(g_a, g_b, bridge, rounds=1)


def test_joint_ppm_rejects_a_gadget_boosted_after_bridge_construction() -> None:
    """A bridge must be rebuilt after either source gadget changes."""
    from qldpc.experimental.surgery import boost_gadget
    from qldpc.experimental.surgery.bridge import build_bridge
    from qldpc.experimental.surgery.circuit import build_joint_ppm_circuit
    from qldpc.experimental.surgery.gadget import build_gadget

    data = load_webster_seed_set(0)
    code = build_generalised_bicycle_code(data["l"], data["A"], data["B"])
    g_l = build_gadget(code, _webster_x_bar_operator(data, "X_bar_1"), basis=Pauli.X)
    g_r = build_gadget(code, _webster_x_bar_operator(data, "X_bar_k2p1"), basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)
    boosted_l = boost_gadget(g_l, method="combinatorial", target=2.0, seed=0)

    with pytest.raises(ValueError, match=r"bridge does not match supplied g_l"):
        build_joint_ppm_circuit(boosted_l, g_r, bridge, rounds=1)


def test_joint_ppm_rejects_an_identity_product() -> None:
    """An intracode L̄·L̄ bridge fixes no logical degree of freedom."""
    from qldpc.experimental.surgery.bridge import build_bridge
    from qldpc.experimental.surgery.circuit import build_joint_ppm_circuit
    from qldpc.experimental.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    logical = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    gadget = build_gadget(code, logical, basis=Pauli.X)

    with pytest.raises(ValueError, match=r"logical loss 0"):
        build_joint_ppm_circuit(gadget, gadget, build_bridge(gadget, gadget), rounds=1)


@pytest.mark.parametrize("code_index", [0, 1, 2, 3])
def test_joint_xx_in_stabilizer_on_webster_intracode(code_index: int) -> None:
    """Webster BB codes 0..3 intra-code: (x_1, x_2 padded, 0...) is in rowspan(H_X^merged).

    Pins the SkipTree adapter construction across the full Webster Table I code family rather than
    just code 0.
    """
    import galois

    from qldpc.experimental.surgery.bridge import build_bridge
    from qldpc.experimental.surgery.circuit import _stitch_intracode
    from qldpc.experimental.surgery.gadget import (
        build_gadget,
    )

    GF2 = galois.GF2
    data = load_webster_seed_set(code_index)
    code = build_generalised_bicycle_code(data["l"], data["A"], data["B"])
    x1 = _webster_x_bar_operator(data, "X_bar_1")
    x2 = _webster_x_bar_operator(data, "X_bar_k2p1")
    g_l = build_gadget(code, x1, basis=Pauli.X)
    g_r = build_gadget(code, x2, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)
    merged = _stitch_intracode(g_l, g_r, bridge)
    HX = np.asarray(merged.matrix_x).astype(np.int_)
    joint = np.zeros(HX.shape[1], dtype=np.int_)
    n = code.num_qudits
    joint[:n] = (x1 + x2) % 2
    augmented = np.vstack([HX, joint.reshape(1, -1)])
    assert np.linalg.matrix_rank(GF2(HX.tolist())) == np.linalg.matrix_rank(GF2(augmented.tolist()))


def test_build_joint_ppm_circuit_intracode_noiseless_observables_zero() -> None:
    """Intra-code Webster joint X̄_1·X̄_{k/2+1}: noiseless detectors + observables = 0.

    Covers the intra-code path on a real Webster fixture.
    """
    from qldpc.experimental.surgery.bridge import build_bridge
    from qldpc.experimental.surgery.circuit import build_joint_ppm_circuit
    from qldpc.experimental.surgery.gadget import (
        build_gadget,
    )

    data = load_webster_seed_set(0)
    code = build_generalised_bicycle_code(data["l"], data["A"], data["B"])
    x1 = _webster_x_bar_operator(data, "X_bar_1")
    x2 = _webster_x_bar_operator(data, "X_bar_k2p1")
    g_l = build_gadget(code, x1, basis=Pauli.X)
    g_r = build_gadget(code, x2, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)
    circuit, _ = build_joint_ppm_circuit(g_l, g_r, bridge, rounds=2)
    sampler = circuit.compile_detector_sampler()
    dets, obs = sampler.sample(8, separate_observables=True)
    assert dets.sum() == 0, "noiseless intra-code: detectors should not fire"
    assert obs.sum() == 0, "noiseless intra-code: observables should be 0"


def test_single_ppm_data_init_default_matches_pre_kwarg() -> None:
    """build_single_ppm_circuit(g, rounds=3) ≡ data_init='+' for basis=X."""
    from qldpc.experimental.surgery.circuit import build_single_ppm_circuit
    from qldpc.experimental.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    c_no_kwarg = build_single_ppm_circuit(g, rounds=3, noise_model=None)
    c_plus = build_single_ppm_circuit(g, rounds=3, noise_model=None, data_init="+")
    assert str(c_no_kwarg) == str(c_plus), "data_init='+' broadcast must match default for basis=X"


def test_single_ppm_data_init_zero_random_outcome(pytestconfig: pytest.Config) -> None:
    """data_init='0' on basis=X gadget → logical |0⟩, obs0 50% flip, obs0 ≡ obs1."""
    from qldpc.experimental.surgery.circuit import build_single_ppm_circuit
    from qldpc.experimental.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    circuit = build_single_ppm_circuit(g, rounds=3, noise_model=None, data_init="0")
    sampler = circuit.compile_detector_sampler(seed=pytestconfig.getoption("randomly_seed"))
    detectors, observables = sampler.sample(shots=4000, separate_observables=True)
    obs0, obs1 = observables[:, 0], observables[:, 1]
    rate0, rate1 = float(obs0.mean()), float(obs1.mean())
    agree = float((obs0 == obs1).mean())
    assert 0.40 < rate0 < 0.60, f"obs0 flip rate {rate0:.2%} not in (40%, 60%)"
    assert 0.40 < rate1 < 0.60, f"obs1 flip rate {rate1:.2%} not in (40%, 60%)"
    assert agree == 1.0, f"obs0 vs obs1 disagree on {int((1 - agree) * 4000)} of 4000 shots"
    assert not detectors.any(), "complementary-basis init must not create random detectors"
    with pytest.raises(ValueError, match="non-deterministic observables"):
        circuit.detector_error_model()


def test_joint_ppm_data_init_truth_table() -> None:
    """Joint Z̄⊗Z̄ on two Steane copies: 4 |a⟩|b⟩ inits give expected parity."""
    from qldpc.experimental.surgery.bridge import build_bridge
    from qldpc.experimental.surgery.circuit import build_joint_ppm_circuit
    from qldpc.experimental.surgery.gadget import build_gadget

    c1, c2 = codes.SteaneCode(), codes.SteaneCode()
    z1 = np.asarray(c1.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    z2 = np.asarray(c2.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g1 = build_gadget(c1, z1, basis=Pauli.Z)
    g2 = build_gadget(c2, z2, basis=Pauli.Z)
    bridge = build_bridge(g1, g2)
    n1 = c1.num_qudits
    cases = [
        ("0" * n1 + "0" * n1, 0),
        ("0" * n1 + "1" * n1, 1),
        ("1" * n1 + "0" * n1, 1),
        ("1" * n1 + "1" * n1, 0),
    ]
    for data_init, expected in cases:
        circuit, _ = build_joint_ppm_circuit(
            g1,
            g2,
            bridge,
            rounds=3,
            noise_model=None,
            data_init=data_init,
        )
        sampler = circuit.compile_sampler()
        raw = sampler.sample(shots=200).astype(np.uint8)
        n_meas = raw.shape[1]
        obs_lines = [ln for ln in str(circuit).splitlines() if ln.startswith("OBSERVABLE_INCLUDE")]
        offsets = [int(t.strip("rec[]")) for t in obs_lines[0].split() if t.startswith("rec[")]
        meas_idx = [n_meas + off for off in offsets]
        obs0 = np.bitwise_xor.reduce(raw[:, meas_idx], axis=1)
        rate = float(obs0.mean())
        assert rate == float(expected), (
            f"data_init={data_init!r} gave obs0 rate {rate:.3f}, expected {expected}"
        )


def test_joint_ppm_data_init_superposition(pytestconfig: pytest.Config) -> None:
    """c1 |0⟩ × c2 |+⟩: Z̄_2 random → obs0 ~50%, obs0 ≡ obs1 every shot."""
    from qldpc.experimental.surgery.bridge import build_bridge
    from qldpc.experimental.surgery.circuit import build_joint_ppm_circuit
    from qldpc.experimental.surgery.gadget import build_gadget

    seed = pytestconfig.getoption("randomly_seed")
    c1, c2 = codes.SteaneCode(), codes.SteaneCode()
    z1 = np.asarray(c1.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    z2 = np.asarray(c2.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g1 = build_gadget(c1, z1, basis=Pauli.Z)
    g2 = build_gadget(c2, z2, basis=Pauli.Z)
    bridge = build_bridge(g1, g2)
    n = c1.num_qudits
    circuit, _ = build_joint_ppm_circuit(
        g1,
        g2,
        bridge,
        rounds=3,
        noise_model=None,
        data_init="0" * n + "+" * n,
    )
    detectors, _ = circuit.compile_detector_sampler(seed=seed).sample(
        shots=100,
        separate_observables=True,
    )
    assert not detectors.any(), "superposition init must not create random detectors"
    with pytest.raises(ValueError, match="non-deterministic observables"):
        circuit.detector_error_model()
    sampler = circuit.compile_sampler(seed=seed)
    raw = sampler.sample(shots=1000).astype(np.uint8)
    n_meas = raw.shape[1]
    obs_lines = [ln for ln in str(circuit).splitlines() if ln.startswith("OBSERVABLE_INCLUDE")]
    cols = []
    for line in obs_lines:
        offsets = [int(t.strip("rec[]")) for t in line.split() if t.startswith("rec[")]
        meas_idx = [n_meas + off for off in offsets]
        cols.append(np.bitwise_xor.reduce(raw[:, meas_idx], axis=1))
    obs = np.stack(cols, axis=1)
    rate0 = float(obs[:, 0].mean())
    rate1 = float(obs[:, 1].mean())
    agree = float((obs[:, 0] == obs[:, 1]).mean())
    assert 0.40 < rate0 < 0.60, f"obs0 rate {rate0:.2%} not random"
    assert 0.40 < rate1 < 0.60, f"obs1 rate {rate1:.2%} not random"
    assert agree == 1.0, f"obs0 vs obs1 disagree on {int((1 - agree) * 1000)} of 1000 shots"


def test_joint_ppm_data_init_tuple_per_qubit_entry() -> None:
    """Each tuple entry may be per-qubit (length n_code), not only len-1 broadcast."""
    from qldpc.experimental.surgery.bridge import build_bridge
    from qldpc.experimental.surgery.circuit import build_joint_ppm_circuit
    from qldpc.experimental.surgery.gadget import build_gadget

    c1, c2 = codes.SteaneCode(), codes.SteaneCode()
    z1 = np.asarray(c1.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    z2 = np.asarray(c2.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g1 = build_gadget(c1, z1, basis=Pauli.Z)
    g2 = build_gadget(c2, z2, basis=Pauli.Z)
    bridge = build_bridge(g1, g2)
    n = c1.num_qudits
    spec_l = "0011010"
    spec_r = "+"
    c_tuple, _ = build_joint_ppm_circuit(
        g1,
        g2,
        bridge,
        rounds=3,
        noise_model=None,
        data_init=(spec_l, spec_r),
    )
    c_string, _ = build_joint_ppm_circuit(
        g1,
        g2,
        bridge,
        rounds=3,
        noise_model=None,
        data_init=spec_l + "+" * n,
    )
    assert str(c_tuple) == str(c_string)


@pytest.mark.parametrize(
    "bad_init,error_substr",
    [
        (("0",), "must have 2 entries"),
        (("00", "+"), r"data_init\[0\]"),
        (("0", "++"), r"data_init\[1\]"),
        ((0, "+"), "must be str"),
    ],
)
def test_joint_ppm_data_init_tuple_validation(bad_init: object, error_substr: str) -> None:
    from qldpc.experimental.surgery.bridge import build_bridge
    from qldpc.experimental.surgery.circuit import build_joint_ppm_circuit
    from qldpc.experimental.surgery.gadget import build_gadget

    c1, c2 = codes.SteaneCode(), codes.SteaneCode()
    z1 = np.asarray(c1.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    z2 = np.asarray(c2.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g1 = build_gadget(c1, z1, basis=Pauli.Z)
    g2 = build_gadget(c2, z2, basis=Pauli.Z)
    bridge = build_bridge(g1, g2)
    expected = TypeError if "must be str" in error_substr else ValueError
    with pytest.raises(expected, match=error_substr):
        build_joint_ppm_circuit(
            g1,
            g2,
            bridge,
            rounds=3,
            noise_model=None,
            data_init=bad_init,  # type: ignore[arg-type]
        )


def test_joint_ppm_data_init_tuple_rejects_intracode() -> None:
    """Tuple form is invalid for intracode joint PPM (single data set)."""
    from qldpc.experimental.surgery.bridge import build_bridge
    from qldpc.experimental.surgery.circuit import build_joint_ppm_circuit
    from qldpc.experimental.surgery.gadget import build_gadget

    data = load_webster_seed_set(0)
    code = build_generalised_bicycle_code(data["l"], data["A"], data["B"])
    x1 = _webster_x_bar_operator(data, "X_bar_1")
    x2 = _webster_x_bar_operator(data, "X_bar_k2p1")
    g_l = build_gadget(code, x1, basis=Pauli.X)
    g_r = build_gadget(code, x2, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)
    assert g_l.code is g_r.code, "intracode setup precondition"
    with pytest.raises(ValueError, match="intracode joint has a single data set"):
        build_joint_ppm_circuit(
            g_l,
            g_r,
            bridge,
            rounds=3,
            noise_model=None,
            data_init=("0", "0"),
        )


@pytest.mark.parametrize(
    "bad_init,error_substr",
    [
        ("00", "does not match num data qubits"),
        ("@" * 7, "invalid chars"),
    ],
)
def test_data_init_validation(bad_init: object, error_substr: str) -> None:
    """Bad data_init raises ValueError with informative message."""
    from qldpc.experimental.surgery.circuit import build_single_ppm_circuit
    from qldpc.experimental.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    with pytest.raises(ValueError, match=error_substr):
        build_single_ppm_circuit(
            g,
            rounds=3,
            noise_model=None,
            data_init=bad_init,  # type: ignore[arg-type]
        )


def test_qubit_coords_layout_steane() -> None:
    """Steane single-PPM circuit emits QUBIT_COORDS in 6 semantic lanes.

    y=0 data (Steane ids 0..6), y=1 κ ancillas (3), y=2 data H_X ancillas (3), y=3 χ ancillas (3),
    y=4 data H_Z ancillas (3), y=5 G ancilla (1).  Ordering chosen so y is monotonic in qubit ID for
    basis=X.
    """
    from qldpc.experimental.surgery.circuit import build_single_ppm_circuit
    from qldpc.experimental.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    circuit = build_single_ppm_circuit(g, rounds=1, noise_model=None)

    coord_map = {
        qid: tuple(int(v) for v in coords)
        for qid, coords in circuit.get_final_qubit_coordinates().items()
    }

    expected = {
        # data qubits on y=0 (unchanged)
        0: (0, 0),
        1: (1, 0),
        2: (2, 0),
        3: (3, 0),
        4: (4, 0),
        5: (5, 0),
        6: (6, 0),
        # κ ancillas on y=1
        7: (0, 1),
        8: (1, 1),
        9: (2, 1),
        # data H_X ancillas on y=2
        10: (0, 2),
        11: (1, 2),
        12: (2, 2),
        # χ ancillas on y=3
        13: (0, 3),
        14: (1, 3),
        15: (2, 3),
        # data H_Z ancillas on y=4
        16: (0, 4),
        17: (1, 4),
        18: (2, 4),
        # G ancilla on y=5
        19: (0, 5),
    }
    assert coord_map == expected, f"\nexpected: {expected}\ngot:      {coord_map}"


def test_detector_coords_steane_round_1_reliable() -> None:
    """Steane single-PPM detector coordinates are (round, lane, check_id).

    Round-1 reliable for a basis=X gadget: 3 data H_X checks (lane=2) + 1 G check (lane=5).  No χ or
    data H_Z because those aren't deterministic on the protocol-default |+⟩ init.  The final readout
    re-checks the same four after one SHIFT_COORDS, so they reappear at round 1.

    Coordinate 0 is the round, which is the time index the sliding-window decoders read by default.
    Coordinate 2 is the check ancilla's qubit id, and coordinate 1 is that ancilla's QUBIT_COORDS
    lane.
    """
    from qldpc.experimental.surgery.circuit import build_single_ppm_circuit
    from qldpc.experimental.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    circuit = build_single_ppm_circuit(g, rounds=1, noise_model=None)

    detector_coords = {
        tuple(int(v) for v in coords) for coords in circuit.get_detector_coordinates().values()
    }
    reliable = {(2, 10), (2, 11), (2, 12), (5, 19)}
    expected = {(round_, lane, cid) for round_ in (0, 1) for lane, cid in reliable}
    assert detector_coords == expected, f"\nexpected: {expected}\ngot:      {detector_coords}"

    # Coordinate 1 really is the lane of the ancilla that coordinate 2 names.
    qubit_lane = {
        qid: int(coords[1]) for qid, coords in circuit.get_final_qubit_coordinates().items()
    }
    for _round, lane, check_id in detector_coords:
        assert qubit_lane[check_id] == lane, f"check {check_id} is on lane {qubit_lane[check_id]}"


@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
def test_joint_ppm_round_0_detectors(basis: PauliXZ) -> None:
    """A joint PPM's first round has detectors for data, gauge, and cycle checks.

    On the default product-state initialization, the deterministic checks in the first round are
    the data checks of the measured basis (lane 2 for X, lane 4 for Z), the bridge-augmented gauge
    checks (lane 5), and the bridge's cycle checks (lane 6, one per row of H_R).
    """
    from qldpc.experimental.surgery import build_bridge, build_gadget
    from qldpc.experimental.surgery.circuit import build_joint_ppm_circuit

    code = codes.SteaneCode()
    logical = np.asarray(code.get_logical_ops(basis)[0]).astype(np.uint8)
    g_l = build_gadget(code, logical, basis=basis)
    g_r = build_gadget(codes.SteaneCode(), logical, basis=basis)
    bridge = build_bridge(g_l, g_r)
    circuit, _ = build_joint_ppm_circuit(g_l, g_r, bridge, rounds=1, noise_model=None)

    round_0_lanes = collections.Counter(
        int(coords[1]) for coords in circuit.get_detector_coordinates().values() if coords[0] == 0
    )
    data_lane = 2 if basis is Pauli.X else 4
    expected = {
        data_lane: 2 * code.get_matrix(basis).shape[0],
        5: bridge.g_l_aug.gauge.shape[0] + bridge.g_r_aug.gauge.shape[0],
        6: bridge.H_R.shape[0],
    }
    assert round_0_lanes == collections.Counter({k: v for k, v in expected.items() if v})


def test_detector_coords_basis_z_preserves_lane_semantics() -> None:
    """basis=Z: reliable detectors use lanes 4 (H_Z) and 5 (G); the readout re-checks round 0.

    The lane map must not route G ancillas to lane 2 (data H_X) nor χ ancillas to lane 3 when the
    basis swaps their matrix slots.  The BBCode [[36, 8]] Z̄_0 gadget has a nonempty G, so lane 5 is
    populated and such a misrouting would show.
    """
    import sympy

    from qldpc.experimental.surgery.circuit import build_single_ppm_circuit
    from qldpc.experimental.surgery.gadget import build_gadget

    xs, ys = sympy.symbols("x y")
    code = codes.BBCode({xs: 3, ys: 6}, xs**3 + ys + ys**2, ys**3 + xs + xs**2)
    z_bar = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g = build_gadget(code, z_bar, basis=Pauli.Z)
    assert g.gauge.shape[0] > 0, "test premise broken: the gadget should have a nonempty G"
    circuit = build_single_ppm_circuit(g, rounds=1, noise_model=None)

    detector_coords = {
        tuple(int(v) for v in coords) for coords in circuit.get_detector_coordinates().values()
    }
    assert {lane for _round, lane, _cid in detector_coords} == {4, 5}
    by_round = [
        {(lane, cid) for round_, lane, cid in detector_coords if round_ == r} for r in (0, 1)
    ]
    assert by_round[0] == by_round[1]


@pytest.mark.parametrize("joint", [False, True])
def test_detector_round_coordinate_advances_once_per_round(joint: bool) -> None:
    """Coordinate 0 of every detector is its round, spanning 0 through ``rounds``.

    Each builder advances it inside its repeat block, after that round's gates and measurements and
    immediately before its detectors, and both share the advance before the final data readout, so
    rounds is set above 1 to reach the repeat-block site.
    """
    from qldpc.experimental.surgery.bridge import build_bridge
    from qldpc.experimental.surgery.circuit import (
        build_joint_ppm_circuit,
        build_single_ppm_circuit,
    )
    from qldpc.experimental.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g_l = build_gadget(code, x, basis=Pauli.X)
    if joint:
        g_r = build_gadget(codes.SteaneCode(), x, basis=Pauli.X)
        circuit, _merged = build_joint_ppm_circuit(g_l, g_r, build_bridge(g_l, g_r), rounds=3)
    else:
        circuit = build_single_ppm_circuit(g_l, rounds=3, noise_model=None)

    rounds_seen = {int(coords[0]) for coords in circuit.get_detector_coordinates().values()}
    assert rounds_seen == {0, 1, 2, 3}


def test_sliding_window_decoder_windows_a_surgery_circuit_by_round() -> None:
    """A window decoder handed the DEM with no time mapping of its own recovers the rounds.

    SlidingWindowDecoder's default time index is coordinate 0 of each detector, which is the reason
    the round goes there.  This exercises that end to end -- through keep_only_observable and the
    DEM conversion -- rather than reading the coordinates off the circuit.  Seven round values at
    window_size=3, stride=1 give five windows sliding one round at a time.
    """
    from qldpc.circuits.noise_model import DepolarizingNoiseModel
    from qldpc.decoders.sinter import SlidingWindowDecoder
    from qldpc.experimental.surgery.circuit import build_single_ppm_circuit, keep_only_observable
    from qldpc.experimental.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    gadget = build_gadget(code, z, basis=Pauli.Z)
    circuit = keep_only_observable(
        build_single_ppm_circuit(gadget, rounds=6, noise_model=DepolarizingNoiseModel(p=1e-3)), 0
    )
    dem = circuit.detector_error_model(decompose_errors=False)
    round_of = {det: int(coords[0]) for det, coords in dem.get_detector_coordinates().items()}

    compiled = SlidingWindowDecoder(window_size=3, stride=1).compile_decoder_for_dem(dem)

    detectors = np.arange(dem.num_detectors)
    spans = [
        sorted({round_of[det] for det in detectors[detection]})
        for detection in compiled.window_detectors
    ]
    assert spans == [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]]


def test_joint_ppm_qubit_coords_intercode_layout() -> None:
    """Intercode joint Z̄⊗Z̄ on two Steane copies: both data blocks share lane y=0.

    n_l = n_r = 7; left data on y=0 at x=0..6; right data on y=0 at x=7..13.
    """
    from qldpc.experimental.surgery.bridge import build_bridge
    from qldpc.experimental.surgery.circuit import build_joint_ppm_circuit
    from qldpc.experimental.surgery.gadget import build_gadget

    c1, c2 = codes.SteaneCode(), codes.SteaneCode()
    z1 = np.asarray(c1.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    z2 = np.asarray(c2.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g1 = build_gadget(c1, z1, basis=Pauli.Z)
    g2 = build_gadget(c2, z2, basis=Pauli.Z)
    bridge = build_bridge(g1, g2)
    circuit, _ = build_joint_ppm_circuit(
        g1,
        g2,
        bridge,
        rounds=1,
        noise_model=None,
    )

    y0 = sorted(
        int(coords[0])
        for coords in circuit.get_final_qubit_coordinates().values()
        if coords[1] == 0
    )
    assert y0 == list(range(14)), f"y=0 x positions: expected 0..13, got {y0}"


def test_joint_ppm_augmented_gauge_rows_stay_in_gauge_lane() -> None:
    """Bridge-added gauge checks use lane 5; only adapter cycle checks use lane 6.

    The detectors are deterministic, including those of the bridge-added ancillas.
    """
    from qldpc.experimental.surgery.bridge import build_bridge
    from qldpc.experimental.surgery.circuit import build_joint_ppm_circuit
    from qldpc.experimental.surgery.gadget import build_gadget

    from .conftest import _webster_z_bar_operator

    data = load_webster_seed_set(1)
    code_l = build_generalised_bicycle_code(data["l"], data["A"], data["B"])
    code_r = build_generalised_bicycle_code(data["l"], data["A"], data["B"])
    g_l = build_gadget(code_l, _webster_z_bar_operator(data, "Z_bar_1"), basis=Pauli.Z)
    g_r = build_gadget(code_r, _webster_z_bar_operator(data, "Z_bar_k2p1"), basis=Pauli.Z)
    bridge = build_bridge(g_l, g_r)
    bare_gauge_count = g_l.gauge.shape[0] + g_r.gauge.shape[0]
    augmented_gauge_count = bridge.g_l_aug.gauge.shape[0] + bridge.g_r_aug.gauge.shape[0]
    assert augmented_gauge_count > bare_gauge_count, "fixture must add a bridge gauge row"

    circuit, _joint_code = build_joint_ppm_circuit(g_l, g_r, bridge, rounds=1)
    circuit.detector_error_model()  # raises if any detector is nondeterministic
    qubit_lane = {
        qid: int(coords[1]) for qid, coords in circuit.get_final_qubit_coordinates().items()
    }
    by_lane = collections.Counter(qubit_lane.values())
    assert by_lane[5] == augmented_gauge_count
    assert by_lane[6] == 2 * bridge.width - 1

    # Each detector's lane coordinate is the lane of the check ancilla that it names.
    for _round, lane, check_id in circuit.get_detector_coordinates().values():
        assert qubit_lane[int(check_id)] == int(lane), f"check {check_id} is not on lane {lane}"


def test_logical_state_init_zero_and_plus_broadcast() -> None:
    """'0' and '+' return length-n broadcast strings — trivial CSS prep."""
    from qldpc.experimental.surgery.circuit import logical_state_init

    code = codes.SteaneCode()
    n = code.num_qudits
    assert logical_state_init(code, "0", log_idx=0) == "0" * n
    assert logical_state_init(code, "+", log_idx=0) == "+" * n


def test_logical_state_init_invalid_state_raises() -> None:
    """Anything outside {'0', '1', '+', '-'} raises ValueError."""
    from qldpc.experimental.surgery.circuit import logical_state_init

    code = codes.SteaneCode()
    with pytest.raises(ValueError, match="state"):
        logical_state_init(code, "2", log_idx=0)


@pytest.mark.parametrize("state", ["1", "-"])
def test_logical_state_init_log_idx_flips_only_that_logical_qubit(state: str) -> None:
    """The flip pattern preserves every stabilizer and anticommutes with only the chosen logical.

    For "1" the X-type flips f must satisfy H_Z f = 0 and Z̄_j · f = δ_{j, log_idx}; for "-" the
    Z-type flips satisfy the analogous conditions with H_X and X̄_j.
    """
    import sympy

    from qldpc.experimental.surgery.circuit import logical_state_init

    xs, ys = sympy.symbols("x y")
    code = codes.BBCode({xs: 3, ys: 6}, xs**3 + ys + ys**2, ys**3 + xs + xs**2)
    base, checks, logicals = (
        ("0", code.matrix_z, code.get_logical_ops(Pauli.Z))
        if state == "1"
        else ("+", code.matrix_x, code.get_logical_ops(Pauli.X))
    )
    prep = logical_state_init(code, state, log_idx=3)
    assert set(prep) <= {base, state}
    flips = np.array([char == state for char in prep], dtype=np.int_)
    assert not ((np.asarray(checks, dtype=np.int_) @ flips) % 2).any()
    pairings = (np.asarray(logicals, dtype=np.int_) @ flips) % 2
    assert pairings.tolist() == [int(j == 3) for j in range(code.dimension)]


@pytest.mark.parametrize("log_idx", [-1, 1])
def test_logical_state_init_log_idx_out_of_range_raises(log_idx: int) -> None:
    """log_idx outside [0, code.dimension) raises IndexError."""
    from qldpc.experimental.surgery.circuit import logical_state_init

    code = codes.SteaneCode()  # k = 1; only log_idx=0 is valid
    with pytest.raises(IndexError, match="log_idx"):
        logical_state_init(code, "1", log_idx=log_idx)


@pytest.mark.parametrize("state,expected_obs0", [("0", 0), ("1", 1)])
def test_logical_state_init_end_to_end_bbcode_basis_z(state: str, expected_obs0: int) -> None:
    """BBCode [[36, 8]] single-PPM (basis=Z): regression for even-weight Z̄.

    For BBCode (l=3, m=6) the chosen Z̄_0 has weight 8 (even), so naive broadcast `"1"*36` produces
    logical |0⟩_L (NOT |1⟩_L) and obs0=0, silently failing any truth table that hardcodes expected=1
    for "1".

    The helper uses X̄_0 to flip the correct support, so obs0 tracks the textbook expectation.  If
    this test ever returns obs0=0 for state="1", the helper has regressed to naive broadcast.
    """
    import sympy

    from qldpc.experimental.surgery.circuit import (
        build_single_ppm_circuit,
        logical_state_init,
    )
    from qldpc.experimental.surgery.gadget import build_gadget

    xs, ys = sympy.symbols("x y")
    code = codes.BBCode({xs: 3, ys: 6}, xs**3 + ys + ys**2, ys**3 + xs + xs**2)
    z_bar = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    assert int(z_bar.sum()) % 2 == 0, "test premise broken: this BBCode should have even-wt Z̄_0"
    g = build_gadget(code, z_bar, basis=Pauli.Z)
    circuit = build_single_ppm_circuit(
        g,
        rounds=3,
        noise_model=None,
        data_init=logical_state_init(code, state, log_idx=0),
    )
    raw = circuit.compile_sampler().sample(shots=200).astype(np.uint8)
    n_meas = raw.shape[1]
    obs0_recs = []
    for ln in str(circuit).splitlines():
        if ln.startswith("OBSERVABLE_INCLUDE(0)"):
            obs0_recs = [int(t.strip("rec[]")) for t in ln.split() if t.startswith("rec[")]
            break
    obs0 = np.bitwise_xor.reduce(raw[:, [n_meas + off for off in obs0_recs]], axis=1)
    rate = float(obs0.mean())
    assert rate == float(expected_obs0), (
        f"state={state!r}: obs0 rate {rate:.3f} != expected {expected_obs0}. "
        f"This is the BBCode even-wt regression test — failure here means "
        f"logical_state_init is no better than naive '{state}' * n broadcast."
    )


@pytest.mark.parametrize("rounds", [1, 3])
@pytest.mark.parametrize("state", ["0", "1"])
def test_multi_round_invariance_steane_basis_z(rounds: int, state: str) -> None:
    """obs0 reads the merged Z̄ eigenvalue independently of R.

    Webster, Smith, Cohen arXiv:2511.15989 §II A gives the single-round identity
    Z̄ = ∏_{v ∈ support} A_v on the merged stabilizer group: the XOR of one round's meas-check
    outcomes equals the eigenvalue bit of Z̄.

    Therefore obs0 = int(state) for every R ≥ 1, since |0⟩_L gives Z̄ = +1 and |1⟩_L gives Z̄ = −1.

    Any round-index drift in _surgery_qec_cycle, _surgery_observable, or
    MeasurementRecord.get_target_rec would break that invariance for some R.  Reading the observable
    as an XOR across all R rounds instead collapses to R·m_v mod 2, which is silently 0 for every
    even R; test_single_ppm_even_rounds_truth_table covers even R.
    """
    from qldpc.experimental.surgery.circuit import (
        build_single_ppm_circuit,
        logical_state_init,
    )
    from qldpc.experimental.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    z_bar = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g = build_gadget(code, z_bar, basis=Pauli.Z)
    circuit = build_single_ppm_circuit(
        g,
        rounds=rounds,
        noise_model=None,
        data_init=logical_state_init(code, state, log_idx=0),
    )
    raw = circuit.compile_sampler().sample(shots=200).astype(np.uint8)
    n_meas = raw.shape[1]
    obs0_recs = []
    for ln in str(circuit).splitlines():
        if ln.startswith("OBSERVABLE_INCLUDE(0)"):
            obs0_recs = [int(t.strip("rec[]")) for t in ln.split() if t.startswith("rec[")]
            break
    obs0 = np.bitwise_xor.reduce(raw[:, [n_meas + off for off in obs0_recs]], axis=1)
    rate = float(obs0.mean())
    expected_obs0 = int(state)
    assert rate == float(expected_obs0), (
        f"rounds={rounds}, state={state!r}: obs0 rate {rate:.3f} != "
        f"expected {expected_obs0} (Webster Z̄=∏A_v should hold for any R)"
    )


@pytest.mark.parametrize("error_qubit", list(range(7)))
def test_single_qubit_x_error_triggers_only_neighboring_z_checks_steane(
    error_qubit: int,
) -> None:
    """Inject X_ERROR(1.0) on data qubit ``error_qubit`` before the first QEC round.

    Injected between state prep and the first QEC round of the Steane basis=Z PPM, whose round-1
    reliable Z-checks compare the measured syndrome to +1.  An X on data qubit i flips the parity of
    every Z-stab whose support contains i, so exactly those detectors must fire, and the assertion
    compares the fired set by row index rather than by count.

    X_ERROR rather than a data_init change because stim's detector sampler reports actual XOR
    tableau-predicted: a prep-only change is already in the tableau prediction and fires nothing,
    while a noise channel is not, so the affected detectors always deviate.
    """
    from qldpc.experimental.surgery.circuit import build_single_ppm_circuit
    from qldpc.experimental.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    z_bar = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g = build_gadget(code, z_bar, basis=Pauli.Z)
    clean_circuit = build_single_ppm_circuit(
        g,
        rounds=1,
        noise_model=None,
        data_init="0" * 7,
    )

    # Splice X_ERROR(1.0) at the boundary between state prep and QEC.  _surgery_state_prep emits
    # only R, RX, X, Z instructions (closed set) before the QEC cycle begins.  Scan for the LAST
    # such op and insert immediately after — this is robust to future QEC ops (MPP, XCX, etc.) that
    # an open-set heuristic would misclassify.
    lines = str(clean_circuit).splitlines()
    prep_ops = ("R", "RX", "X", "Z")
    last_prep_idx = -1
    for i, ln in enumerate(lines):
        s = ln.strip()
        if not s:
            continue  # pragma: no cover  -- stim's str() never emits blank lines today
        op = s.split()[0].split("(")[0]
        if op in prep_ops:
            last_prep_idx = i
    assert last_prep_idx >= 0, "could not locate any prep op (R/RX/X/Z) in Steane PPM circuit"
    injected_lines = (
        lines[: last_prep_idx + 1] + [f"X_ERROR(1.0) {error_qubit}"] + lines[last_prep_idx + 1 :]
    )
    injected_circuit = stim.Circuit("\n".join(injected_lines))

    sampler = injected_circuit.compile_detector_sampler()
    detection_events, _ = sampler.sample(
        shots=1,
        separate_observables=True,
    )
    events = detection_events[0]

    # Identify ROUND-1 reliable Z-side detectors from the clean reference: the deterministic-0
    # detectors with round coordinate 0.  Only those are flipped by a prep-time X error; the
    # post-SHIFT detectors compare the round-1 syndrome against the data-derived one and stay at 0.
    # They are emitted in data-H_Z row order, set by _classify_reliable_round1_checks iterating
    # qubit_ids.checks_z[:m_Z], so deterministic_zero_round1[j] corresponds to H_Z row j.
    clean_sampler = clean_circuit.compile_detector_sampler()
    clean_events, _ = clean_sampler.sample(
        shots=256,
        separate_observables=True,
    )
    all_det_zero = np.where(clean_events.sum(axis=0) == 0)[0]
    det_coords = clean_circuit.get_detector_coordinates()
    deterministic_zero = np.array(
        [d for d in all_det_zero if det_coords[d][0] == 0.0],
        dtype=int,
    )

    HZ = np.asarray(code.matrix_z).astype(int)
    n_reliable_z = HZ.shape[0]  # 3 for Steane
    assert len(deterministic_zero) == n_reliable_z, (
        f"expected exactly {n_reliable_z} round-1 deterministic-zero "
        f"detectors on clean Steane basis=Z PPM (rounds=1), got "
        f"{len(deterministic_zero)} — reliable-check emission order may "
        f"have changed"
    )

    # Steane Z-stabs touching error_qubit (row indices)
    z_stabs_touching = {int(j) for j in np.where(HZ[:, error_qubit] == 1)[0]}
    fired_z_stab_rows = {j for j in range(len(deterministic_zero)) if events[deterministic_zero[j]]}
    assert fired_z_stab_rows == z_stabs_touching, (
        f"X_ERROR on qubit {error_qubit}: expected Z-stab rows "
        f"{sorted(z_stabs_touching)} to fire, got "
        f"{sorted(fired_z_stab_rows)}. This is the syndrome-extraction "
        f"wiring regression: CX swap, wrong measurement basis, "
        f"EdgeColoring schedule bug, or a stabilizer row that was "
        f"reordered/replaced. The set comparison catches bugs that "
        f"swap detector contents while preserving cardinality."
    )


def test_joint_ppm_even_rounds_truth_table() -> None:
    """obs0 must encode logical X̄_l X̄_r parity correctly at EVEN rounds.

    _surgery_observable must read a single round's product of meas-check outcomes (Webster, Smith,
    Cohen arXiv:2511.15989 §II A: Z̄ = ∏_v A_v).  XOR-ing across all R rounds instead would give
    R · m_v ≡ 0 mod 2 at even R, so the parity would silently vanish exactly here.  Uses
    ``compile_sampler`` + manual XOR to read the raw observable bit rather than stim's noiseless
    flip, which is derived from the same prediction under test.
    """
    from qldpc.experimental.surgery.bridge import build_bridge
    from qldpc.experimental.surgery.circuit import build_joint_ppm_circuit
    from qldpc.experimental.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g_l = build_gadget(code, x, basis=Pauli.X)
    g_r = build_gadget(codes.SteaneCode(), x, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)
    # basis=X, so we sweep ("+", "+"), ("-", "+"), ("+", "-"), ("-", "-").  "-" on data flips X̄ to
    # -1; X̄_l X̄_r = product → parity bit.
    cases = [
        (("+", "+"), 0),
        (("-", "+"), 1),
        (("+", "-"), 1),
        (("-", "-"), 0),
    ]
    for data_init, expected in cases:
        circuit, _ = build_joint_ppm_circuit(
            g_l,
            g_r,
            bridge,
            rounds=2,
            noise_model=None,
            data_init=data_init,
        )
        raw = circuit.compile_sampler().sample(shots=16).astype(np.uint8)
        n_meas = raw.shape[1]
        obs_lines = [ln for ln in str(circuit).splitlines() if ln.startswith("OBSERVABLE_INCLUDE")]
        offs0 = [int(t.strip("rec[]")) for t in obs_lines[0].split() if t.startswith("rec[")]
        offs1 = [int(t.strip("rec[]")) for t in obs_lines[1].split() if t.startswith("rec[")]
        obs0 = np.bitwise_xor.reduce(raw[:, [n_meas + o for o in offs0]], axis=1)
        obs1 = np.bitwise_xor.reduce(raw[:, [n_meas + o for o in offs1]], axis=1)
        assert (obs0 == expected).all(), (
            f"data_init={data_init!r}: obs0 has {(obs0 != expected).sum()}/"
            f"16 shots disagreeing with expected parity bit {expected}"
        )
        assert (obs0 == obs1).all(), f"data_init={data_init!r}: obs0 != obs1 in noiseless run"


def test_single_ppm_even_rounds_truth_table() -> None:
    """obs0 must encode single-patch X̄ (or Z̄) parity at EVEN rounds.

    Same regression as test_joint_ppm_even_rounds_truth_table but for the single-patch PPM
    construction.  Sweeps "+" and "-" data inits in basis=X and "0", "1" in basis=Z to expose the
    cumulative-XOR bug at even rounds.  Uses compile_sampler + manual XOR for the same reason as
    that test.
    """
    from qldpc.experimental.surgery.circuit import build_single_ppm_circuit, logical_state_init
    from qldpc.experimental.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    basis_cases: list[tuple[PauliXZ, list[tuple[str, int]]]] = [
        (Pauli.X, [("+", 0), ("-", 1)]),
        (Pauli.Z, [("0", 0), ("1", 1)]),
    ]
    for basis, cases in basis_cases:
        op = (
            code.get_logical_ops(Pauli.X)[0]
            if basis is Pauli.X
            else code.get_logical_ops(Pauli.Z)[0]
        )
        op_arr = np.asarray(op).astype(np.uint8)
        g = build_gadget(code, op_arr, basis=basis)
        for state, expected in cases:
            data_init = logical_state_init(code, state=state, log_idx=0)
            circuit = build_single_ppm_circuit(
                g,
                rounds=2,
                noise_model=None,
                data_init=data_init,
            )
            raw = circuit.compile_sampler().sample(shots=16).astype(np.uint8)
            n_meas = raw.shape[1]
            obs_lines = [
                ln for ln in str(circuit).splitlines() if ln.startswith("OBSERVABLE_INCLUDE")
            ]
            offs0 = [int(t.strip("rec[]")) for t in obs_lines[0].split() if t.startswith("rec[")]
            offs1 = [int(t.strip("rec[]")) for t in obs_lines[1].split() if t.startswith("rec[")]
            obs0 = np.bitwise_xor.reduce(raw[:, [n_meas + o for o in offs0]], axis=1)
            obs1 = np.bitwise_xor.reduce(raw[:, [n_meas + o for o in offs1]], axis=1)
            assert (obs0 == expected).all(), (
                f"basis={basis!r} state={state!r}: obs0 has "
                f"{(obs0 != expected).sum()}/16 shots disagreeing with "
                f"expected parity bit {expected}"
            )
            assert (obs0 == obs1).all(), (
                f"basis={basis!r} state={state!r}: obs0 != obs1 in noiseless run"
            )


def test_keep_only_observable_drops_others_and_recurses_into_repeat() -> None:
    """keep_only_observable retains the matching OBSERVABLE_INCLUDE observable.

    Recurses into REPEAT blocks, dropping all other observable IDs.
    """
    from qldpc.experimental.surgery.circuit import keep_only_observable

    inner = stim.Circuit("""
        TICK
        OBSERVABLE_INCLUDE(0) rec[-1]
        OBSERVABLE_INCLUDE(1) rec[-2]
    """)
    outer = stim.Circuit()
    outer.append("M", [0, 1])
    outer.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], 1)
    outer.append(stim.CircuitRepeatBlock(2, inner))
    outer.append("OBSERVABLE_INCLUDE", [stim.target_rec(-2)], 0)

    kept = keep_only_observable(outer, keep_idx=0)
    text = str(kept)
    # obs(0) outside REPEAT preserved
    assert "OBSERVABLE_INCLUDE(0)" in text
    # obs(1) outside REPEAT removed
    assert text.count("OBSERVABLE_INCLUDE(1)") == 0
    # REPEAT block still present and filtered (only obs(0) inside)
    assert "REPEAT 2" in text
    repeat_body_lines = [ln.strip() for ln in text.splitlines() if "OBSERVABLE_INCLUDE" in ln]
    assert all("OBSERVABLE_INCLUDE(0)" in ln for ln in repeat_body_lines)


def test_expand_joint_data_init_rejects_non_str_non_seq_type() -> None:
    """_expand_joint_data_init raises TypeError on data_init that isn't str/tuple/list/None."""
    from qldpc.experimental.surgery.circuit import _expand_joint_data_init

    with pytest.raises(TypeError, match="data_init must be"):
        _expand_joint_data_init(
            {"bad": "input"},  # type: ignore[arg-type]
            n_l=4,
            n_r=4,
            intercode=True,
        )


def test_single_ppm_dem_ok_bb_36_8_with_boost() -> None:
    """Single-PPM DEM constructs cleanly on BB [[36, 8]] with boost.

    Contract test: single-PPM does not call build_bridge / SkipTree, so nothing in the bridge's
    handling of boosts or of duplicate weight-2 rows can reach it.  The fixture asserts both
    conditions the bridge is sensitive to, so if a refactor ever routes single-PPM through bridge
    code, stim's non-deterministic-detector rejection catches it here.
    """
    import sympy

    from qldpc.circuits.noise_model import DepolarizingNoiseModel
    from qldpc.experimental.surgery.cheeger import boost_gadget, cheeger_constant
    from qldpc.experimental.surgery.circuit import (
        build_single_ppm_circuit,
        keep_only_observable,
    )
    from qldpc.experimental.surgery.gadget import build_gadget

    xs, ys = sympy.symbols("x y")
    code = codes.BBCode({xs: 3, ys: 6}, xs**3 + ys + ys**2, ys**3 + xs + xs**2)
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g = build_gadget(code, z, basis=Pauli.Z)
    # Premise: restricted incidence has duplicate weight-2 rows.
    assert g.incidence.shape[0] > np.unique(g.incidence, axis=0).shape[0], (
        "test premise broken: BB [[36, 8]] Z̄_0 restriction should have duplicate κ rows"
    )
    assert cheeger_constant(g) == 1.0
    g = boost_gadget(g, method="combinatorial", target=2.0, max_extra_qubits=20, seed=3)

    noise = DepolarizingNoiseModel(1e-3, include_idling_error=False)
    circuit = build_single_ppm_circuit(g, rounds=3, noise_model=noise)
    stripped = keep_only_observable(circuit, keep_idx=0)
    dem = stripped.detector_error_model(approximate_disjoint_errors=True)
    assert dem.num_detectors > 0


def test_build_single_ppm_circuit_rejects_non_positive_rounds() -> None:
    """rounds < 1 is rejected rather than silently yielding the rounds=1 circuit."""
    from qldpc.experimental.surgery.circuit import build_single_ppm_circuit
    from qldpc.experimental.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    with pytest.raises(ValueError, match="rounds must be >= 1"):
        build_single_ppm_circuit(g, rounds=0, noise_model=None)


def test_build_joint_ppm_circuit_rejects_non_positive_rounds() -> None:
    """rounds < 1 is rejected on the joint-PPM path too."""
    from qldpc.experimental.surgery.bridge import build_bridge
    from qldpc.experimental.surgery.circuit import build_joint_ppm_circuit
    from qldpc.experimental.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    logical = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    gadget = build_gadget(code, logical, basis=Pauli.X)
    with pytest.raises(ValueError, match="rounds must be >= 1"):
        build_joint_ppm_circuit(gadget, gadget, build_bridge(gadget, gadget), rounds=0)


def _lightest_undetectable_logical_fault(circuit: stim.Circuit) -> int | None:
    """Weight of the lightest undetectable fault flipping observable 0, when that weight is 1 or 2.

    Returns None when every undetectable observable-flipping fault has weight 3 or more.

    A fault goes undetected exactly when the detector flips of its error mechanisms cancel.  One
    error mechanism qualifies when it flips no detector at all; two qualify when they flip the same
    detectors.  Grouping the detector error model's error mechanisms by their detector column
    therefore settles weights 1 and 2 in a single pass, without enumerating pairs.
    """
    from qldpc.decoders.dems import DetectorErrorModelArrays
    from qldpc.experimental.surgery.circuit import keep_only_observable

    arrays = DetectorErrorModelArrays(keep_only_observable(circuit, 0))
    detector_flips, observable_flips, _ = arrays.get_arrays()
    flips = np.asarray(observable_flips.todense()).astype(np.uint8)[0]
    columns = np.asarray(detector_flips.todense()).astype(np.uint8)

    groups: dict[bytes, list[int]] = collections.defaultdict(list)
    for index in range(columns.shape[1]):
        groups[columns[:, index].tobytes()].append(index)

    silent = np.zeros(columns.shape[0], dtype=np.uint8).tobytes()
    if any(flips[index] for index in groups.get(silent, ())):
        return 1
    for members in groups.values():
        if any(flips[index] for index in members) and any(not flips[index] for index in members):
            return 2
    return None


@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
def test_repeated_rounds_close_the_single_fault_readout_path(basis: PauliXZ) -> None:
    """One error mechanism flips the PPM outcome undetectably at rounds=1, but never at rounds=3.

    obs0 is the XOR of the last QEC round's meas-check (S'_meas) outcomes, so at rounds=1 nothing
    compares that round against another and a single error mechanism on the readout suffices.  The
    repeated rounds are what supply the detector redundancy, which is the circuit's whole
    fault-tolerance argument; this pins that they deliver it.

    The weight-2 search is exhaustive, so a rounds=3 result of 2 is a genuine pair and None means
    the fault distance is at least 3.  Steane's Z̄ gadget reaches only 2, so distance preservation
    is not asserted here -- see ``cheeger_constant`` for what is and is not established.
    """
    from qldpc.circuits.noise_model import DepolarizingNoiseModel
    from qldpc.experimental.surgery.circuit import build_single_ppm_circuit
    from qldpc.experimental.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    operator = np.asarray(code.get_logical_ops(basis)[0]).astype(np.uint8)
    gadget = build_gadget(code, operator, basis=basis)
    noise = DepolarizingNoiseModel(p=1e-3)

    single_round = build_single_ppm_circuit(gadget, rounds=1, noise_model=noise)
    assert _lightest_undetectable_logical_fault(single_round) == 1, (
        "rounds=1 should admit a single-fault readout path; the contrast below is otherwise vacuous"
    )

    repeated = build_single_ppm_circuit(gadget, rounds=3, noise_model=noise)
    assert _lightest_undetectable_logical_fault(repeated) != 1, (
        "no single fault may flip the PPM outcome undetectably once the rounds repeat"
    )
