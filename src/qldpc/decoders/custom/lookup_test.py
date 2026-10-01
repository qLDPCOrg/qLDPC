# SPDX-License-Identifier: Apache-2.0

"""Unit tests for lookup.py."""

from __future__ import annotations

import collections
import sys
import warnings

import galois
import numpy as np
import pytest
import scipy.sparse
import stim

from qldpc import codes, decoders, math
from qldpc.decoders.conftest import SurfaceCodeProblem, ToyProblem
from qldpc.decoders.custom.lookup import _VectorCodec, get_observable_decoder_lookup


def test_lookup(toy_problem: ToyProblem) -> None:
    """Lookup decoding should be straightforward."""
    matrix, error, syndrome = toy_problem

    decoder = decoders.get_decoder_lookup(matrix, max_weight=2)
    assert np.array_equal(error, decoder.decode(syndrome))
    assert len(decoder) == len(decoder.syndrome_to_error)

    # decode with a detector error model
    dem = decoders.DetectorErrorModelArrays.from_arrays(matrix, None, 1e-3).to_dem()
    decoder = decoders.get_decoder_lookup(dem, max_weight=2)
    assert np.array_equal(error, decoder.decode(syndrome))

    erasing_decoder = decoders.get_decoder_lookup(matrix, max_weight=2, add_erasure_bit=True)
    assert erasing_decoder.has_erasure_bit
    assert np.array_equal(erasing_decoder.decode(syndrome), [*error, 0])

    observable_decoder = get_observable_decoder_lookup(dem, max_weight=2)
    assert np.array_equal(observable_decoder.decode_observables(syndrome), [])


def test_observable_lookup_decoding() -> None:
    """Lookup decoding can identify the most likely observable flip for each syndrome."""
    obs_matrix: math.IntegerArray

    # toy detector error model and error syndrome
    dem = stim.DetectorErrorModel("""
        error(0.10) D0
        error(0.09) D0 L0
        error(0.06) D0 L0
    """)
    dem_arrays = decoders.DetectorErrorModelArrays(dem, simplify=False)
    pcm, obs_matrix, error_probs = dem_arrays.get_arrays()
    syndrome = np.array([1], dtype=int)

    # given only the parity check matrix, a LookupDecoder will return the most likely error
    decoder = decoders.LookupDecoder(pcm, max_weight=1, error_channel=error_probs)
    assert np.array_equal(obs_matrix @ decoder.decode(syndrome), [0])

    # provided a DEM, the LookupDecoder will simplify and predict the most likely observable flip
    decoder = decoders.LookupDecoder(dem, max_weight=1)
    assert np.array_equal(obs_matrix @ decoder.decode(syndrome), [1])

    # an ObservableLookupDecoder returns the observable flip directly
    observable_decoder = decoders.ObservableLookupDecoder(dem, max_weight=1)
    assert np.array_equal(observable_decoder.decode_observables(syndrome), [1])
    # an unseen syndrome falls back to a zero observable flip of the correct length
    assert np.array_equal(observable_decoder.decode_observables(np.array([0], dtype=int)), [0])

    # this also works when given a parity check matrix and observable_flip_matrix
    observable_decoder = decoders.ObservableLookupDecoder(
        pcm, max_weight=1, error_channel=error_probs, observable_flip_matrix=obs_matrix
    )
    assert np.array_equal(observable_decoder.decode_observables(syndrome), [1])

    # The above example is "trivial" in the sense that simplifying the DEM is sufficient to predict
    # the correct observable flips....
    dem_arrays = decoders.DetectorErrorModelArrays(dem, simplify=True)
    pcm, obs_matrix, error_probs = dem_arrays.get_arrays()
    decoder = decoders.LookupDecoder(pcm, max_weight=1, error_channel=error_probs)
    assert np.array_equal(obs_matrix @ decoder.decode(syndrome), [1])

    # However, sometimes simplifying is not enough.  Consider the following DEM, in which each error
    # has a unique (detector, observable) pattern, so simplifying changes nothing:
    dem = stim.DetectorErrorModel("""
        error(0.04) D0 D1  # E0: syndrome (1, 1), obs_flip=0
        error(0.25) D0     # E1: syndrome (1, 0), obs_flip=0
        error(0.10) D1     # E2: syndrome (0, 1), obs_flip=0
        error(0.10) D0 L0  # E3: syndrome (1, 0), obs_flip=1
        error(0.25) D1 L0  # E4: syndrome (0, 1), obs_flip=1
    """)
    dem_arrays = decoders.DetectorErrorModelArrays(dem)
    pcm, obs_matrix, error_probs = dem_arrays.get_arrays()
    syndrome = np.array([1, 1], dtype=int)

    # without knowing about observables, the most likely error is E0, with obs_flip=0
    decoder = decoders.LookupDecoder(pcm, max_weight=2)
    assert np.array_equal(obs_matrix @ decoder.decode(syndrome), [0])

    # however, it is more likely that either (E1 + E4) XOR (E2 + E3) occurred, which have obs_flip=1
    decoder = decoders.LookupDecoder(dem, max_weight=2)
    assert np.array_equal(obs_matrix @ decoder.decode(syndrome), [1])

    # a WeightedObservableLookupDecoder can be built from a DEM and predict observable flips
    weighted_observable = decoders.WeightedObservableLookupDecoder(dem, max_weight=2)
    # the min-weight error E0 has obs_flip=0
    assert np.array_equal(weighted_observable.decode_observables(syndrome), [0])
    assert np.array_equal(weighted_observable.decode_observables(np.array([0, 1], dtype=int)), [0])
    # a table hit carries the dtype that a table miss falls back to
    predicted_flip = weighted_observable.decode_observables(syndrome)
    assert predicted_flip.dtype == weighted_observable.default_correction.dtype

    # ... or from a parity check matrix and an explicit observable_flip_matrix
    weighted_observable = decoders.WeightedObservableLookupDecoder(
        pcm, max_weight=2, observable_flip_matrix=obs_matrix
    )
    assert np.array_equal(weighted_observable.decode_observables(syndrome), [0])

    # post-selecting on a detector drops it from the syndrome keys; decode still takes the full
    # syndrome and internally removes the post-selected bits before the lookup
    decoder = decoders.LookupDecoder(dem, max_weight=2, post_select=[0])
    assert np.array_equal(obs_matrix @ decoder.decode(np.array([0, 1], dtype=int)), [1])  # E4
    weighted = decoders.WeightedLookupDecoder(dem, max_weight=2, post_select=[0])
    assert np.array_equal(obs_matrix @ weighted.decode(np.array([0, 1], dtype=int)), [0])  # E2: D1

    # grouping errors by observable flip requires a way to weigh errors against each other
    with pytest.raises(ValueError, match="error_channel, or penalty_func"):
        decoders.LookupDecoder(pcm, max_weight=2, observable_flip_matrix=obs_matrix)


def test_observable_lookup_deprecation_warning_location() -> None:
    """Legacy observable lookup warnings identify the user call and its typed replacement."""
    dem = stim.DetectorErrorModel("error(0.1) D0 L0")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        decoder = decoders.get_decoder(
            dem,
            with_lookup=True,
            max_weight=1,
            predict_observable_flips=True,
        )

    assert np.array_equal(decoder.decode(np.array([1], dtype=int)), [1])
    assert len(caught) == 2
    assert all(warning.filename == __file__ for warning in caught)
    messages = [str(warning.message) for warning in caught]
    assert all("ObservableLookupDecoder" in message for message in messages)
    assert any("decode_observables" in message for message in messages)


def test_explicit_observable_lookup_decoders() -> None:
    """Observable lookup decoders expose output-specific methods and types."""
    dem = stim.DetectorErrorModel("error(0.1) D0 L0")
    syndrome = np.array([1], dtype=int)

    decoder = decoders.ObservableLookupDecoder(dem, max_weight=1)
    assert not hasattr(decoder, "decode")
    assert np.array_equal(decoder.decode_observables(syndrome), [1])
    assert np.array_equal(decoder.decode_observables_batch(np.array([[1], [0]])), [[1], [0]])

    weighted = decoders.WeightedObservableLookupDecoder(dem, max_weight=1)
    assert not hasattr(weighted, "decode")
    assert np.array_equal(weighted.decode_observables(syndrome), [1])
    assert np.array_equal(weighted.decode_observables_batch(np.array([[1], [0]])), [[1], [0]])
    assert issubclass(decoders.WeightedLookupDecoder, decoders.LookupDecoder)

    # an empty batch yields empty predictions
    no_syndromes = np.zeros((0, 1), dtype=int)
    assert decoder.decode_observables_batch(no_syndromes).shape == (0, 1)
    assert weighted.decode_observables_batch(no_syndromes).shape == (0, 1)

    with pytest.raises(TypeError, match="observable flips rather than errors"):
        decoders.get_error_decoder(dem, decoder=decoder)  # type: ignore[arg-type]

    # deprecated lookup decoders that predict observable flips are marked as such, so they are
    # rejected where an error decoder is required
    with pytest.warns(DeprecationWarning, match="use ObservableLookupDecoder"):
        legacy = decoders.LookupDecoder(dem, max_weight=1, predict_observable_flips=True)
    with pytest.warns(DeprecationWarning, match="use WeightedObservableLookupDecoder"):
        legacy_weighted = decoders.WeightedLookupDecoder(
            dem, max_weight=1, predict_observable_flips=True
        )
    for legacy_decoder in [legacy, legacy_weighted]:
        assert legacy_decoder.decode_returns_observables
        assert np.array_equal(legacy_decoder.decode(syndrome), [1])
        with pytest.raises(TypeError, match="observable flips rather than errors"):
            decoders.get_error_decoder(dem, decoder=legacy_decoder)
    assert not decoders.LookupDecoder(dem, max_weight=1).decode_returns_observables


def test_tie_breaking() -> None:
    """Equally likely errors for one syndrome resolve in favor of the lightest."""
    # every error of this code is equally likely, so every candidate ties on probability
    pcm = np.array([[1, 1, 1]], dtype=int)
    error_channel = [0.5, 0.5, 0.5]
    syndrome = np.array([1], dtype=int)

    decoder = decoders.LookupDecoder(pcm, 3, error_channel=error_channel)
    assert np.count_nonzero(decoder.decode(syndrome)) == 1

    # grouping errors by observable flip must break ties the same way
    decoder = decoders.LookupDecoder(
        pcm, 3, error_channel=error_channel, observable_flip_matrix=np.array([[1, 0, 0]])
    )
    assert np.count_nonzero(decoder.decode(syndrome)) == 1

    # a penalty function that reports no preference likewise leaves weight to decide
    weighted_decoder = decoders.WeightedLookupDecoder(np.array([[1, 1, 1, 1]], dtype=int), 4)
    assert (
        np.count_nonzero(
            weighted_decoder.decode(np.array([1], dtype=int), penalty_func=lambda _: 0.0)
        )
        == 1
    )

    # A symplectic error weighs the qudits it addresses, not its nonzero entries: a Y on one qudit
    # has two nonzero entries, so counting entries cannot tell it from an error on two qudits.
    code = codes.SurfaceCode(3)
    quantum_syndrome = np.array([1, 0, 0, 0, 1, 1, 0, 0], dtype=int)
    decoder = decoders.LookupDecoder(
        code.matrix,
        2,
        symplectic=True,
        observable_flip_matrix=code.get_logical_ops(),
        penalty_func=lambda _: 0.0,
    )
    assert math.symplectic_weight(decoder.decode(quantum_syndrome)) == 1

    weighted_decoder = decoders.WeightedLookupDecoder(code.matrix, 2, symplectic=True)
    decoded = weighted_decoder.decode(quantum_syndrome, penalty_func=lambda _: 0.0)
    assert math.symplectic_weight(decoded) == 1


def test_lookup_post_selection_violation() -> None:
    """A syndrome that a LookupDecoder post-selects against decodes as one never enumerated."""
    matrix = np.eye(3, dtype=int)  # syndrome bit kk is flipped by error kk alone
    decoder = decoders.LookupDecoder(matrix, max_weight=1, post_select=[0], add_erasure_bit=True)

    # a syndrome nontrivial on bit 0 was never enumerated, so it is erased
    assert np.array_equal([0, 0, 0, 1], decoder.decode(np.array([1, 1, 0])))
    assert np.array_equal([0, 0, 0, 1], decoder.decode(np.array([1, 0, 0])))

    # a WeightedLookupDecoder post-selects identically
    weighted_decoder = decoders.WeightedLookupDecoder(
        matrix, max_weight=1, post_select=[0], add_erasure_bit=True
    )
    assert np.array_equal([0, 1, 0, 0], weighted_decoder.decode(np.array([0, 1, 0])))
    assert np.array_equal([0, 0, 0, 1], weighted_decoder.decode(np.array([1, 1, 0])))


def test_invalid_arguments() -> None:
    """A LookupDecoder rejects contradictory or insufficient arguments."""
    pcm = np.eye(2, dtype=int)
    dem = stim.DetectorErrorModel("error(0.1) D0 L0")

    with pytest.raises(ValueError, match=r"providing a stim\.DetectorErrorModel"):
        decoders.LookupDecoder(dem, 1, error_channel=[0.1])
    with pytest.raises(ValueError, match="both an error_channel and a penalty_func"):
        decoders.LookupDecoder(pcm, 1, error_channel=[0.1, 0.1], penalty_func=lambda _: 0.0)

    # an observable lookup decoder built from a parity check matrix requires observables
    for decoder_class in [
        decoders.ObservableLookupDecoder,
        decoders.WeightedObservableLookupDecoder,
    ]:
        with pytest.raises(ValueError, match="requires an observable_flip_matrix"):
            decoder_class(pcm, 1)

    # a detector error model without observables predicts trivial observable flips
    dem_without_observables = stim.DetectorErrorModel("error(0.1) D0\nerror(0.1) D0 D1")
    for decoder_class in [
        decoders.ObservableLookupDecoder,
        decoders.WeightedObservableLookupDecoder,
    ]:
        observable_decoder = decoder_class(dem_without_observables, 1)
        assert observable_decoder.decode_observables(np.array([1, 0])).shape == (0,)

    # reject malformed channels and invalid probabilities
    for error_channel in [np.array([0.1]), np.array([[0.1, 0.2]])]:
        with pytest.raises(ValueError, match=r"error_channel must have shape \(2,\)"):
            decoders.LookupDecoder(pcm, 1, error_channel=error_channel)
    for invalid_probability in [-0.1, 1.1, np.nan, np.inf, -np.inf]:
        with pytest.raises(ValueError, match="finite probabilities between 0 and 1"):
            decoders.LookupDecoder(pcm, 1, error_channel=[invalid_probability, 0.1])

    # The endpoints are deterministic but valid probabilities.
    decoders.LookupDecoder(pcm, 1, error_channel=[0, 1])


def test_confidence_ratio() -> None:
    """A confidence_ratio omits ambiguous syndromes so they decode to erasure."""
    pcm = np.eye(1, dtype=int)

    # a confidence_ratio must be a non-negative number
    with pytest.raises(ValueError, match="non-negative"):
        decoders.LookupDecoder(pcm, 1, error_channel=[0.1], confidence_ratio=-1)
    with pytest.raises(ValueError, match="non-negative"):
        decoders.LookupDecoder(pcm, 1, error_channel=[0.1], confidence_ratio=float("nan"))
    # a positive confidence_ratio signals erasure, so it conflicts with add_erasure_bit=False
    with pytest.raises(ValueError, match="add_erasure_bit=False"):
        decoders.LookupDecoder(
            pcm, 1, error_channel=[0.1], add_erasure_bit=False, confidence_ratio=2
        )
    # ... and it requires grouping errors by observable flip
    with pytest.raises(ValueError, match="observable flip"):
        decoders.LookupDecoder(pcm, 1, error_channel=[0.1], confidence_ratio=2)

    # confidence_ratio=0 is a requirement-free no-op: it needs neither an erasure bit nor obs flips
    decoder = decoders.LookupDecoder(pcm, 1, error_channel=[0.1], confidence_ratio=0)
    assert not decoder.has_erasure_bit
    assert np.array_equal(decoder.decode(np.array([1], dtype=int)), [1])

    # a zero-probability error mechanism does not crash a syndrome that only it can explain: here
    # syndrome (1, 0) is reachable only via the probability-0 mechanism, so its group has no
    # finite-probability representative, but the decoder still returns that mechanism's error
    decoder = decoders.LookupDecoder(
        np.array([[1, 0], [0, 1]], dtype=int),
        max_weight=1,
        error_channel=[0.0, 0.1],
        observable_flip_matrix=np.array([[1, 0]], dtype=int),
    )
    assert np.array_equal(decoder.decode(np.array([1, 0], dtype=int)), [1, 0])

    # In this DEM, syndrome (1, 1)'s most likely observable flip (obs_flip=1) is only ~1.067 times
    # as likely as the alternative; see test_observable_lookup_decoding for the enumeration.
    dem = stim.DetectorErrorModel("""
        error(0.04) D0 D1
        error(0.25) D0
        error(0.10) D1
        error(0.10) D0 L0
        error(0.25) D1 L0
    """)
    syndrome = np.array([1, 1], dtype=int)

    # confidence_ratio=0 assigns the most likely flip (obs_flip=1) and adds no erasure bit
    observable_decoder = decoders.ObservableLookupDecoder(dem, max_weight=2, confidence_ratio=0)
    assert np.array_equal(observable_decoder.decode_observables(syndrome), [1])

    # a confidence_ratio above the ~1.067 threshold omits the syndrome and auto-enables the erasure
    # bit, so the syndrome decodes to erasure: an all-zero flip with the erasure bit set
    observable_decoder = decoders.ObservableLookupDecoder(dem, max_weight=2, confidence_ratio=1.5)
    assert observable_decoder.has_erasure_bit
    assert np.array_equal(observable_decoder.decode_observables(syndrome), [0, 1])

    # a syndrome with a single consistent observable flip is always confident, so it is kept and
    # decodes to that flip with the auto-enabled erasure bit left clear
    dem = stim.DetectorErrorModel("error(0.1) D0 L0")
    observable_decoder = decoders.ObservableLookupDecoder(dem, max_weight=1, confidence_ratio=1e6)
    assert np.array_equal(observable_decoder.decode_observables(np.array([1], dtype=int)), [1, 0])

    # An infinite confidence_ratio erases every syndrome that has a competing flip which can occur,
    # so a competing flip of zero probability is no competition.  The large finite ratio above does
    # not cover that case.
    syndrome = np.array([1], dtype=int)
    for model, expected in [
        ("error(0.1) D0 L0\nerror(0.2) D0", [0, 1]),  # both flips can occur, so erase
        ("error(0.1) D0 L0\nerror(0) D0", [1, 0]),  # the competing flip cannot occur
    ]:
        observable_decoder = decoders.ObservableLookupDecoder(
            stim.DetectorErrorModel(model),
            max_weight=1,
            confidence_ratio=np.inf,
        )
        assert np.array_equal(observable_decoder.decode_observables(syndrome), expected)


def test_quantum_lookup_decoding(surface_code_problem: SurfaceCodeProblem) -> None:
    """Lookup-decode random weight-2 errors in a GF(3) surface code."""
    code, _error, syndrome = surface_code_problem
    decoder: decoders.ErrorDecoder
    decoder = decoders.LookupDecoder(code.matrix, symplectic=True, max_weight=2)
    decoded_error = decoder.decode(syndrome).view(code.field)
    assert np.array_equal(syndrome, code.matrix @ math.symplectic_conjugate(decoded_error))

    decoder = decoders.LookupDecoder(
        code.matrix,
        symplectic=True,
        add_erasure_bit=True,
        max_weight=2,
        penalty_func=lambda vec: int(np.count_nonzero(vec)),
    )
    decoded_error = decoder.decode(syndrome).view(code.field)
    assert decoded_error[-1] == 0
    assert np.array_equal(syndrome, code.matrix @ math.symplectic_conjugate(decoded_error[:-1]))
    assert decoder.decode(np.ones_like(syndrome))[-1] == 1

    decoder = decoders.WeightedLookupDecoder(
        code.matrix, symplectic=True, add_erasure_bit=True, max_weight=2
    )
    assert len(decoder) == len(decoder.syndrome_to_candidates)
    decoded_error = decoder.decode(syndrome).view(code.field)
    assert decoded_error[-1] == 0
    assert np.array_equal(syndrome, code.matrix @ math.symplectic_conjugate(decoded_error[:-1]))
    assert decoder.decode(np.ones_like(syndrome))[-1] == 1

    # passing penalty_func=None returns the last-recorded (lowest-weight) consistent candidate
    decoded_error = decoder.decode(syndrome, penalty_func=None).view(code.field)
    assert np.array_equal(syndrome, code.matrix @ math.symplectic_conjugate(decoded_error[:-1]))


def test_quantum_observable_flip_prediction() -> None:
    """A predicted observable flip is one that some error consistent with the syndrome induces.

    The flip that an error induces in a logical operator is their symplectic product, which pairs
    the X sector of one with the Z sector of the other.  A plain matrix product pairs each sector
    with itself, and exchanging the sectors is independent of the characteristic, so getting this
    wrong predicts unachievable flips over every field rather than only in odd characteristic.

    A lookup table has nothing to predict from but the errors it enumerates, so the flips induced by
    those errors are exactly the flips it may return.  The enumeration is shared with the decoder;
    the symplectic product it is checked against is written out here.
    """
    for order in [2, 3]:
        code = codes.SurfaceCode(3, field=order)
        logicals = code.get_logical_ops()
        achievable_flips: dict[tuple[int, ...], set[tuple[int, ...]]] = collections.defaultdict(set)
        for error, syndrome in decoders.LookupDecoder._iter_errors_and_syndromes(
            code.matrix, 1, None, True
        ):
            conjugate = math.symplectic_conjugate(error.view(code.field))
            achievable_flips[syndrome].add(tuple((logicals @ conjugate).view(np.ndarray).tolist()))

        # at this weight every syndrome admits exactly one flip, so the check below is an equality;
        # this pins that, since a syndrome admitting several flips would check much less
        assert max(map(len, achievable_flips.values())) == 1

        weighted = decoders.WeightedObservableLookupDecoder(
            code.matrix, max_weight=1, observable_flip_matrix=logicals, symplectic=True
        )
        for syndrome, flips in achievable_flips.items():
            predicted_flip = weighted.decode_observables(np.array(syndrome, dtype=int))
            assert tuple(predicted_flip.tolist()) in flips

        # the same operators given as a sparse matrix, or as a plain array whose entries have to be
        # reduced into the field, name the same logical operators and so predict the same flips
        plain_logicals = logicals.view(np.ndarray).astype(int)
        for equivalent in [
            scipy.sparse.csc_matrix(plain_logicals),
            plain_logicals + order,
        ]:
            same = decoders.ObservableLookupDecoder(
                code.matrix,
                max_weight=1,
                observable_flip_matrix=equivalent,
                symplectic=True,
                penalty_func=lambda vec: int(np.count_nonzero(vec)),
            )
            for syndrome, flips in achievable_flips.items():
                assert (
                    tuple(same.decode_observables(np.array(syndrome, dtype=int)).tolist()) in flips
                )

    # the errors that a lookup table enumerates live over the field of its parity check matrix, so
    # an observable flip matrix over any other field cannot say what they flip
    with pytest.raises(ValueError, match="cannot be paired with"):
        decoders.ObservableLookupDecoder(
            np.array([[1, 1, 0], [0, 1, 1]]),
            max_weight=1,
            observable_flip_matrix=galois.GF(3)([[1, 2, 1]]),
            penalty_func=lambda vec: int(np.count_nonzero(vec)),
        )


def test_observable_flip_matrix_arithmetic() -> None:
    """An observable flip matrix is read over the field of the parity check matrix.

    A plain integer matrix names the same operators as the galois.FieldArray holding them, and so
    predicts the same flips.  galois stores a small field in a narrow dtype, whose wrap-around at
    256 does not commute with reducing modulo an odd order, so the product needs a wider one.  And
    an extension field is not the integers modulo its order -- over GF(4), 2 * 2 is 3 rather than 0
    -- so the product there has to be taken with field arithmetic, which reports flips that an
    integer product does not.
    """
    # A syndrome of (16, 0) over GF(17) is induced by the single error [16, 0, 0], so the flip that
    # the observable [16, 0, 0] takes from it is 16 * 16 = 1.  Every entry is in the field, and the
    # product still overflows the uint8 that galois stores GF(17) in.
    field = galois.GF(17)
    observable_flip_matrix = field([[16, 0, 0]])
    assert observable_flip_matrix.dtype == np.uint8  # the premise of the check below
    for flip_matrix in [observable_flip_matrix, observable_flip_matrix.view(np.ndarray)]:
        decoder = decoders.ObservableLookupDecoder(
            field([[1, 1, 0], [0, 1, 1]]),
            max_weight=1,
            observable_flip_matrix=flip_matrix,
            penalty_func=lambda vec: int(np.count_nonzero(vec)),
        )
        assert (
            int(decoder.decode_observables(np.array([16, 0], dtype=int))[0])
            == 16 * 16 % field.order
        )

    field = galois.GF(4)
    pcm = field([[1, 1, 0], [0, 1, 1]])
    observable_flip_matrix = field([[2, 0, 2]])
    achievable_flips: dict[tuple[int, ...], set[int]] = collections.defaultdict(set)
    for error, syndrome in decoders.LookupDecoder._iter_errors_and_syndromes(pcm, 1, None, False):
        achievable_flips[syndrome].add(int((observable_flip_matrix @ error.view(field))[0]))
    assert 3 in set.union(*achievable_flips.values())  # unreachable by an integer product

    decoder = decoders.ObservableLookupDecoder(
        pcm,
        max_weight=1,
        observable_flip_matrix=observable_flip_matrix,
        penalty_func=lambda vec: int(np.count_nonzero(vec)),
    )
    for syndrome, flips in achievable_flips.items():
        assert int(decoder.decode_observables(np.array(syndrome, dtype=int))[0]) in flips

    # Unlike a prime field, an extension field is not the integers modulo its order, so an
    # out-of-range integer cannot be reduced into that field without changing its meaning.
    for invalid_entry in [-1, field.order]:
        with pytest.raises(ValueError, match="must have elements"):
            decoders.ObservableLookupDecoder(
                pcm,
                max_weight=1,
                observable_flip_matrix=np.array([[invalid_entry, 0, 0]]),
                penalty_func=lambda vec: int(np.count_nonzero(vec)),
            )


def test_penalty_func() -> None:
    """Lookup tables can build penalty functions that penalize unlikely errors."""
    error_channel = [0.2, 0.1]
    penalty_func = decoders.LookupDecoder._build_penalty_func(error_channel)
    assert penalty_func([0, 0]) < penalty_func([1, 0]) < penalty_func([0, 1]) < penalty_func([1, 1])


@pytest.mark.parametrize("order", [2, 3, 4, 257, 2**17])
def test_vector_codec_round_trip(order: int) -> None:
    """Packed vectors round-trip exactly, with leading zeros, empty vectors, and their dtype."""
    for length in [0, 1, 9, 17]:
        codec = _VectorCodec(order, length, np.int64)
        for vector in [np.zeros(length, dtype=int), np.arange(length) % order]:
            packed = codec.pack(vector)
            assert isinstance(packed, bytes)
            unpacked = codec.unpack(packed)
            assert unpacked.dtype == np.int64
            assert np.array_equal(unpacked, vector)

    # invalid vectors cannot be packed, so they cannot collide with valid ones
    codec = _VectorCodec(order, 2, np.int64)
    for invalid in [[0, order], [-1, 0], [0, 0, 0], [0.5, 0]]:
        assert codec.pack(np.array(invalid)) is None
    assert codec.pack(np.array([1.0, 0.0])) is not None
    assert codec.pack(np.array([1, 0], dtype=object)) is not None


@pytest.mark.parametrize("order", [2, 3, 4, 257, 2**17])
@pytest.mark.parametrize("length", [0, 1, 9, 17])
def test_vector_codec_batch_round_trip(order: int, length: int) -> None:
    """Packed rows round-trip exactly, including empty and non-contiguous vectors."""
    codec = _VectorCodec(order, length, np.int64)
    values = (np.arange(3 * 2 * length) % order).reshape(3, 2 * length)[:, ::2]
    assert not values.flags.c_contiguous or length <= 1

    packed = codec.pack_rows_trusted(values)
    assert packed.shape == (len(values), codec.num_bytes)
    unpacked = codec.unpack_rows(packed)
    assert unpacked.dtype == np.int64
    assert np.array_equal(unpacked, values)

    if order == 2 and length == 9:
        vector = np.array([[1, 0, 1, 0, 0, 0, 0, 1, 1]], dtype=np.uint8)
        assert codec.pack_rows_trusted(vector).tolist() == [[0b10000101, 0b00000001]]


def test_vector_codec_batch_validity() -> None:
    """Batch validity is row-specific and rejects non-field or non-integral entries."""
    codec = _VectorCodec(3, 2, np.int64)
    assert codec.valid_rows(np.array([[0, 2], [0, 3], [-1, 0]])).tolist() == [
        True,
        False,
        False,
    ]
    assert codec.valid_rows(np.array([[0.0, 2.0], [0.5, 1.0]])).tolist() == [True, False]
    assert codec.valid_rows(np.array([[0, 2]], dtype=object)).tolist() == [True]
    with pytest.raises(ValueError, match="Expected vectors of shape"):
        codec.valid_rows(np.zeros((2, 3), dtype=int))
    with pytest.raises(ValueError, match="Expected packed vectors of shape"):
        codec.unpack_rows(np.zeros((2, 3), dtype=np.uint8))


def test_lookup_batch_validation() -> None:
    """Packed batch helpers reject malformed row and validity-mask shapes."""
    decoder = decoders.LookupDecoder(np.eye(2, dtype=int), max_weight=1)
    with pytest.raises(ValueError, match="Expected syndromes of shape"):
        decoder.decode_errors_batch(np.zeros(2, dtype=int))
    with pytest.raises(ValueError, match="Expected syndromes of shape"):
        decoder.decode_errors_batch(np.array(0))
    with pytest.raises(ValueError, match="Expected packed syndromes of shape"):
        decoder._lookup_packed_rows(np.zeros((1, 2), dtype=np.uint8))
    with pytest.raises(ValueError, match="Expected a validity mask"):
        decoder._lookup_packed_rows(np.zeros((1, 1), dtype=np.uint8), np.ones(2, dtype=bool))

    observable = decoders.ObservableLookupDecoder(
        stim.DetectorErrorModel("error(0.1) D0 L0"), max_weight=1
    )
    with pytest.raises(ValueError, match="Expected bit-packed syndromes of shape"):
        observable.decode_shots_bit_packed(np.zeros((1, 2), dtype=np.uint8))
    with pytest.raises(ValueError, match="Expected bit-packed syndromes of shape"):
        observable.decode_shots_bit_packed(np.array(0, dtype=np.uint8))


def test_packed_lookup_table_memory() -> None:
    """A packed lookup table needs much less memory than one of tuples and arrays."""
    code = codes.HammingCode(5)
    decoder = decoders.LookupDecoder(code.matrix, max_weight=1)
    packed = decoder._syndrome_to_error.items()
    unpacked = decoder.syndrome_to_error.copy().items()
    packed_size = sum(sys.getsizeof(key) + sys.getsizeof(value) for key, value in packed)
    unpacked_size = sum(sys.getsizeof(key) + sys.getsizeof(value) for key, value in unpacked)
    assert 3 * packed_size < unpacked_size


@pytest.mark.parametrize("field", [galois.GF(2), galois.GF(3), galois.GF(4)])
def test_syndrome_to_error_mapping(field: type[galois.FieldArray]) -> None:
    """The syndrome_to_error mapping reads, writes, and deletes entries of the packed table."""
    matrix = field([[1, 1, 0], [0, 1, 1]])
    decoder = decoders.LookupDecoder(matrix, max_weight=1, add_erasure_bit=True)
    table = decoder.syndrome_to_error
    assert table is decoder.syndrome_to_error
    assert len(table) == len(decoder) == len(set(map(tuple, table)))

    # every entry is consistent with its syndrome
    for syndrome, error in table.items():
        assert error[-1] == 0
        assert np.array_equal(matrix @ field(error[:-1]), syndrome)
        assert np.array_equal(decoder.decode(np.array(syndrome)), error)

    # values are copies
    syndrome = next(iter(table))
    table[syndrome][0] = (table[syndrome][0] + 1) % field.order
    assert np.array_equal(table[syndrome], decoder.decode(np.array(syndrome)))

    # assignment and deletion affect decoding
    custom = np.array([1, 1, 1, 0])
    table[syndrome] = custom
    assert np.array_equal(decoder.decode(np.array(syndrome)), custom)
    del table[syndrome]
    assert syndrome not in table
    assert np.array_equal(decoder.decode(np.array(syndrome)), decoder.default_correction)
    with pytest.raises(KeyError):
        del table[syndrome]
    with pytest.raises(KeyError):
        table[(field.order, 0)]
    with pytest.raises(KeyError):
        table[(None, 1)]  # type: ignore[index]

    # invalid entries are rejected
    with pytest.raises(ValueError, match="syndrome must be"):
        table[(0, 0, 0)] = custom
    with pytest.raises(ValueError, match="prediction must be"):
        table[(0, 0)] = np.array([0, 0, field.order, 0])

    # snapshots are ordinary dictionaries
    snapshot = table.copy()
    assert type(snapshot) is dict and len(snapshot) == len(table)
    assert "_PackedLookupTable" in repr(table)

    # an out-of-field or misshapen syndrome decodes as one never seen
    post_selected = decoders.LookupDecoder(matrix, max_weight=1, post_select=[0])
    assert np.array_equal(post_selected.decode(np.array([0])), post_selected.default_correction)
    assert np.array_equal(decoder.decode(np.array([field.order, 0])), decoder.default_correction)


@pytest.mark.parametrize("field", [galois.GF(2), galois.GF(3), galois.GF(4)])
def test_lookup_batch_decoding(field: type[galois.FieldArray]) -> None:
    """Packed batch lookup matches scalar lookup over binary and nonbinary fields."""
    matrix = field([[1, 1, 0], [0, 1, 1]])
    error_decoder = decoders.LookupDecoder(matrix, max_weight=1, add_erasure_bit=True)
    syndromes = np.array([[0, 0], [1, 0], [0, 1], [field.order, 0]], dtype=int)
    expected_errors = np.array([error_decoder.decode_errors(row) for row in syndromes])
    assert np.array_equal(error_decoder.decode_errors_batch(syndromes), expected_errors)
    assert np.array_equal(error_decoder.decode_batch(syndromes), expected_errors)
    assert error_decoder.decode_errors_batch(syndromes[:0]).shape == (0, 4)

    weighted_decoder = decoders.WeightedLookupDecoder(matrix, max_weight=1, add_erasure_bit=True)
    expected_weighted = np.array([weighted_decoder.decode_errors(row) for row in syndromes])
    assert np.array_equal(weighted_decoder.decode_errors_batch(syndromes), expected_weighted)
    assert np.array_equal(weighted_decoder.decode_batch(syndromes), expected_weighted)

    observable_decoder = decoders.ObservableLookupDecoder(
        matrix,
        max_weight=1,
        observable_flip_matrix=field([[1, 0, 1]]),
        penalty_func=lambda error: int(np.count_nonzero(error)),
        add_erasure_bit=True,
    )
    expected_flips = np.array([observable_decoder.decode_observables(row) for row in syndromes])
    assert np.array_equal(observable_decoder.decode_observables_batch(syndromes), expected_flips)
    assert observable_decoder.decode_observables_batch(syndromes[:0]).shape == (0, 2)

    post_selected = decoders.LookupDecoder(
        matrix, max_weight=1, post_select=[0], add_erasure_bit=True
    )
    expected_post_selected = np.array([post_selected.decode_errors(row) for row in syndromes[:-1]])
    assert np.array_equal(post_selected.decode_errors_batch(syndromes[:-1]), expected_post_selected)

    wide_syndromes = np.zeros((len(syndromes), 4), dtype=int)
    wide_syndromes[:, ::2] = syndromes
    assert np.array_equal(
        error_decoder.decode_errors_batch(wide_syndromes[:, ::2]), expected_errors
    )


@pytest.mark.parametrize("num_observables", [0, 7, 8, 9])
@pytest.mark.parametrize("add_erasure_bit", [False, True])
def test_observable_lookup_sinter_batches(num_observables: int, add_erasure_bit: bool) -> None:
    """Observable lookup decodes unpacked and packed Sinter shots directly."""
    dem = stim.DetectorErrorModel(
        "\n".join(f"error(0.1) D{oo} L{oo}" for oo in range(num_observables))
    )
    decoder = decoders.ObservableLookupDecoder(dem, max_weight=1, add_erasure_bit=add_erasure_bit)
    shots = np.zeros((3, num_observables), dtype=np.uint8)
    if num_observables:
        shots[1, 0] = 1
    if num_observables > 1:
        shots[2, :2] = 1  # a table miss and therefore an erasure, when enabled

    expected = decoder.decode_observables_batch(shots)
    assert np.array_equal(decoder.decode_shots(shots), expected)

    packed_shots = np.packbits(shots, bitorder="little", axis=1)
    packed_expected = np.packbits(expected[:, :num_observables], bitorder="little", axis=1)
    if add_erasure_bit:
        packed_expected = np.hstack([packed_expected, expected[:, -1:].astype(np.uint8)])
    assert np.array_equal(decoder.decode_shots_bit_packed(packed_shots), packed_expected)

    if num_observables % 8 and num_observables:
        dirty_padding = packed_shots.copy()
        dirty_padding[:, -1] |= np.uint8(0x80)
        assert np.array_equal(decoder.decode_shots_bit_packed(dirty_padding), packed_expected)

    assert decoder.decode_shots(shots[:0]).shape == expected[:0].shape
    assert decoder.decode_shots_bit_packed(packed_shots[:0]).shape == packed_expected[:0].shape


def test_observable_lookup_sinter_post_selection_and_mutation() -> None:
    """Direct packed shots preserve post-selection and mutable-table updates."""
    dem = stim.DetectorErrorModel("error(0.1) D0 L0\nerror(0.1) D1")
    decoder = decoders.ObservableLookupDecoder(
        dem, max_weight=1, post_select=[0], add_erasure_bit=True
    )
    shots = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    expected = decoder.decode_observables_batch(shots)
    assert np.array_equal(decoder.decode_shots(shots), expected)
    assert np.array_equal(
        decoder.decode_shots_bit_packed(np.packbits(shots, bitorder="little", axis=1)),
        np.hstack(
            [
                np.packbits(expected[:, :1], bitorder="little", axis=1),
                expected[:, -1:],
            ]
        ),
    )

    decoder.syndrome_to_error[(1,)] = np.array([1, 0])
    assert np.array_equal(decoder.decode_shots(shots[:1]), [[1, 0]])


def test_nonbinary_observable_lookup_rejects_sinter_batches() -> None:
    """Sinter shot APIs are binary even though ordinary lookup batches are field-aware."""
    field = galois.GF(3)
    decoder = decoders.ObservableLookupDecoder(
        field([[1]]),
        max_weight=1,
        observable_flip_matrix=field([[1]]),
        penalty_func=lambda error: int(np.count_nonzero(error)),
    )
    with pytest.raises(ValueError, match=r"only available over GF\(2\)"):
        decoder.decode_shots(np.array([[1]], dtype=np.uint8))
    with pytest.raises(ValueError, match=r"only available over GF\(2\)"):
        decoder.decode_shots_bit_packed(np.array([[1]], dtype=np.uint8))
