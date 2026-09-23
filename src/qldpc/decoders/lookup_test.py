"""Unit tests for lookup.py.

Copyright 2025 The qLDPC Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import collections
import itertools

import galois
import numpy as np
import pytest
import scipy.sparse
import stim

from qldpc import codes, decoders, math
from qldpc.decoders.conftest import SurfaceCodeProblem, ToyProblem


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

    # with predict_observable_flips=True, the decoder returns the observable flip directly
    decoder = decoders.LookupDecoder(dem, max_weight=1, predict_observable_flips=True)
    assert np.array_equal(decoder.decode(syndrome), [1])
    # an unseen syndrome falls back to a zero observable flip of the correct length
    assert np.array_equal(decoder.decode(np.array([0], dtype=int)), [0])

    # this also works when given a parity check matrix and observable_flip_matrix
    decoder = decoders.LookupDecoder(
        pcm,
        max_weight=1,
        error_channel=error_probs,
        observable_flip_matrix=obs_matrix,
        predict_observable_flips=True,
    )
    assert np.array_equal(decoder.decode(syndrome), [1])

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

    # a WeightedLookupDecoder can be built from a DEM and predict observable flips directly
    weighted = decoders.WeightedLookupDecoder(dem, max_weight=2, predict_observable_flips=True)
    assert np.array_equal(weighted.decode(syndrome), [0])  # min-weight error E0 has obs_flip=0
    assert np.array_equal(weighted.decode(np.array([0, 1], dtype=int)), [0])

    # ... or from a parity check matrix and an explicit observable_flip_matrix
    weighted = decoders.WeightedLookupDecoder(
        pcm, max_weight=2, observable_flip_matrix=obs_matrix, predict_observable_flips=True
    )
    assert np.array_equal(weighted.decode(syndrome), [0])  # min-weight error E0 has obs_flip=0

    # post-selecting on a detector drops it from the syndrome keys; decode still takes the full
    # syndrome and internally removes the post-selected bits before the lookup
    decoder = decoders.LookupDecoder(dem, max_weight=2, post_select=[0])
    assert np.array_equal(obs_matrix @ decoder.decode(np.array([0, 1], dtype=int)), [1])  # E4
    weighted = decoders.WeightedLookupDecoder(dem, max_weight=2, post_select=[0])
    assert np.array_equal(obs_matrix @ weighted.decode(np.array([0, 1], dtype=int)), [0])  # E2: D1

    # grouping errors by observable flip requires a way to weigh errors against each other
    with pytest.raises(ValueError, match="error_channel, or penalty_func"):
        decoders.LookupDecoder(pcm, max_weight=2, observable_flip_matrix=obs_matrix)


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


def test_lookup_post_selection_violation() -> None:
    """A syndrome that a LookupDecoder post-selects against decodes as one never enumerated."""
    matrix = np.eye(3, dtype=int)  # syndrome bit kk is flipped by error kk alone
    decoder = decoders.LookupDecoder(matrix, max_weight=1, post_select=[0], add_erasure_bit=True)

    # a syndrome trivial on bit 0 is decoded from the table, unerased
    assert np.array_equal([0, 1, 0, 0], decoder.decode(np.array([0, 1, 0])))

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

    with pytest.raises(ValueError, match="providing a stim.DetectorErrorModel"):
        decoders.LookupDecoder(dem, 1, error_channel=[0.1])
    with pytest.raises(ValueError, match="both an error_channel and a penalty_func"):
        decoders.LookupDecoder(pcm, 1, error_channel=[0.1, 0.1], penalty_func=lambda _: 0.0)
    with pytest.raises(ValueError, match="requires providing a stim.DetectorErrorModel"):
        decoders.LookupDecoder(pcm, 1, error_channel=[0.1, 0.1], predict_observable_flips=True)


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
    decoder = decoders.LookupDecoder(
        dem, max_weight=2, predict_observable_flips=True, confidence_ratio=0
    )
    assert np.array_equal(decoder.decode(syndrome), [1])

    # a confidence_ratio above the ~1.067 threshold omits the syndrome and auto-enables the erasure
    # bit, so the syndrome decodes to erasure: an all-zero flip with the erasure bit set
    decoder = decoders.LookupDecoder(
        dem, max_weight=2, predict_observable_flips=True, confidence_ratio=1.5
    )
    assert decoder.has_erasure_bit
    assert np.array_equal(decoder.decode(syndrome), [0, 1])

    # a syndrome with a single consistent observable flip is always confident, so it is kept and
    # decodes to that flip with the auto-enabled erasure bit left clear
    dem = stim.DetectorErrorModel("error(0.1) D0 L0")
    decoder = decoders.LookupDecoder(
        dem, max_weight=1, predict_observable_flips=True, confidence_ratio=1e6
    )
    assert np.array_equal(decoder.decode(np.array([1], dtype=int)), [1, 0])


def test_quantum_lookup_decoding(surface_code_problem: SurfaceCodeProblem) -> None:
    """Lookup-decode random weight-2 errors in a GF(3) surface code."""
    code, _error, syndrome = surface_code_problem
    decoder: decoders.Decoder
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
        decoder: decoders.Decoder

        achievable_flips: dict[tuple[int, ...], set[tuple[int, ...]]] = collections.defaultdict(set)
        for error, syndrome in decoders.LookupDecoder._iter_errors_and_syndromes(
            code.matrix, 1, None, True
        ):
            conjugate = math.symplectic_conjugate(error.view(code.field))
            achievable_flips[syndrome].add(tuple((logicals @ conjugate).view(np.ndarray).tolist()))

        # at this weight every syndrome admits exactly one flip, so the check below is an equality;
        # this pins that, since a syndrome admitting several flips would check much less
        assert max(map(len, achievable_flips.values())) == 1

        for decoder in [
            decoders.LookupDecoder(
                code.matrix,
                max_weight=1,
                observable_flip_matrix=logicals,
                predict_observable_flips=True,
                symplectic=True,
                penalty_func=lambda vec: int(np.count_nonzero(vec)),
            ),
            decoders.WeightedLookupDecoder(
                code.matrix,
                max_weight=1,
                observable_flip_matrix=logicals,
                predict_observable_flips=True,
                symplectic=True,
            ),
        ]:
            for syndrome, flips in achievable_flips.items():
                prediction = decoder.decode(np.array(syndrome, dtype=int))
                assert tuple(prediction.tolist()) in flips

        # the same operators given as a sparse matrix, or as a plain array whose entries have to be
        # reduced into the field, name the same logical operators and so predict the same flips
        plain_logicals = logicals.view(np.ndarray).astype(int)
        for equivalent in [
            scipy.sparse.csc_matrix(plain_logicals),
            plain_logicals + order,
        ]:
            same = decoders.LookupDecoder(
                code.matrix,
                max_weight=1,
                observable_flip_matrix=equivalent,
                predict_observable_flips=True,
                symplectic=True,
                penalty_func=lambda vec: int(np.count_nonzero(vec)),
            )
            for syndrome, flips in achievable_flips.items():
                assert tuple(same.decode(np.array(syndrome, dtype=int)).tolist()) in flips

    # the errors that a lookup table enumerates live over the field of its parity check matrix, so
    # an observable flip matrix over any other field cannot say what they flip
    with pytest.raises(ValueError, match="cannot be paired with"):
        decoders.LookupDecoder(
            np.array([[1, 1, 0], [0, 1, 1]]),
            max_weight=1,
            observable_flip_matrix=galois.GF(3)([[1, 2, 1]]),
            predict_observable_flips=True,
            penalty_func=lambda vec: int(np.count_nonzero(vec)),
        )


def test_observable_flip_matrix_arithmetic() -> None:
    """An observable flip matrix is read over the field of the parity check matrix.

    Entries outside that field are reduced into it, so a plain integer matrix predicts the same
    flips as the galois.FieldArray holding the same operators.  galois stores a small field in a
    narrow dtype, whose wrap-around at 256 does not commute with reducing modulo an odd order, so
    the product needs a wider one.  And an extension field is not the integers modulo its order --
    over GF(4), 2 * 2 is 3 rather than 0 -- so the product there has to be taken with field
    arithmetic, which reports flips that an integer product does not.
    """

    def predict(pcm: math.IntegerArray, observable_flip_matrix: math.IntegerArray) -> list[int]:
        """Every flip predicted for a GF(3) syndrome, in a fixed order."""
        decoder = decoders.LookupDecoder(
            pcm,
            max_weight=1,
            observable_flip_matrix=observable_flip_matrix,
            predict_observable_flips=True,
            penalty_func=lambda vec: int(np.count_nonzero(vec)),
        )
        syndromes = itertools.product(range(3), repeat=pcm.shape[0])
        return [int(decoder.decode(np.array(syndrome, dtype=int))[0]) for syndrome in syndromes]

    field = galois.GF(3)
    pcm = field([[1, 2, 0], [0, 1, 2]])
    observables = [[1, 0, 2]]
    expected = predict(pcm, field(observables))
    assert any(expected)  # a rule that predicted nothing would not distinguish any arithmetic
    assert predict(pcm, np.array(observables)) == expected
    assert predict(pcm, np.array(observables) + field.order) == expected
    # A syndrome of (16, 0) over GF(17) is induced by the single error [16, 0, 0], so the flip that
    # the observable [16, 0, 0] takes from it is 16 * 16 = 1.  Every entry is in the field, and the
    # product still overflows the uint8 that galois stores GF(17) in.
    field = galois.GF(17)
    observable_flip_matrix = field([[16, 0, 0]])
    assert observable_flip_matrix.dtype == np.uint8  # the premise of the check below
    for flip_matrix in [observable_flip_matrix, observable_flip_matrix.view(np.ndarray)]:
        decoder = decoders.LookupDecoder(
            field([[1, 1, 0], [0, 1, 1]]),
            max_weight=1,
            observable_flip_matrix=flip_matrix,
            predict_observable_flips=True,
            penalty_func=lambda vec: int(np.count_nonzero(vec)),
        )
        assert int(decoder.decode(np.array([16, 0], dtype=int))[0]) == 16 * 16 % field.order

    field = galois.GF(4)
    pcm = field([[1, 1, 0], [0, 1, 1]])
    observable_flip_matrix = field([[2, 0, 2]])
    achievable_flips: dict[tuple[int, ...], set[int]] = collections.defaultdict(set)
    for error, syndrome in decoders.LookupDecoder._iter_errors_and_syndromes(pcm, 1, None, False):
        achievable_flips[syndrome].add(int((observable_flip_matrix @ error.view(field))[0]))
    assert 3 in set.union(*achievable_flips.values())  # unreachable by an integer product

    decoder = decoders.LookupDecoder(
        pcm,
        max_weight=1,
        observable_flip_matrix=observable_flip_matrix,
        predict_observable_flips=True,
        penalty_func=lambda vec: int(np.count_nonzero(vec)),
    )
    for syndrome, flips in achievable_flips.items():
        assert int(decoder.decode(np.array(syndrome, dtype=int))[0]) in flips


def test_penalty_func() -> None:
    """Lookup tables can build penalty functions that penalize unlikely errors."""
    error_channel = [0.2, 0.1]
    penalty_func = decoders.LookupDecoder._build_penalty_func(error_channel)
    assert penalty_func([0, 0]) < penalty_func([1, 0]) < penalty_func([0, 1]) < penalty_func([1, 1])
