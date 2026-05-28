"""Unit tests for wedderburn_artin.py

Copyright 2023 The qLDPC Authors and Infleqtion Inc.

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

import itertools
import re

import galois
import numpy as np
import pytest

from qldpc import abstract


def test_wedderburn_artin_transformations(
    ring: abstract.GroupRing, pytestconfig: pytest.Config
) -> None:
    """Decompose semisimple rings into simple components.

    Runs for GroupRing(CyclicGroup(3), field=4) and GroupRing(AlternatingGroup(4), field=5).
    """
    seed = pytestconfig.getoption("randomly_seed")

    transformer = ring.get_transformer()

    # the embedding of ring.field = GF(q) scalars is an isomorphism
    for component_transformer in transformer.transformers:
        for aa in ring.field.elements:
            embedded_a = component_transformer.embedded_scalars[aa]
            assert aa == component_transformer.embedded_scalars_inverse[embedded_a]
        for aa, bb in itertools.product(ring.field.elements, repeat=2):
            embedded_a = component_transformer.embedded_scalars[aa]
            embedded_b = component_transformer.embedded_scalars[bb]
            embedded_ab = component_transformer.embedded_scalars[aa * bb]
            assert embedded_a * embedded_b == embedded_ab

    # check embedding of the power basis for GF(q^d) and the standard basis for the matrix algebra
    for component_transformer in transformer.transformers:
        size = component_transformer.size
        degree = component_transformer.degree
        for ii, jk in itertools.product(range(degree), range(size**2)):
            # map b^i |j><k| to a tensor that is 1 at (i, j, k)
            matrix_element = component_transformer.matrix_basis[jk]
            scalar = component_transformer.power_basis[ii]
            vec = abstract.RingMember.from_vector(scalar, ring).regular_lift() @ matrix_element
            coefficients = component_transformer.decomposition_coefficient_extractor @ vec
            expected_value = ring.field.Zeros((size**2, degree))
            expected_value[jk, ii] = 1
            assert np.array_equal(coefficients, expected_value.ravel())

    # the extraction of decomposition coefficients is invertible
    for component_transformer in transformer.transformers:
        component_basis = component_transformer.pci_reg.column_space()
        random_vec = ring.field.Random(len(component_basis), seed=seed + 1) @ component_basis
        coefficients = component_transformer.decomposition_coefficient_extractor @ random_vec
        assert np.array_equal(
            random_vec, component_transformer.decomposition_coefficient_recombiner @ coefficients
        )

    # the Wedderburn-Artin decomposition is an isomorphism
    member_a = get_random_ring_member(ring, seed + 1)
    member_b = get_random_ring_member(ring, seed + 2)
    member_ab = member_a * member_b
    separate = [
        component_transformer.project(member_a) @ component_transformer.project(member_b)
        for component_transformer in transformer.transformers
    ]
    assert all(np.array_equal(aa, bb) for aa, bb in zip(separate, transformer.decompose(member_ab)))
    assert transformer.recompose(separate) == member_ab

    # we can also decompose RingArrays
    ring_array = abstract.RingArray([[member_a, member_b]])
    assert np.array_equal(
        ring_array,
        transformer.recompose_array(transformer.decompose_array(ring_array)),
    )

    # ...and take the transpose of RingMembers and RingArrays
    member_a_T = transformer.transpose(member_a)
    member_b_T = transformer.transpose(member_b)
    assert transformer.transpose(member_a_T) == member_a
    assert transformer.transpose(member_b_T) == member_b
    assert transformer.transpose(member_a * member_b) == member_b_T * member_a_T
    assert np.array_equal(
        transformer.transpose_array(ring_array),
        abstract.RingArray([[member_a_T], [member_b_T]]),
    )

    # project_array(merge_blocks=True) and embed_array(from_blocks=True) round-trip correctly
    for component_transformer in transformer.transformers:
        projected_merged = component_transformer.project_array(ring_array, merge_blocks=True)
        assert projected_merged.shape == (
            ring_array.shape[0] * component_transformer.size,
            ring_array.shape[1] * component_transformer.size,
        )
        projected_unmerged = component_transformer.project_array(ring_array, merge_blocks=False)
        assert np.array_equal(
            component_transformer.embed_array(projected_merged, from_blocks=True),
            component_transformer.embed_array(projected_unmerged, from_blocks=False),
        )


def get_random_ring_member(ring: abstract.GroupRing, seed: int) -> abstract.RingMember:
    """Construct a random ring member: a sum of random group generators with random coefficients."""
    coeffs = ring.field.Random(ring.group.order, seed=seed)
    terms = [(coeff, gen) for coeff, gen in zip(coeffs, ring.group.generate())]
    return abstract.RingMember(ring, *terms)


def test_wedderburn_artin_errors(
    ring_cyclic3_gf2: abstract.GroupRing, pytestconfig: pytest.Config
) -> None:
    """The Wedderburn-Artin decomposition has limitations."""
    ring = ring_cyclic3_gf2
    transformer = ring.get_transformer()

    different_ring = abstract.GroupRing(abstract.CyclicGroup(3), field=4)
    with pytest.raises(ValueError, match="different ring"):
        transformer.decompose(different_ring.one)
    with pytest.raises(ValueError, match="different ring"):
        transformer.decompose_array(abstract.RingArray([different_ring.one]))

    with pytest.raises(ValueError, match="Provided .* components for a ring that should have"):
        transformer.recompose([])
    with pytest.raises(ValueError, match="Provided .* components for a ring that should have"):
        transformer.recompose_array([])

    with pytest.raises(ValueError, match="inconsistent shapes"):
        transformer.recompose_array([ring.field.Zeros((1, 1, 1)), ring.field.Zeros((1, 1))])

    with pytest.raises(ValueError, match=re.escape("does not live in GF(q^d)^{n × n}")):
        transformer.recompose(galois.GF(3).Ones(2))
    with pytest.raises(ValueError, match=re.escape("does not store matrices in GF(q^d)^{n × n}")):
        transformer.recompose_array(galois.GF(3).Ones(2))

    ring = abstract.GroupRing(abstract.CyclicGroup(2), field=2)
    with pytest.raises(ValueError, match="only exists for semisimple rings"):
        ring.get_transformer()
    with pytest.raises(ValueError, match="only exists for semisimple rings"):
        abstract.WedderburnArtinComponentTransformer(ring.one)
