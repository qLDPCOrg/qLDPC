# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ring_array.py."""

from __future__ import annotations

from typing import Any

import galois
import numpy as np
import pytest

from qldpc import abstract
from qldpc.abstract import ring_array as ring_array_module


def test_printing() -> None:
    """Convert a RingArray into a human-readable string."""
    group = abstract.DihedralGroup(6)
    ring = abstract.GroupRing(group, 3)
    one = ring.one
    x, y = ring.generators
    vec = [one + y * x**2 * y, x + y]
    ring_array = abstract.RingArray.build(vec)
    assert str(ring_array) == "[1 + y x^2 y, x + y]"


def test_ring_array(pytestconfig: pytest.Config) -> None:
    """Construct and lift a RingArray."""
    seed = pytestconfig.getoption("randomly_seed")
    np.random.seed(seed)

    int_matrix = np.random.randint(2, size=(3, 3))
    matrix = abstract.RingArray.build(int_matrix)
    assert matrix.group.equiv(abstract.TrivialGroup())
    assert np.array_equal(matrix.lift(), int_matrix)
    assert np.array_equal(
        (matrix @ matrix).lift(),
        matrix.lift() @ matrix.lift(),
    )
    assert isinstance(np.kron(matrix, matrix), abstract.RingArray)

    # infer the base ring from the entries
    ring = abstract.GroupRing(abstract.TrivialGroup())
    assert np.array_equal(
        abstract.RingArray.build([1, ring.one]),
        abstract.RingArray([ring.one, ring.one]),
    )

    # fail to construct a valid ring array
    rings = [abstract.GroupRing(abstract.TrivialGroup(), field) for field in [2, 3]]
    with pytest.raises(TypeError, match="must be RingMember-valued"):
        abstract.RingArray([[0]])
    with pytest.raises(ValueError, match="Cannot determine the underlying ring"):
        abstract.RingArray([])
    with pytest.raises(ValueError, match="Inconsistent rings"):
        abstract.RingArray([ring.one for ring in rings])
    with pytest.raises(ValueError, match="Inconsistent rings"):
        abstract.RingArray.build([ring.one for ring in rings])
    with pytest.raises(ValueError, match="requires a 1 or 2-dimensional"):
        abstract.RingArray.build([[[1]]], rings[0]).lift()

    new_matrix = abstract.RingArray.build([[1]], abstract.CyclicGroup(1))
    with pytest.raises(ValueError, match="different base rings"):
        matrix @ new_matrix
    with pytest.raises(ValueError, match="different base rings"):
        np.kron(matrix, new_matrix)

    # np.concatenate passes its arrays inside a *sequence* argument; ring mixing must still be
    # caught, and a same-ring concatenation must preserve the RingArray type and its base ring.
    one_c1 = abstract.RingArray.build([[1]], abstract.CyclicGroup(1))
    joined = np.concatenate([one_c1, new_matrix], 0)  # positional axis: a scalar arg survives too
    assert isinstance(joined, abstract.RingArray)
    assert joined.ring == new_matrix.ring
    with pytest.raises(ValueError, match="different base rings"):
        np.concatenate([one_c1, abstract.RingArray.build([[1]], abstract.CyclicGroup(2))])
    out_c2 = abstract.RingArray.build([[0], [0]], abstract.CyclicGroup(2))
    with pytest.raises(ValueError, match="different base rings"):
        np.concatenate([one_c1, new_matrix], out=out_c2)


def test_ring_array_explicit_ring_compatibility() -> None:
    """Explicit target rings accept valid embeddings and reject incompatible re-homing."""
    source_ring = abstract.GroupRing(abstract.DihedralGroup(3), field=2)
    bimodule_ring = abstract.GroupRing(source_ring.group * source_ring.group, field=2)
    matrix = abstract.RingArray.build([[source_ring.generators[0]]], bimodule_ring)
    assert matrix.ring == bimodule_ring
    assert all(member in bimodule_ring.group for _, member in matrix[0, 0])

    # the identity of a larger group is the identity of the target group
    assert abstract.RingArray.build([[bimodule_ring.one]], source_ring)[0, 0] == source_ring.one

    different_field = abstract.GroupRing(source_ring.group, field=3)
    with pytest.raises(ValueError, match="incompatible coefficient fields"):
        abstract.RingArray.build([[source_ring.one]], different_field)

    incompatible_group = abstract.GroupRing(abstract.CyclicGroup(2), field=2)
    with pytest.raises(ValueError, match="not a member"):
        abstract.RingArray.build([[source_ring.generators[0]]], incompatible_group)

    cyclic_ring = abstract.GroupRing(abstract.CyclicGroup(3), field=2)
    with pytest.raises(ValueError, match="not a member"):
        abstract.RingArray.build([[incompatible_group.group.generators[0]]], cyclic_ring)


def test_empty_ring_arrays() -> None:
    """Empty RingArrays keep their ring unless given another."""
    ring = abstract.GroupRing(abstract.CyclicGroup(3), field=2)
    empty = abstract.RingArray.build(np.zeros((0, 2), dtype=int), ring)
    assert abstract.RingArray(empty).ring == abstract.RingArray.build(empty).ring == ring
    other_ring = abstract.GroupRing(abstract.CyclicGroup(4), field=2)
    assert abstract.RingArray(empty, other_ring).ring == other_ring


def test_transpose() -> None:
    """Transpose a RingArray."""
    group = abstract.CyclicGroup(4)
    x0, x1, x2, x3 = group.generate()
    matrix = abstract.RingArray.build([[x0, 0, x1], [x2, 0, abstract.RingMember(group, x3)]])
    assert np.array_equal(matrix.T.T, matrix)

    # as in numpy, an array with fewer than two dimensions keeps its shape
    vector = abstract.RingArray.build([x1, x2], matrix.ring)
    assert np.array_equal(vector.T, abstract.RingArray.build([x3, x2], matrix.ring))
    scalar = abstract.RingArray.build(x1, matrix.ring)
    assert np.array_equal(scalar.T, abstract.RingArray.build(x3, matrix.ring))

    # a RingArray without a base ring transposes its entries in their own rings
    ring = matrix.ring
    plain = np.asarray(abstract.RingArray.build([[x1, 1], [0, x1]], ring)).view(abstract.RingArray)
    expected = abstract.RingArray.build([[x3, 0], [1, x3]], ring)
    assert np.array_equal(np.asarray(plain.T), np.asarray(expected))


@pytest.mark.parametrize(
    "ring",
    [
        abstract.GroupRing(abstract.DihedralGroup(3), field=2),
        abstract.GroupRing(abstract.AbelianGroup(2, 3), field=4),
    ],
)
def test_regular_rep(ring: abstract.GroupRing, pytestconfig: pytest.Config) -> None:
    """The regular representation enables straightforward linear algebra over group algebras."""
    seed = pytestconfig.getoption("randomly_seed")
    dense_vector = ring.field.Random(4 * ring.group.order, seed=seed)
    dense_array = ring.field.Random((3, 4, ring.group.order), seed=seed + 1)

    vector = abstract.RingArray.from_field_vector(dense_vector, ring)
    matrix = abstract.RingArray.from_field_array(dense_array, ring)
    assert np.array_equal(dense_vector, abstract.RingArray.to_field_vector(vector))
    assert np.array_equal(dense_array, abstract.RingArray.to_field_array(matrix))
    assert np.array_equal(
        (matrix @ vector).to_field_vector(),
        matrix.regular_lift() @ vector.to_field_vector(),
    )

    assert not np.any(matrix @ matrix.null_space().T)
    assert not np.any(matrix.regular_lift() @ matrix.null_space().regular_lift().T)
    assert not np.any(matrix.regular_lift() @ matrix.regular_lift().null_space().T)


def test_from_field_array() -> None:
    """Integer-valued arrays and arrays over the field of the ring are valid coefficients."""
    ring = abstract.GroupRing(abstract.CyclicGroup(3))
    identity = np.eye(3, dtype=int)
    for coefficients in [identity.astype(bool), identity.astype(float)]:
        matrix = abstract.RingArray.from_field_array(coefficients, ring)
        assert np.array_equal(matrix.to_field_array(), identity)

    with pytest.raises(ValueError, match="last axis"):
        abstract.RingArray.from_field_array(np.eye(4, dtype=int), ring)

    # coefficients are read as in RingMember.from_vector
    with pytest.raises(ValueError, match="cannot be read in"):
        abstract.RingArray.from_field_array(galois.GF(4)([2] * ring.dim), ring)

    # a coefficient vector is 1-D, with a multiple of ring.dim entries
    with pytest.raises(ValueError, match="must be a multiple of"):
        abstract.RingArray.from_field_vector(np.zeros(ring.dim + 1, dtype=int), ring)
    with pytest.raises(ValueError, match="must be 1-D"):
        abstract.RingArray.from_field_vector(np.zeros((2, ring.dim), dtype=int), ring)


def test_build_edge_cases() -> None:
    """Integers are read as in ring arithmetic, and floats and field elements as coefficients."""
    ring = abstract.GroupRing(abstract.DihedralGroup(3), field=3)
    data_list: list[abstract.groups.NestedSequence] = [
        [[-1, 4]],
        np.array([[2.0, 1.0]]).tolist(),
        [list(ring.field([2, 1]))],
    ]
    for data in data_list:
        assert np.array_equal(
            abstract.RingArray.build(data, ring), abstract.RingArray.build([[2, 1]], ring)
        )
    assert np.array_equal(
        abstract.RingArray.build(np.array([[True, False]]), ring),
        abstract.RingArray.build([[1, 0]], ring),
    )

    # floats are read as coefficients, so they are not reduced
    with pytest.raises(ValueError, match=r"must be in \[0, 3\)"):
        abstract.RingArray.build([[4.0]], ring)

    # without a ring, an array over a finite field gives the algebra of the trivial group over that
    # field, and arrays over other fields are read only if their integers denote the same elements
    field = galois.GF(9, irreducible_poly="x^2 + 1")
    array = abstract.RingArray.build(field([[3, 5]]))
    assert array.field is field
    assert abstract.RingArray.build(galois.GF(9).Zeros((0,))).field is galois.GF(9)
    assert array[0, 0] * array[0, 1] == array.ring.one * (field(3) * field(5))
    assert np.array_equal(
        abstract.RingArray.build(galois.GF(3)([[1, 2]]), array.ring),
        abstract.RingArray.build([[1, 2]], array.ring),
    )
    with pytest.raises(ValueError, match="cannot be read in"):
        abstract.RingArray.build(galois.GF(9)([[3]]), array.ring)

    # field elements given one by one keep their field
    elements = list(galois.GF(3)([2, 1]))
    assert abstract.RingArray.build(elements).field is galois.GF(3)
    with pytest.raises(ValueError, match="cannot be read in"):
        abstract.RingArray.build(elements, abstract.GroupRing(abstract.CyclicGroup(2), field=4))
    with pytest.raises(ValueError, match="Inconsistent fields"):
        abstract.RingArray.build([galois.GF(3)(1), galois.GF(5)(1)])

    # so do field arrays in a list, for each way to build a RingArray
    ring_16 = abstract.GroupRing(abstract.CyclicGroup(2), field=16)
    rows = [galois.GF(4)([2, 3])]
    assert abstract.RingArray.build([galois.GF(3)([1, 2])]).field is galois.GF(3)
    with pytest.raises(ValueError, match="cannot be read in"):
        abstract.RingArray.build(rows, ring_16)
    with pytest.raises(ValueError, match="cannot be read in"):
        abstract.RingArray.from_field_array(rows, ring_16)
    with pytest.raises(ValueError, match="cannot be read in"):
        abstract.RingArray.from_field_vector(list(rows[0]), ring_16)

    # each entry of an array is one ring member
    entries = np.empty(1, dtype=object)
    for entry in [[1], (1,), ring.field([1]), abstract.RingArray.build([1], ring)]:
        entries[0] = entry
        with pytest.raises(ValueError, match="must be scalars, group members, or ring members"):
            abstract.RingArray.build(entries, ring)

    # an entry that is a 0-d array of objects is read as the object it holds
    entries[0] = np.array(ring.one, dtype=object)
    assert abstract.RingArray.build(entries, ring)[0] == ring.one


def test_matmul_fallbacks() -> None:
    """Numpy multiplies RingArrays given keyword arguments, or with an unknown ring."""
    ring = abstract.GroupRing(abstract.CyclicGroup(7), field=4)
    matrix = abstract.RingArray.build(np.eye(4, dtype=int), ring)
    expected = np.asarray(matrix) @ np.asarray(matrix)
    output = np.empty((4, 4), dtype=object)
    product = np.matmul(matrix, matrix, out=output)
    assert type(product) is abstract.RingArray and product.ring is ring
    assert np.shares_memory(product, output)
    assert np.array_equal(product, expected)
    unknown_ring = np.asarray(matrix).view(abstract.RingArray)
    assert np.array_equal(unknown_ring @ unknown_ring, expected)
    unknown_ring[0, 0] = matrix[0, 1]
    assert unknown_ring[0, 0] is matrix[0, 1]

    # a RingArray without a base ring says so, rather than returning None
    with pytest.raises(ValueError, match="no base ring"):
        _ = unknown_ring.ring

    # an empty inner dimension gives zeros, which are integers without a base ring
    empty_ring = abstract.GroupRing(abstract.DihedralGroup(3), field=5)
    vector = abstract.RingArray.build(np.zeros(0, dtype=int), empty_ring)
    assert vector @ vector == empty_ring.zero
    unknown_vector = np.asarray(vector).view(abstract.RingArray)
    assert unknown_vector @ unknown_vector == np.sum(unknown_vector) == 0


def test_entries_of_ring_arrays() -> None:
    """Entries set or combined through indexing, fill, and numpy functions are ring members."""
    ring = abstract.GroupRing(abstract.DihedralGroup(3), field=5)
    other_ring = abstract.GroupRing(abstract.DihedralGroup(3), field=3)
    x = ring.generators[0]
    matrix = abstract.RingArray.build([[x, 1]], ring)

    # values are read as in RingArray.build, and other rings are rejected
    matrix[0, 0] = 6
    matrix[0, 1] = ring.group.generators[0]
    assert all(type(entry) is abstract.RingMember for entry in matrix.ravel())
    assert np.array_equal(
        matrix.to_field_array(), abstract.RingArray.build([[1, x]], ring).to_field_array()
    )
    with pytest.raises(ValueError, match="cannot be read in"):
        matrix[0, 0] = galois.GF(25)(5)

    # the error points to RingArray.build only for members of another group algebra over the same
    # field
    trivial_ring = abstract.GroupRing(abstract.TrivialGroup(), field=5)
    with pytest.raises(ValueError, match=r"RingArray\.build"):
        matrix[0, 0] = trivial_ring.one
    matrix_ring = abstract.MatrixRing([[0, 1], [1, 0]])
    trivial_ring_2 = abstract.GroupRing(abstract.TrivialGroup(), field=2)
    for target_ring, other_member in [
        (ring, other_ring.one),
        (matrix_ring, trivial_ring_2.one),
        (trivial_ring_2, matrix_ring.one),
    ]:
        target = abstract.RingArray.build([0], target_ring)
        with pytest.raises(ValueError, match=r"different rings$"):
            target[0] = other_member

    # members of an equal ring, and 0-dimensional arrays of members, are set as members
    equal_ring = abstract.GroupRing(abstract.DihedralGroup(3), field=5)
    matrix[0, 0] = equal_ring.generators[0]
    assert matrix[0, 0] == x
    matrix[0, 0] = abstract.RingArray(np.array(ring.one, dtype=object))
    assert type(matrix[0, 0]) is abstract.RingMember and matrix[0, 0] == ring.one

    # numpy functions and methods that write into a RingArray read values in the same way, before
    # they write them
    for write in [
        lambda array: np.copyto(dst=array, src=np.array([[np.True_, 1]], dtype=object)),
        lambda array: np.put(a=array, ind=[0, 1], v=[1, 1]),
        lambda array: np.place(arr=array, mask=np.ones((1, 2), dtype=bool), vals=[1]),
        lambda array: np.place(array, np.ones((1, 2), dtype=bool), [1]),
        lambda array: np.putmask(array, [[True, True]], 1),
        lambda array: np.putmask(array, [[True, True]], values=1),
        lambda array: np.put_along_axis(array, np.array([[0, 1]]), 1, axis=1),
        lambda array: array.fill([1]),
    ]:
        copy = matrix.copy()
        write(copy)
        assert all(type(entry) is abstract.RingMember for entry in copy.ravel())
        assert np.array_equal(copy, abstract.RingArray.build([[1, 1]], ring))
    with pytest.raises(ValueError, match="different rings"):
        np.copyto(copy, np.array([[other_ring.one, 1]], dtype=object))
    assert np.array_equal(copy, abstract.RingArray.build([[1, 1]], ring))
    square = abstract.RingArray.build(np.zeros((2, 2), dtype=int), ring)
    np.fill_diagonal(a=square, val=1)
    assert np.array_equal(square, abstract.RingArray.build(np.eye(2, dtype=int), ring))
    square = abstract.RingArray.build(np.zeros((2, 2), dtype=int), ring)
    np.fill_diagonal(square, 1)
    assert np.array_equal(square, abstract.RingArray.build(np.eye(2, dtype=int), ring))

    # entries of None, which np.empty_like gives, stay None
    assert all(entry is None for entry in np.empty_like(square).ravel())

    # values are read into copies only when some entry needs reading, so the arrays that a caller
    # passes keep their entries, and views of a RingArray stay views
    plain = np.array([0, 1], dtype=object)
    assert type(np.broadcast_arrays(matrix, plain)[1][0, 0]) is abstract.RingMember
    assert all(type(entry) is int for entry in plain)
    assert np.shares_memory(np.transpose(matrix), matrix)

    # numpy functions write into a plain target in place
    np.copyto(plain, matrix[0])
    assert all(type(entry) is abstract.RingMember for entry in plain)

    # numpy functions that join a RingArray with other arrays, including field arrays, also read
    # their entries
    with pytest.raises(ValueError, match="different rings"):
        np.concatenate([matrix, np.array([[other_ring.one, other_ring.one]], dtype=object)])
    stacked = np.concatenate([matrix, np.array([[0, ring.group.identity]], dtype=object)])
    assert np.array_equal(stacked, abstract.RingArray.build([[1, x], [0, 1]], ring))
    for join in [
        lambda array: np.concatenate([matrix, array]),
        lambda array: np.append(matrix, values=array, axis=0),
    ]:
        with pytest.raises(ValueError, match="cannot be read in"):
            join(galois.GF(25)([[5, 7]]))

    # field arrays and field scalars combine like integers, and are read as ring members also where
    # they serve as indices, so indices must be plain arrays
    field_matrix = ring.field([[1], [2]])
    assert np.array_equal(matrix @ field_matrix, abstract.RingArray.build([[1 + 2 * x]], ring))
    assert np.array_equal(matrix * ring.field(2), 2 * matrix)
    with pytest.raises(ValueError, match="cannot be read in"):
        np.take(matrix, galois.GF(2)([0, 1]), axis=1)

    # operations with boolean results, such as comparisons, give plain arrays, also when written
    # into an output, and logical ufuncs give booleans for single members
    comparison = matrix == matrix.copy()
    assert type(comparison) is np.ndarray and comparison.all()
    assert type(np.equal(matrix, matrix, out=np.zeros((1, 2), dtype=bool))) is np.ndarray
    assert type(np.any(matrix, axis=1, out=np.zeros(1, dtype=bool))) is np.ndarray
    assert type(np.logical_not(abstract.RingArray(np.array(ring.one, dtype=object)))) is bool

    # ufuncs from np.vectorize and np.frompyfunc return their results as computed, and those from
    # np.frompyfunc do not write into RingArrays, while numpy's own ufuncs keep RingArrays, also
    # those that are not attributes of np, such as the one behind .real
    sevens = np.vectorize(lambda member: 7, otypes=[object])(matrix)
    assert type(sevens) is np.ndarray and sevens.tolist() == [[7, 7]]
    before = matrix.copy()
    with pytest.raises(TypeError):
        np.frompyfunc(lambda member: 7, 1, 1)(matrix, out=matrix)
    assert np.array_equal(matrix, before)
    assert type(matrix.real) is abstract.RingArray and np.array_equal(matrix.real, matrix)

    # empty sums and products are ring members, also when written into an output
    empty = abstract.RingArray.build(np.zeros((2, 0), dtype=int), ring)
    zeros = abstract.RingArray.build(np.zeros((2, 2), dtype=int), ring)
    output = abstract.RingArray.build(np.ones((2, 2), dtype=int), ring)
    np.matmul(empty, empty.T, out=output)
    assert np.array_equal(output, zeros)
    output = abstract.RingArray.build([1, 1], ring)
    assert np.sum(empty, axis=1, out=output) is output
    assert np.array_equal(output, abstract.RingArray.build([0, 0], ring))
    plain = np.ones(2, dtype=object)
    result = np.sum(empty, axis=1, out=plain)
    assert type(result) is abstract.RingArray and result.ring is ring
    assert np.shares_memory(result, plain) and all(entry == ring.zero for entry in plain)
    # a field array as an output raises an error from galois
    with pytest.raises(TypeError):
        np.sum(empty, axis=1, out=ring.field([1, 1]))
    assert np.sum(empty) == ring.zero
    assert np.prod(empty) == ring.one
    assert np.trace(empty[:, :0]) == ring.zero
    assert np.dot(empty[0], empty[0]) == np.inner(empty[0], empty[0]) == ring.zero
    assert np.einsum("i,i", empty[0], empty[0]) == np.nansum(empty) == ring.zero
    assert np.nanprod(empty) == ring.one
    assert np.sum(empty, dtype=int) == np.add.reduce(empty[0], dtype=int) == ring.zero
    assert np.count_nonzero(matrix) == 2

    # functions that return several arrays return RingArrays
    halves = np.split(abstract.RingArray.build([[1, x]], ring), 2, axis=1)
    assert all(isinstance(half, abstract.RingArray) and half.ring is ring for half in halves)
    unique = np.unique_counts(abstract.RingArray.build([x], ring))
    assert isinstance(unique.values, abstract.RingArray) and unique.counts.tolist() == [1]


def test_howell_form_provenance(ring_cyclic3_gf2: abstract.GroupRing) -> None:
    """A semisimple Howell form records how it was built, until its entries change."""
    transformer = ring_cyclic3_gf2.get_transformer()
    matrix = abstract.RingArray.build([[1, 0]], ring_cyclic3_gf2)
    howell = matrix.howell_normal_form_semisimple(transformer, right=True)
    assert howell._get_hnf_provenance() == (transformer, True, True)
    assert matrix._get_hnf_provenance() is None
    column = howell[:, :1]
    assert isinstance(column, abstract.RingArray) and column._get_hnf_provenance() is None
    howell[0, 1] = 1
    assert howell._get_hnf_provenance() == (transformer, True, False)

    # writes through views, and reshaping in place, also invalidate the provenance
    howell = matrix.howell_normal_form_semisimple(transformer, right=True)
    howell[0][1] = 1
    assert howell._get_hnf_provenance() == (transformer, True, False)
    howell = matrix.howell_normal_form_semisimple(transformer, right=True)
    howell.resize((2, 1), refcheck=False)
    assert howell._get_hnf_provenance() == (transformer, True, False)


def test_ufunc_outputs() -> None:
    """Ufuncs write into RingArrays over the same ring."""
    ring = abstract.GroupRing(abstract.DihedralGroup(3), field=5)
    x, y = ring.generators
    matrix = abstract.RingArray.build([[x + y, 1 + x]], ring)
    output = abstract.RingArray.build([[0, 0]], ring)
    assert np.add(matrix, matrix, out=output) is output
    assert np.array_equal(output, 2 * matrix)

    # both outputs of a ufunc with two outputs are read
    quotient, remainder = (abstract.RingArray.build([0, 0], ring) for _ in range(2))
    results = np.divmod(np.array([7, 8]), 3, out=(quotient, remainder))
    assert results[0] is quotient and results[1] is remainder
    assert np.array_equal(quotient, abstract.RingArray.build([2, 2], ring))
    assert np.array_equal(remainder, abstract.RingArray.build([1, 2], ring))

    other_ring = abstract.GroupRing(abstract.DihedralGroup(3), field=3)
    with pytest.raises(ValueError, match="different base rings"):
        np.add(matrix, matrix, out=abstract.RingArray.build([[0, 0]], other_ring))

    # operands that implement numpy's protocols themselves get the call, also with an output
    class Operand:
        def __array_ufunc__(self, *args: object, **kwargs: object) -> str:
            return "ufunc"

        def __array_function__(self, *args: object, **kwargs: object) -> str:
            return "function"

    operand: Any = Operand()
    assert np.add(matrix, operand, out=output) == "ufunc"
    assert np.add(matrix, operand, out=np.zeros((1, 2), dtype=object)) == "ufunc"
    assert np.dot(matrix, operand, out=np.zeros((1, 1), dtype=object)) == "function"


def test_ring_row_reduction(ring_cyclic3_gf2: abstract.GroupRing) -> None:
    """RingArrays can be row reduced in various ways."""
    # we can row-reduce a RingArray over a cyclic group algebra, here with a second row that is the
    # first times 1 + gen, which is not a unit
    ring = ring_cyclic3_gf2
    one = ring.one
    gen = ring.generators[0]
    matrix = abstract.RingArray.build(
        [
            [gen, 0, one],
            [gen + gen**2, 0, one + gen],
            [0, 0, gen],
        ]
    )
    matrix_row_reduced = abstract.RingArray.build([[1, 0, 0], [0, 0, 1], [0, 0, 0]], ring)
    matrix_hnf = matrix_row_reduced[:2, :]  # without the all-zero row
    assert np.array_equal(matrix.row_reduce(), matrix_row_reduced)
    assert np.array_equal(matrix.howell_normal_form(), matrix_hnf)
    assert np.array_equal(matrix.howell_normal_form(poly=True), matrix_hnf)

    # RingArray.row_reduce requires semisimple rings
    ring = abstract.GroupRing(abstract.CyclicGroup(2), field=2)
    with pytest.raises(ValueError, match="only supports semisimple rings"):
        abstract.RingArray.build([[1, 0], [1, 1]], ring).row_reduce()


def test_deprecated_argument_order() -> None:
    """Deprecated argument orders emit DeprecationWarning."""
    ring = abstract.GroupRing(abstract.TrivialGroup())

    vector = ring.field.Random(2 * ring.group.order, seed=0)
    with pytest.warns(DeprecationWarning, match="DEPRECATED"):
        # the deprecated argument order does not match the typed signature
        ring_array = abstract.RingArray.from_field_vector(ring, vector)  # type:ignore[arg-type]
    assert np.array_equal(ring_array.to_field_vector(), vector)

    matrix = ring.field.Random((1, 2, ring.group.order), seed=1)
    with pytest.warns(DeprecationWarning, match="DEPRECATED"):
        # the deprecated argument order does not match the typed signature
        ring_array = abstract.RingArray.from_field_array(ring, matrix)  # type:ignore[arg-type]
    assert np.array_equal(ring_array.to_field_array(), matrix)

    # the deprecated argument order also works for other rings
    other = abstract.MatrixRing(np.array([[0, 1], [1, 0]]))
    matrix = other.field([[[1, 0], [1, 1]]])
    with pytest.warns(DeprecationWarning, match="DEPRECATED"):
        # the deprecated argument order does not match the typed signature
        ring_array = abstract.RingArray.from_field_array(other, matrix)  # type:ignore[arg-type]
    assert np.array_equal(ring_array.to_field_array(), matrix)
    vector = matrix.ravel()
    with pytest.warns(DeprecationWarning, match="DEPRECATED"):
        # the deprecated argument order does not match the typed signature
        ring_array = abstract.RingArray.from_field_vector(other, vector)  # type:ignore[arg-type]
    assert np.array_equal(ring_array.to_field_vector(), vector)


def test_deprecated_attribute_lookup() -> None:
    """The module-level lookup of deprecated names rejects other names."""
    with pytest.raises(AttributeError, match="has no attribute 'not_a_real_export'"):
        _ = ring_array_module.not_a_real_export  # type: ignore[attr-defined]


def test_ring_array_over_matrix_ring() -> None:
    """RingArrays over a matrix ring lift to block matrices of the matrices of their entries."""
    group = abstract.DihedralGroup(3)
    group_ring = abstract.GroupRing(group)
    mat_a, mat_b = [group_ring.regular_lift(member) for member in group.generators]
    ring = abstract.MatrixRing(mat_a, mat_b)
    a, b = ring.generators
    array = abstract.RingArray.build([[1, a + 1, 0], [b, 0, a * b]], ring)
    assert array.ring == ring and array.field is ring.field
    assert np.array_equal(abstract.RingArray([[a, b]]), abstract.RingArray.build([[a, b]], ring))
    assert abstract.RingArray(np.empty((0, 1), dtype=object), ring).ring == ring
    with pytest.raises(AttributeError, match="no base group"):
        _ = array.group

    # an array lifts to the block matrix of the matrices of its entries
    eye, zero = ring.field.Identity(6), ring.field.Zeros((6, 6))
    expected = np.block([[eye, mat_a + eye, zero], [mat_b, zero, mat_a @ mat_b]])
    assert np.array_equal(array.lift(), expected)

    # entries must be members of the ring of the array, or of an equal ring
    assert abstract.RingArray.build([[a]], abstract.MatrixRing(mat_a, mat_b))[0, 0] == a
    with pytest.raises(ValueError, match="different ring"):
        abstract.RingArray.build([[a]], abstract.MatrixRing(np.array([[0, 1], [1, 0]])))
    with pytest.raises(ValueError, match="different ring"):
        abstract.RingArray.build([[a]], group_ring)
    with pytest.raises(ValueError, match="different ring"):
        abstract.RingArray.build([[group_ring.one]], ring)
    with pytest.raises(
        TypeError, match="Group members combine only with members of group algebras"
    ):
        abstract.RingArray.build([[group.generators[0]]], ring)

    # integers are multiples of the identity, in the field of the ring
    ternary_ring = abstract.MatrixRing(np.array([[0, 1], [1, 0]]), field=3)
    integer_array = abstract.RingArray.build([[2]], ternary_ring)
    assert np.array_equal(integer_array.lift(), ternary_ring.field([[2, 0], [0, 2]]))


def test_ring_array_dimensions(pytestconfig: pytest.Config) -> None:
    """Lifts have blocks of size ``lift_dim``, and regular lifts and coefficients have ``dim``."""
    upper = np.array([[0, 1], [0, 0]])
    ring = abstract.MatrixRing(upper, upper.T)
    assert ring.dim == 4 and ring.lift_dim == 2
    upper_member, lower_member = ring.generators
    array = abstract.RingArray.build([[1, upper_member, 0], [lower_member, 0, 1]], ring)
    assert array.lift().shape == (2 * ring.lift_dim, 3 * ring.lift_dim)
    assert array.regular_lift().shape == (2 * ring.dim, 3 * ring.dim)
    empty = abstract.RingArray.build(np.zeros((0, 2), dtype=int), ring)
    assert empty.lift().shape == (0, 2 * ring.lift_dim)
    assert empty.regular_lift().shape == (0, 2 * ring.dim)

    # arrays are in correspondence with arrays of coefficients, one for each basis element
    seed = pytestconfig.getoption("randomly_seed")
    coefficients = ring.field.Random((2, 3, ring.dim), seed=seed)
    from_array = abstract.RingArray.from_field_array(coefficients, ring)
    assert np.array_equal(from_array.to_field_array(), coefficients)
    from_vector = abstract.RingArray.from_field_vector(coefficients.ravel(), ring)
    assert np.array_equal(from_vector.to_field_vector(), coefficients.ravel())
    with pytest.raises(ValueError, match="must have length 4"):
        abstract.RingArray.from_field_array(coefficients[..., :2], ring)


def test_kernel_over_matrix_rings() -> None:
    """The columns of a kernel are annihilated by an array, and span its null space."""
    upper = np.array([[0, 1], [0, 0]])
    ring_with_involution = abstract.MatrixRing(upper, upper.T)
    ring_without_involution = abstract.MatrixRing(upper.T)
    assert ring_with_involution.has_involution and not ring_without_involution.has_involution
    for ring in [ring_with_involution, ring_without_involution]:
        # the first two columns of this array form an invertible matrix, so the array maps R^3 onto
        # R^2 in either order of multiplication, and its null space is a free module of rank one
        array = abstract.RingArray.build([[1, ring.generators[-1], 0], [0, 1, 1]], ring)
        for right in [False, True]:
            kernel = array.kernel(right=right)
            assert kernel.shape[0] == array.shape[1]
            assert not np.any(abstract.matmul(array, kernel, right=right))
            vectors = kernel.transpose().to_field_array().reshape(-1, array.shape[1] * ring.dim)
            assert np.linalg.matrix_rank(vectors) == len(vectors) == ring.dim

        # the kernel of an invertible matrix has no columns
        assert abstract.RingArray.build([[1]], ring).kernel().shape == (1, 0)

    # with an involution, null-space rows are the conjugate transposes of kernel columns
    array = abstract.RingArray.build(
        [[1, ring_with_involution.generators[0]]], ring_with_involution
    )
    for right in [False, True]:
        assert np.array_equal(array.null_space(right=right), array.kernel(right=right).T)

    # kernels are defined only for matrices
    with pytest.raises(ValueError, match="2-dimensional"):
        abstract.RingArray.build([1, 0], ring_with_involution).kernel()


def test_ring_array_without_involution() -> None:
    """``null_space`` and transposes raise for a ring without an involution."""
    ring = abstract.MatrixRing(np.array([[0, 0], [1, 0]]))
    (nilpotent,) = ring.generators
    array = abstract.RingArray.build([[1, nilpotent]], ring)
    with pytest.raises(ValueError, match=r"use RingArray\.kernel"):
        array.null_space()
    with pytest.raises(ValueError, match="no involution"):
        _ = array.T

    # transposes raise even for an empty array
    with pytest.raises(ValueError, match="no involution"):
        _ = abstract.RingArray.build(np.zeros((0, 2), dtype=int), ring).T


def test_row_reduce_requires_group_algebra(ring_cyclic3_gf2: abstract.GroupRing) -> None:
    """Row reduction is implemented only for group algebras, even given a transformer."""
    ring = abstract.MatrixRing(np.array([[0, 1], [1, 0]]))
    matrix = abstract.RingArray.build([[1]], ring)
    with pytest.raises(NotImplementedError, match="only for group algebras"):
        matrix.row_reduce()
    with pytest.raises(NotImplementedError, match="only for group algebras"):
        matrix.row_reduce(ring_cyclic3_gf2.get_transformer())
