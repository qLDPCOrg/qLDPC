"""Wedderburn-Artin decomposition for semisimple group algebras

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

import dataclasses
import functools
import itertools
import math
import operator
from collections.abc import Sequence

import galois
import numpy as np

import qldpc

from .rings import GroupRing, RingArray, RingMember


class WedderburnArtinTransformer:
    r"""Instrument for implementing the Wedderburn-Artin decomposition of semisimple rings.

    The Wedderburn-Artin theorem states that every semisimple ring R is isomorphic to a direct
    product of matrix algebras over division rings:
        R ≅ ⨂_i R_i
    where
        R_i = D_i^{n_i × n_i},
    and D^{n × n} denotes the space of n × n matrices over the division ring D.  By Wedderburn's
    little theorem, every finite division ring is a finite field, so if R is a group algebra over a
    finite field F then every division ring D_i is a field extension of F.  If R is a commutative
    ring, then all n_i = 1, so
        R = ⨂_i D_i  (if R is commutative).

    This class is an instrument for decomposing elements of r ∈ R into simple components, taking
        r -> (r_1, r_2, ...) ∈ ⨂_i R_i,
    and and embedding elements of ⨂_i R_i back into R.

    References:
    - https://en.wikipedia.org/wiki/Wedderburn%E2%80%93Artin_theorem
    - https://en.wikipedia.org/wiki/Wedderburn%27s_little_theorem
    """

    ring: GroupRing
    transformers: list[WedderburnArtinComponentTransformer]
    random_number_generator: np.random.Generator

    def __init__(self, ring: GroupRing, *, seed: np.random.Generator | int | None = None) -> None:
        if not ring.is_semisimple:
            raise ValueError("The Wedderburn-Artin decomposition only exists for semisimple rings")
        self.ring = ring
        self.random_number_generator = (
            seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        )
        self.transformers = [
            WedderburnArtinComponentTransformer(pci, seed=self.random_number_generator)
            for pci in self.ring.get_primitive_central_idempotents()
        ]

    def decompose(self, element: RingMember) -> list[galois.FieldArray]:
        """Decompose an element of a ring into its Wedderburn-Artin components."""
        return [transformer.project(element) for transformer in self.transformers]

    def decompose_array(
        self, array: RingArray, *, merge_blocks: bool = False
    ) -> list[galois.FieldArray]:
        """Decompose a RingArray element-wise into Wedderburn-Artin components.

        Each component of N-dimensional RingArray is an (N+2)-dimensional galois.FieldArray.

        If merge_blocks is True, this method treats each projected element as a block matrix in the
        last two axes of the provided array, such that a projection with shape
            (..., r, c, rb, cb)
            is transposed and reshaped into an array with shape
            (..., r * rb, c * cb).
        """
        return [
            transformer.project_array(array, merge_blocks=merge_blocks)
            for transformer in self.transformers
        ]

    def recompose(self, components: Sequence[galois.FieldArray]) -> RingMember:
        """Invert WedderburnArtinTransformer.decompose."""
        if len(components) != len(self.transformers):
            raise ValueError(
                f"Provided {len(components)} WedderburnArtinTransformer components for a ring that"
                f" should have {len(self.transformers)}"
            )
        terms = [trans.embed(comp) for comp, trans in zip(components, self.transformers)]
        return functools.reduce(operator.add, terms)

    def recompose_array(
        self, components: Sequence[galois.FieldArray], *, from_blocks: bool = False
    ) -> RingArray:
        """Invert WedderburnArtinTransformer.decompose_array."""
        if len(components) != len(self.transformers):
            raise ValueError(
                f"Provided {len(components)} WedderburnArtinTransformer components for a ring that"
                f" should have {len(self.transformers)}"
            )
        if not len(set([array.shape[:-2] for array in components])) == 1:
            raise ValueError("Asked to combine arrays of inconsistent shapes")
        terms = [
            trans.embed_array(array, from_blocks=from_blocks)
            for array, trans in zip(components, self.transformers)
        ]
        return functools.reduce(operator.add, terms)

    def transpose(self, element: RingMember) -> RingMember:
        """Transpose the matrices representing the element within each simple component.

        Warning: this transpose should not be confused with RingMember.T, which maps every group
        member to its inverse, transposing the regular representation of a RingMember.
        """
        if self.ring.is_commutative:
            return element
        components = [np.swapaxes(component, -1, -2) for component in self.decompose(element)]
        return self.recompose(components)

    def transpose_array(self, array: RingArray) -> RingArray:
        """Transpose the array its entries within each simple component."""
        if self.ring.is_commutative:
            return array.transpose().view(RingArray)
        components = [np.swapaxes(component, -1, -2) for component in self.decompose_array(array)]
        return self.recompose_array(components).transpose().view(RingArray)


@dataclasses.dataclass
class WedderburnArtinComponentTransformer:
    r"""Transformer to map between a semisimple ring R and a simple component S of R.

    Let R = GF(q)[G] ≅ GF(q)^{|G|} be a finite group algebra, whose elements can be formally written
    as a GF(q)-linear combination of group members in G.  That is, if r ∈ R, then
        r = sum_{g in G} r_g g.
    By the Wedderburn-Artin theorem and Wedderburn's little theorem, every
    simple component S of R is isomorphic to the space of matrices over a field extension of GF(q):
        S ≅ GF(q^d)^{n × n}.
    We refer to d as the "degree" of S, and n as the "size" of S.
    If G is abelian, then its size is n = 1, so S ≅ GF(q^d).

    The simple component S can be identified by a primitive central idempotent (PCI) e ∈ R, where
        1. "Idempotent" means that e is a projector: e·e = e.
        2. "Central" means that e commutes with all of R.
        3. "Primitive" means that e cannot be decomposed into a sum of central idempotents in R.
    Given a PCI e of R, the corresponding simple component is S = R·e = { r·e : r ∈ R }.

    This class is an instrument for projecting elements of R onto a simple component S corresponding
    to a provided PCI e, and embedding elements of S back into R.

    References:
    - https://en.wikipedia.org/wiki/Wedderburn%E2%80%93Artin_theorem
    - https://en.wikipedia.org/wiki/Wedderburn%27s_little_theorem
    """

    ring: GroupRing  # base ring, R
    field: type[galois.FieldArray]  # base field of the ring, F = GF(q) = GF(p^k)

    pci: RingMember  # primitive central idempotent (PCI) e that projects onto S
    pci_vec: galois.FieldArray  # representation of the PCI as a vector in GF(q)^{|G|}
    pci_reg: galois.FieldArray  # PCI lifted to its regular representation in GF(q)^{|G| × |G|}

    center: galois.FieldArray  # basis for the center Z(S) of elements in S that commute with S
    degree: int  # degree d of the field extension GF(q^d) for S
    size: int  # size (n) of the matrices in the isomorphism S ≅ GF(q^d)^{n × n}

    power_basis: galois.FieldArray  # power basis B = (e, b, b^2, ..., b^{d-1}) for GF(q^d)
    power_basis_dual: galois.FieldArray  # dual A of the power basis, for which A @ B.T = 1_d

    extended_field: type[galois.FieldArray]  # field extension GF(p^(kd)) ≅ GF(q^d)
    embedded_scalars: galois.FieldArray  # embedding of GF(q) into GF(p^(kd))
    embedded_scalars_inverse: galois.FieldArray  # inverse of the embedding of GF(q) into GF(p^(kd))
    embedded_power_basis: galois.FieldArray  # embedding of B into GF(p^(kd))
    embedded_power_basis_dual: galois.FieldArray  # dual of embedded_power_basis w.r.t. field trace

    matrix_basis: galois.FieldArray  # matrix elements |i><j| for GF(q^d)^{n × n}

    # matrices to project R -> S ≅ GF(q^d)^{n × n} ≅ GF(q)^{n × n × d}, and embed back into R
    decomposition_coefficient_extractor: galois.FieldArray
    decomposition_coefficient_recombiner: galois.FieldArray

    def __init__(self, pci: RingMember, *, seed: np.random.Generator | None = None) -> None:
        """Initialize from a primitive central idempotent (PCI) of a ring.

        WARNING: This class assumes--and does not verify--that the provided RingMember is a PCI.
        """
        if not pci.ring.is_semisimple:
            raise ValueError("The Wedderburn-Artin decomposition only exists for semisimple rings")
        seed = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)

        self.ring = pci.ring
        self.field = self.ring.field

        self.pci = pci
        self.pci_vec = pci.to_vector()
        self.pci_reg = pci.regular_lift()

        self.center = self._get_center()
        self.degree = len(self.center)
        self.size = math.isqrt(np.linalg.matrix_rank(self.pci_reg) // self.degree)

        self.power_basis = self._get_power_basis(seed)
        self.power_basis_dual = qldpc.math.get_dual_basis(self.power_basis, validate=False)

        self.extended_field = galois.GF(self.field.order**self.degree)
        self.embedded_scalars, self.embedded_power_basis = self._get_center_embeddings()
        self.embedded_power_basis_dual = self._get_embedded_power_basis_dual()

        max_embedded_scalar = max(self.embedded_scalars.view(np.ndarray))
        self.embedded_scalars_inverse = self.field.Zeros(max_embedded_scalar + 1)
        for qq, pp in enumerate(self.embedded_scalars):
            self.embedded_scalars_inverse[int(pp)] = qq

        self.matrix_basis = self._get_matrix_basis(seed)
        self.decomposition_coefficient_extractor = self._get_decomposition_coefficient_extractor()
        self.decomposition_coefficient_recombiner = self._get_decomposition_coefficient_recombiner()

    def _get_center(self) -> galois.FieldArray:
        r"""Identify a basis for the center Z(S) of S.

        The center Z(S) is the subspace of "scalars" in S that commute with all elements of S:
            Z(S) = { z ∈ S : z·s = s·z for all s ∈ S }.

        We can decompose Z(S) = S ⋂ Z(R), where Z(R) is the center of R.  Letting L(r) and A(g)
        denote, respectively, the regular and adjoint representations of r ∈ R and g ∈ G, we then
        note that
            S ≅ ker(L(e) - 1)
        and
            Z(R) ≅ ⋂_{generators g of G} ker(A(g) - 1).
        We can therefore find a basis for Z(S) by intersecting the null space of L(e) - 1 with the
        null spaces of A(g) - 1 for the generators g of G.

        Returns:
            - A matrix in GF(q)^{d × |G|} whose rows form a basis for Z(S).
        """
        # identify the null space of L(e) - 1, which spans S
        center = self.pci_reg.column_space()  # equal to the null space of L(e) - 1

        if self.ring.is_commutative:
            # if R is commutative, then Z(S) = S, so we are done
            return center

        # intersect with the null spaces of A(g) - 1 for all generators g
        identity = self.field.Identity(self.ring.group.order)
        for generator in self.ring.group.generators:
            mat = self.ring.group.adjoint_lift(generator).view(self.field) - identity
            center = (mat @ center.T).null_space() @ center

        return center

    def _get_power_basis(self, seed: np.random.Generator) -> galois.FieldArray:
        r"""Construct a power basis for Z(S), used for the field extension GF(q) -> GF(q^d).

        Mathematically,
            GF(q^d) ≅ GF(q)[x] / f(x),
        where
        - GF(q)[x] is the set of univariate polynomials with coefficients in GF(q).
        - f(x) ∈ GF(q)[x] is an irreducible polynomial with degree d.  Here "irreducible"
            essentially means "prime": f(x) has no nontrivial factors of degree <= d.
        The monomial x is called the primitive element of GF(q)[x] / f(x), and its powers,
            (x^0, x^1, x^2, ..., x^{d-1}),
        form a "power basis" for GF(q)[x] / f(x).

        We need to find elements of Z(S) that act as GF(q^d) scalars when embedding S into
        GF(q^d)^{n × n}.  To this end, we identify a suitable generator b ∈ Z(S) that can serve as
        the primitive element of a field extension GF(q)[x] / f(x).  Crucially, the powers of this
        generator, collected into the power basis
            B = (b^0, b^1, b^2, ..., b^{d-1}),
        must be linearly independent.  Here b^0 = e is the identity element of S.

        To find a suitable power basis B, we...
            1. Pick a random element b ∈ Z(S).
            2. Collect powers of b into rows of the matrix B = (b^0, b^1, b^2, ..., b^{d-1}).
            3. Check whether the elements of B span GF(q)-linear vector space of dimension d.
                If so, return B.  Otherwise, go back to step 1.

        Returns:
            - A matrix in GF(q)^{d × |G|} whose j-th row is b^j ∈ Z(S).
        """
        if self.degree == 1:
            return self.pci_vec.reshape(1, -1).view(self.field)

        while True:
            generator_vec = self._random_nonzero_vec(len(self.center), seed) @ self.center
            generator = RingMember.from_vector(generator_vec, self.ring)
            generator_mat = generator.regular_lift()

            basis = [self.pci_vec, generator_vec]
            for _ in range(self.degree - 2):  # pragma: no cover
                basis.append(generator_mat @ basis[-1])

            basis_in_field = self.field(basis)
            if np.linalg.matrix_rank(basis_in_field) == self.degree:
                return basis_in_field

    def _random_nonzero_vec(self, length: int, seed: np.random.Generator) -> galois.FieldArray:
        """Return a random nonzero vector over GF(q)."""
        while not np.any(vector := self.field.Random(len(self.center), seed=seed)):
            pass  # pragma: no cover
        return vector

    def _get_center_embeddings(self) -> tuple[galois.FieldArray, galois.FieldArray]:
        r"""Construct embeddings of elements in the center Z(S) into GF(p^{kd}) ≅ GF(q^d).

        There are two parts to this embedding:
        1. Embedding GF(q) scalars into GF(p^{kd}).
        2. Embedding the power basis B = (b^0, b^1, b^2, ..., b^{d-1}) for GF(q^d) into GF(p^{kd}).

        Returns:
            - A vector in GF(p^{kd})^d whose j-th element is the embedding of GF(q)(j).
            - A vector in GF(p^{kd})^d whose j-th element is the embedding of b^j.
        """
        if self.degree == 1:
            return self.field.elements, self.field.Ones([1])

        """
        PART 1
        ------
        To embed GF(q) scalars into GF(p^{kd}), we...
            1. Identify the generator α of GF(q) = GF(p^k).
            2. Find the minimal polynomial m(x) of α, which has α as a root in GF(p^k).
            3. Interpret the coefficients of m(x) as elements of GF(p^{kd}).
            4. Identify a root σ ∈ GF(p^{kd}) of m(x).
        A scalar in GF(q) = GF(p^k) = GF(p)[x] / m(x) can then be embedded into GF(p^{kd}) by...
            1. Expanding the scalar as a polynomial in α with coefficients in GF(p).
            2. Interpreting the coefficients as elements of GF(p^{kd}).
            3. Replacing α by σ.
        See:
        - https://mhostetter.github.io/galois/v0.1.1/api/galois.GF/
        - https://mhostetter.github.io/galois/v0.1.1/api/galois.FieldArray.minimal_poly/
        """
        minimal_poly = self.field.primitive_element.minimal_poly()  # this is m(x) ∈ GF(q)[x]
        extended_minimal_poly = galois.Poly(minimal_poly.coeffs, field=self.extended_field)
        embedded_root = extended_minimal_poly.roots()[0]  # this is σ ∈ GF(p^{kd})
        embedded_root_powers = [self.extended_field(1)]
        for _ in range(self.field.order - 1):
            embedded_root_powers.append(embedded_root_powers[-1] * embedded_root)
        embedded_scalars = []
        for scalar in self.field.elements:
            poly_coeffs = scalar.vector()[::-1]
            terms = [
                self.extended_field(coeff) * power
                for coeff, power in zip(poly_coeffs, embedded_root_powers)
            ]
            embedded_scalars.append(functools.reduce(operator.add, terms))

        """
        PART 2
        ------
        To embed elements of the power basis B into GF(p^{kd}) we...
            1. Identify the minimal polynomial f(x) of b, for which f(b) = 0.  This is the
                polynomial we use to construct GF(q^d) = GF(q)[x] / f(x) with f(b) = 0.
            2. Map the GF(q) coefficients of f(x) into GF(p^{kd}) to obtain the polynomial g(x).
            3. Use any root of g(x) as the generator of the embedded power basis.
        To find f(x), we seek coefficients f_j ∈ GF(q) for which
            f(b) = sum_{j=0}^d f_j b^j = 0,
        where we define b^0 = e.  We can set f_d = 1 without loss of generality, reducing the
        problem to
            sum_{j=0}^{d-1} f_j b^j = -b^d.
        The remaining coefficients can be found by solving a linear system of equations.
        """
        gen_to_dim_power = self._regular_lift(self.power_basis[1]) @ self.power_basis[-1]
        linear_system = np.column_stack([self.power_basis.T, -gen_to_dim_power])
        poly_coeffs = linear_system.view(self.field).row_reduce()[: self.degree, -1]
        poly_coeffs = np.append(poly_coeffs, self.field(1))
        irreducible_poly = galois.Poly(poly_coeffs[::-1], field=self.field)  # this is f(x)
        extended_irreducible_poly = galois.Poly(
            [embedded_scalars[cc] for cc in irreducible_poly.coeffs], field=self.extended_field
        )
        embedded_generator = extended_irreducible_poly.roots()[0]
        embedded_power_basis = [self.extended_field(1)]
        for _ in range(self.degree - 1):
            embedded_power_basis.append(embedded_power_basis[-1] * embedded_generator)

        return self.extended_field(embedded_scalars), self.extended_field(embedded_power_basis)

    def _regular_lift(self, vector: galois.FieldArray, *, right: bool = False) -> galois.FieldArray:
        """Lift a member of S from GF(q)^{|G|} to a matrix that encodes ring multiplication."""
        return RingMember.from_vector(vector, self.ring).regular_lift(right=right)

    def _get_embedded_power_basis_dual(self) -> galois.FieldArray:
        r"""Construct the dual of the embedded power basis E[B].

        Let E : GF(q)^{|G|} -> GF(p^{kd}) be the embedding of the center Z(S) into GF(p^{kd}).

        For an embedded power basis E[B] = (E[b^0], E[b^1], E[b^2], ..., E[b^{d-1}]) ∈ GF(p^{kd})^d,
        the dual basis
            E[A] = (E[a_0], E[a_1], E[a_2], ..., E[a_{d-1}])
        satisfies
            Tr_{GF(q^d)/GF(q)}[E[a_i] E[b^j]] = delta_{ij},
        where Tr_{GF(q^d)/GF(q)} denotes a field trace from GF(q^d) to GF(q); see self.field_trace.

        The dual basis allows us to "pick off" the coefficients of a polynomial in the center
            Z(S) ≅ GF(q^d) ≅ GF(q)[x] / f(x)
        that has been embedded into GF(p^{kd}), which is useful for mapping back into Z(S) by:
            1. Mapping z ∈ GF(p^{kd}) to (z_0, z_1, ..., z_{d-1}) ∈ GF(q)^d with z_j = Tr[E[a_j] z].
            2. Combining these coefficients to recover sum_j z_j b^j ∈ Z(S).

        Returns:
            - A vector in GF(p^{kd}) whose j-th entry is the embedding E[a_j].
        """
        if self.degree == 1:
            return self.extended_field.Ones([1])
        matrix = self.extended_field(
            [
                self.extended_field_trace(aa * bb)
                for aa, bb in itertools.product(self.embedded_power_basis, repeat=2)
            ]
        ).reshape([self.degree] * 2)
        return np.linalg.inv(matrix) @ self.embedded_power_basis

    def extended_field_trace(self, value: galois.FieldArray) -> galois.FieldArray:
        """Compute the field trace from GF(q^d) to GF(q).

        The field trace of z from GF(q^d) to GF(q) is defined by
            Tr_{GF(q^d)/GF(q)}[z] = sum_{i=0}^{d-1} z^{q^i}.

        See:
        - https://en.wikipedia.org/wiki/Field_trace
        """
        conjugates = [value ** (self.field.order**pow) for pow in range(self.degree)]
        values = self.extended_field(np.stack(conjugates, axis=0)).sum(axis=0)
        return values.reshape(value.shape).view(self.extended_field)

    def _get_matrix_basis(self, seed: np.random.Generator) -> galois.FieldArray:
        """Construct standard basis of matrix elements |i><j| ∈ S ≅ GF(q^d)^{n × n}.

        This method first decomposes the PCI e of S into primitive (possibly non-central)
        idempotents e_i that sum to the PCI: e = sum_i e_i.  The primitive idempotents e_i are the
        "diagonal" matrix elements |i><i|.  These idempotents are, in turn, used to construct
        off-diagonal matrix elements e_ij = |i><j| ∈ e_i S e_j.

        Returns:
            - A matrix in GF(q)^{n^2 × |G|} whose (ij, :) entry is |i><j| = e_ij ∈ S.
        """
        if self.ring.is_commutative:
            return self.pci_vec.reshape(1, -1).view(self.field)

        # collect primitive idempotents along the diagonal of the matrix_basis
        pids_as_vecs = self._get_primitive_idempotents(seed)
        basis_as_vecs = self.field.Zeros([self.size, self.size, self.ring.group.order])
        basis_as_vecs[np.arange(self.size), np.arange(self.size), :] = pids_as_vecs

        # construct matrices for left- and right-multiplication by the primitive idempotents
        pid_mats_l = [self._regular_lift(idempotent) for idempotent in pids_as_vecs]
        pid_mats_r = [self._regular_lift(idempotent, right=True) for idempotent in pids_as_vecs]

        # construct the off-diagonal matrix elements |0><i| and |i><0|
        for ii in range(1, self.size):
            # projections onto e_0 R e_i and e_i R e_0
            projection_0_i = pid_mats_l[0] @ pid_mats_r[ii]
            projection_i_0 = pid_mats_l[ii] @ pid_mats_r[0]

            # bases for e_0 R e_i and e_i R e_0 in GF(q)^{|G|}
            basis_0_i = projection_0_i.column_space()
            basis_i_0 = projection_i_0.column_space()

            # choose suitable elements of e_0 R e_i and e_i R e_0 as |0><i| and |i><0|
            basis_as_vecs[0, ii, :], basis_as_vecs[ii, 0, :] = self._get_off_diagonal_basis_vecs(
                basis_0_i, basis_i_0, seed
            )

        # build left-regular representations of |0><i| and |i><0|
        basis_as_mats = {}
        for ii in range(1, self.size):
            basis_as_mats[0, ii] = self._regular_lift(basis_as_vecs[0, ii, :])
            basis_as_mats[ii, 0] = self._regular_lift(basis_as_vecs[ii, 0, :])

        # construct the remaining matrix elements |i><j| = |i><0|·|0><j|
        for ii, jj in itertools.combinations(range(1, self.size), r=2):
            basis_as_vecs[ii, jj, :] = basis_as_mats[ii, 0] @ basis_as_vecs[0, jj, :]
            basis_as_vecs[jj, ii, :] = basis_as_mats[jj, 0] @ basis_as_vecs[0, ii, :]

        return basis_as_vecs.reshape(-1, basis_as_vecs.shape[-1])

    def _get_primitive_idempotents(
        self, seed: np.random.Generator, idempotent_vec: galois.FieldArray | None = None
    ) -> galois.FieldArray:
        """Decompose an idempotent of S into primitive idempotents.

        Recall that S ≅ GF(q^d)^{n × n}.  The PCI of S is the identity matrix of GF(q^d)^{n × n}.
        If n = 1 (or: when R is commutative), then the multiplicative identity in GF(q^d) is the
        only element of GF(q^d) that squares to itself, so the PCI of S is the only idempotent in S.
        If n > 1, however, then the PCI can be decomoposed into primitive (non-central) idempotents,
            e = sum_{i=1}^n e_i,
        where e_i is essentially the |i><i| matrix element of GF(q^d)^{n × n}.

        This method decomposes an idempotent of S into primitive idempotent recursively.  Given an
        idempotent e_start, this method proceeds as follows:
            1. Determine whether e_start is primitive.  If so, return {e_start}.
            2. Find two or more idempotents that sum to e_start.
            3. Decompose each of the idempotents found at step 2 by a recursive call to this method.
            4. Return the combined set of all idempotents found from decomposition at step 3.

        Returns:
            - A matrix in GF(q)^{n × |G|} whose j-th row the primitive idempotent e_i ∈ S.
        """
        assert not self.ring.is_commutative  # this method should not have been called
        if idempotent_vec is None:
            idempotent_vec = self.pci_vec

        """
        PART 1
        ------
        To determine whether the idempotent is primitive, we first identify the sub-algebra that it
        projects onto, or the image of the map r -> e_start r e_start.  If this image spans a
        d-dimensional vector space over GF(q), then it must be GF(q^d), which means that the
        idempotent must be primitive.
        """
        idempotent = RingMember.from_vector(idempotent_vec, self.ring)
        subalgebra_proj = idempotent.regular_lift() @ idempotent.regular_lift(right=True)
        subalgebra_basis = subalgebra_proj.column_space()
        if len(subalgebra_basis) == self.degree:  # pragma: no cover (we may not hit this in tests)
            return idempotent_vec.reshape(1, -1).view(self.field)

        """
        PART 2
        ------
        At this point, we know that the idempotent e_start is not primitive, so we seek to decompose
        e_start into a sum of two or more idempotents.  To this end, we proceed as follows:
            1. Pick a random element α in the sub-algebra stabilized by e_start.
            2. Find the minimal polynomial of α, or a minimal-degree polynomial m(x) with m(α) = 0.
            3. Identify the irreducible factors of m(x).  If m(x) is irreducible, return to step 1.
            4. Factor m(x) into irreducible ("prime") polynomials p_j(x) as
                    m(x) = prod_j p_j(x)^{k_j},
                and define
                    f_j(x) = p_j(x)^{k_j},
                    g_j(x) = m(x) / f_j(x).
        These polynomials will be used to construct idempotents that sum to e_start.
        """

        # find an element of the sub-algebra whose minimal polynomial has nontrivial factors
        while True:
            random_member = self.field.Random(len(subalgebra_basis), seed=seed) @ subalgebra_basis
            minimal_poly, powers = self._get_minimal_polynomial(random_member, idempotent_vec)
            factors, multiplicities = minimal_poly.factors()
            if len(factors) > 1:
                factors = [factor**mult for factor, mult in zip(factors, multiplicities)]
                break

        # above, we found factors f_j(x) of m(x); now, build the quotients g_j(x) = m(x) / f_j(x)
        quotients = [minimal_poly // factor for factor in factors]

        """
        PART 3
        ------
        We now use the minimal polynomial m(x) of α ∈ e_start R e_start, its factors f_j(x), and the
        quotients g_j(x) to construct idempotents that sum to e_start.  To this end, we use the
        extended Euclidean algorithm to find polynomials u_j and v_j for which
            u_j f_j + v_j g_j = gcd(f_j, g_j),
        and define
            F_j = u_j f_j / gcd(f_j, g_j),
            G_j = v_j g_j / gcd(f_j, g_j).
        Some observations:
            1. By construction, F_j + G_j = 1, where "1" is e_start in this sub-algebra of S.
            2. (F_j G_j)(x) contains a factor of (f_j g_j)(x) = m(x), so (F_j G_j)(α) = 0.
            3. G_j(α) = G_j(α) (F_j + G_j)(α) = (G_j F_j)(α) + G_j(α)^2 = G_j(α)^2.
            4. If i != j, then (G_i G_j)(x) contains a factor of m(x) so, (G_i G_j)(α) = 0.
        Observation 3 implies that G_j(α) is idempotent, and observation 4 implies that these
        idempotents are orthogonal, so {G_j(α)}_j is a set of orthogonal idempotents.
        """

        new_idempotents = []
        quotient_coeff: galois.Poly
        gcd: galois.Poly
        for factor, quotient in zip(factors, quotients):
            gcd, factor_coeff, quotient_coeff = galois.egcd(factor, quotient)  # type:ignore[assignment,arg-type]
            idempotent_poly = quotient_coeff * quotient // gcd  # <- G_j(x)
            new_idempotent = idempotent_poly.coeffs[::-1] @ powers[: len(idempotent_poly)]  # <- e_j
            new_idempotents.append(new_idempotent)

        # if sum_j G_j != e_start, then add the remainder to our set of new idempotents
        if np.any(remainder := functools.reduce(operator.add, new_idempotents) - idempotent_vec):
            new_idempotents.append(remainder)  # pragma: no cover

        if len(new_idempotents) == self.size:  # pragma: no cover (we may not hit this in tests)
            # the number of mutually orthogonal idempotents guarantees that they are primitive
            return self.field(new_idempotents)
        else:  # pragma: no cover (we may not hit this in tests)
            # recursively decompose the new idempotents to find primitive idempotents
            primitives = [self._get_primitive_idempotents(seed, id) for id in new_idempotents]
            return np.vstack(primitives).view(self.field)

    def _get_minimal_polynomial(
        self, element: galois.FieldArray, idempotent: galois.FieldArray
    ) -> tuple[galois.Poly, galois.FieldArray]:
        """Find the minimal polynomial an element within a sub-algebra stabilized by an idempotent.

        The minimal polynomial of α is the lowest-degree monic polynomial m(x) with m(α) = 0.
        Here "monic" means that the m(x) has a leading coefficient of one:
            m(x) = x^a + sum_{j=0}^{a-1} m_j x^j,
        where all coefficients m_j ∈ GF(q), and we define x^0 to be the idempotent.

        This method assumes--and does not verify---that:
            1. The provided idempotent is an idempotent of R.
            2. The provided element lives in the sub-algebra of R stabilized by the idempotent.

        Returns:
            - The minimal polynomial of RingMember(element, self.ring).
            - A matrix in GF(q)^{d × |G|} whose j-th row is element^j ∈ S.
        """

        """
        PART 1: Construct a matrix of linearly independent column vectors: [α^0, α^1, α^2, ...].

        We start with the one-column matrix [α^0], and repeatedly double the size of this matrix by
            [α^0] -> [α^0, α^1] -> [α^0, α^1, α^2, α^4] -> ...
        We stop when the number of columns exceeds the rank r of the matrix, at which point we save
        α^r for later use and throw out all but the first r columns, [α^0, α^1, α^2, ..., α^{r-1}].
        """
        element_mat = self._regular_lift(element)
        powers = idempotent.reshape(-1, 1).view(self.field)
        while True:
            new_powers = element_mat @ powers
            powers = np.column_stack([powers, new_powers]).view(self.field)
            rank = np.linalg.matrix_rank(powers)
            if powers.shape[1] > rank:
                break
            element_mat = element_mat @ element_mat
        extra = powers[:, rank]  # element^rank
        powers = powers[:, :rank]

        """
        PART 2: Construct the minimal polynomial m(x) of α.

        This polynomial is defined by coefficients m_j for which
            m(α) = α^r + sum_{j=0}^{r-1} m_j α^r = 0.
        These coefficients can be found by solving a linear system of equations.
        """
        linear_system = np.column_stack([powers, -extra]).view(self.field)
        poly_coeffs = linear_system.row_reduce()[:rank, -1]
        poly_coeffs = np.append(poly_coeffs, self.field(1))

        return galois.Poly(poly_coeffs[::-1], field=self.field), powers.T

    def _get_off_diagonal_basis_vecs(
        self, basis_ij: galois.FieldArray, basis_ji: galois.FieldArray, seed: np.random.Generator
    ) -> tuple[galois.FieldArray, galois.FieldArray]:
        """Construct standard-basis matrix elements |i><j| and |j><i| of S ≅ GF(q^d)^{n × n}.

        The strategy is as follows:
        1. Sample random vectors x_ij ∈ e_i R e_j and x_ji ∈ e_j R e_i.
        2. Construct y_i = x_ij x_ji = α e_i for some α ∈ GF(q^d).
        3. Compute z = n/|G| Tr_G[y_i] = α e, where Tr_G[y] = sum_{g in G} g y g^{-1}.
        3. Return e_ij = x_ij and e_ji = x_ji / α.

        A cautionary note: for representation-theoretic reasons, e_ji is the "dual" of e_ij in the
        sense that
            e_ij e_ji = e_i,
            e_ji e_ij = e_j,
        but it may not be the case that e_ji = e_ij.T in the sense of RingMember.T, because
        RingMember.T is an involution that inverts group members, which need not have anything to do
        with matrix transposition in GF(q^d)^{n × n}.

        Returns:
            - A vector in GF(q)^{|G|} representing e_ij ∈ S.
            - A vector in GF(q)^{|G|} representing e_ji ∈ S.
        """
        # sample x_ij ∈ e_i R e_j and x_ji ∈ e_j R e_i
        vec_ij = self._random_nonzero_vec(len(basis_ij), seed) @ basis_ij
        vec_ji = self._random_nonzero_vec(len(basis_ji), seed) @ basis_ji

        # construct y_i = α e_i, z = α e, and extract α as a scalar in GF(p^{kd})
        vec_i = self._regular_lift(vec_ij) @ vec_ji
        normalization = (self.field(1) * self.size) / (self.field(1) * self.ring.group.order)
        vec_z = self.ring.group_trace_matrix @ vec_i * normalization
        scalar = self._center_to_scalar(vec_z)

        # return e_ij = x_ij, e_ji = x_ji / α
        scalar_inv_as_mat = self._regular_lift(self._scalar_to_center(scalar ** (-1)))
        return vec_ij, scalar_inv_as_mat @ vec_ji

    def _center_to_scalar(self, vec_in_center: galois.FieldArray) -> galois.FieldArray:
        """Convert a "scalar' s ∈ Z(S) ≅ GF(q}^{|G|} into an element of GF(p^{kd}) ≅ GF(q^d)."""
        power_basis_coeffs = self.power_basis_dual @ vec_in_center
        embedded_power_basis_coeffs = self.embedded_scalars[power_basis_coeffs.view(np.ndarray)]
        return embedded_power_basis_coeffs @ self.embedded_power_basis

    def _scalar_to_center(self, scalar: galois.FieldArray) -> galois.FieldArray:
        """Embed a scalar in GF(p^{kd}) ≅ GF(q^d) back into the center Z(S) ≅ GF(q}^{|G|}."""
        embedded_coefficients = self.extended_field_trace(self.embedded_power_basis_dual * scalar)
        coefficients = self.embedded_scalars_inverse[embedded_coefficients.view(np.ndarray)]
        return coefficients @ self.power_basis

    def _get_decomposition_coefficient_extractor(self) -> galois.FieldArray:
        """Build a matrix that maps elements of S to their GF(q) coefficients in GF(q^d)^{n × n}.

        Consider a ring member r ∈ R ≅ GF(q)^{|G|} whose projection s = r·e ∈ S has the expansion
            s = sum_{i,j,k} s_ijk b^i e_jk ∈ GF(q)^{|G|},
        where each coefficient s_ijk ∈ GF(q).  This method constructs the linear map
            GF(q)^{|G|} -> GF(q)^{n × n × d}
        that takes a ring member r to the coefficients s_ijk.

        Returns:
            - A matrix in GF(q)^{n^2·d × |G|} that maps a ring member r to [s_ijk]_ijk.
        """
        # matrices representing the action of e_ij from multiplication on the left and right
        matrix_basis_as_mats_l = self.field(
            [self._regular_lift(vec) for vec in self.matrix_basis]
        ).reshape(self.size, self.size, self.ring.group.order, self.ring.group.order)
        matrix_basis_as_mats_r = self.field(
            [self._regular_lift(vec, right=True) for vec in self.matrix_basis]
        ).reshape(self.size, self.size, self.ring.group.order, self.ring.group.order)

        # construct a matrix that maps α e_i -> (α_0, α_1, ..., α_{d-1}), where α = sum_j α_j b^j
        normalization = (self.field(1) * self.size) / (self.field(1) * self.ring.group.order)
        get_diagonal_entry_scalar = (
            self.power_basis_dual @ self.ring.group_trace_matrix * normalization
        )

        # take r -> (e_j r e_k e_kj) = s_jk e_j -> s_ijk
        tensor = self.field(
            [
                get_diagonal_entry_scalar
                @ matrix_basis_as_mats_r[kk, jj]
                @ matrix_basis_as_mats_r[kk, kk]
                @ matrix_basis_as_mats_l[jj, jj]
                for jj, kk in itertools.product(range(self.size), repeat=2)
            ]
        )
        return tensor.reshape(self.size**2 * self.degree, self.ring.group.order).view(self.field)

    def _get_decomposition_coefficient_recombiner(self) -> galois.FieldArray:
        """Build a matrix that embeds GF(q) coefficients of GF(q^d)^{n × n} into S.

        The matrix built here is a left pseudo-inverse of that built in
        _get_decomposition_coefficient_extractor.

        Returns:
            - A matrix in GF(q)^{|G| × n^2·d} that maps [s_ijk]_ijk to s ∈ S ≅ GF(q)^{|G|}.
        """
        power_basis_mats = [self._regular_lift(bb) for bb in self.power_basis]
        return self.field(
            [
                power_basis_mats[ii] @ self.matrix_basis[jj_kk]
                for jj_kk in range(self.size**2)
                for ii in range(self.degree)
            ]
        ).T.view(self.field)

    def project(self, element: RingMember) -> galois.FieldArray:
        """Project an element of the parent ring into a simple component S ≅ GF(q^d)^{n × n}."""
        if element.ring is not self.ring:
            raise ValueError(
                "A Wedderburn-Artin transformer initialized for one ring was asked to decompose an"
                " element of a different ring"
            )
        coefficients = self.decomposition_coefficient_extractor @ element.to_vector()
        embedded_coefficients = self.embedded_scalars[coefficients.view(np.ndarray)]
        matrix_values = embedded_coefficients.reshape(-1, self.degree) @ self.embedded_power_basis
        return matrix_values.reshape(self.size, self.size)

    def project_array(self, array: RingArray, *, merge_blocks: bool = False) -> galois.FieldArray:
        """Project a RingArray element-wise into a simple component S ≅ GF(q^d)^{n × n}.

        An N-dimensional RingArray gets projected into an (N+2)-dimensional galois.FieldArray.

        If merge_blocks is True, this method treats each projected element as a block matrix in the
        last two axes of the provided array, such that a projection with shape
            (..., r, c, self.size, self.size)
            is transposed and reshaped into an array with shape
            (..., r * self.size, c * self.size).
        """
        if array.ring is not self.ring:
            raise ValueError(
                "A Wedderburn-Artin transformer initialized for one ring was asked to decompose an"
                " element of a different ring"
            )
        vectors = array.to_field_array().reshape(array.size, self.ring.group.order)
        coefficients = vectors @ self.decomposition_coefficient_extractor.T
        embedded_coefficients = self.embedded_scalars[coefficients.view(np.ndarray)]
        matrix_values = (
            embedded_coefficients.reshape(array.size * self.size**2, self.degree)
            @ self.embedded_power_basis
        )
        matrix_values = matrix_values.reshape(*array.shape, self.size, self.size)
        if merge_blocks:
            assert array.ndim >= 2
            repeat = array.size // (array.shape[-2] * array.shape[-1]) if array.size else 0
            old_shape = (repeat, array.shape[-2], array.shape[-1], self.size, self.size)
            new_shape = (
                *array.shape[:-2],
                array.shape[-2] * self.size,
                array.shape[-1] * self.size,
            )
            matrix_values = (
                matrix_values.reshape(old_shape).transpose(0, 1, 3, 2, 4).reshape(new_shape)
            )
        return matrix_values.view(self.extended_field)

    def embed(self, element: galois.FieldArray) -> RingMember:
        """Invert WedderburnArtinComponentTransformer.project."""
        if type(element) is not self.extended_field or element.shape != (self.size, self.size):
            raise ValueError(r"The provided element does not live in GF(q^d)^{n × n}")
        embedded_coefficients = self.extended_field_trace(
            np.outer(element.ravel(), self.embedded_power_basis_dual).view(self.extended_field)
        ).ravel()
        coefficients = self.embedded_scalars_inverse[embedded_coefficients.view(np.ndarray)]
        vector = self.decomposition_coefficient_recombiner @ coefficients
        return RingMember.from_vector(vector, self.ring)

    def embed_array(self, array: galois.FieldArray, *, from_blocks: bool = False) -> RingArray:
        """Invert WedderburnArtinComponentTransformer.project_array."""
        if from_blocks:
            block_rows, rem_rows = divmod(array.shape[-2], self.size)
            block_cols, rem_cols = divmod(array.shape[-1], self.size)
            assert rem_rows == rem_cols == 0
            repeat = array.size // (array.shape[-2] * array.shape[-1]) if array.size else 0
            old_shape = (repeat, block_rows, self.size, block_cols, self.size)
            new_shape = (*array.shape[:-2], block_rows, block_cols, self.size, self.size)
            array = (array.reshape(old_shape).transpose(0, 1, 3, 2, 4).reshape(new_shape)).view(
                type(array)
            )
        if type(array) is not self.extended_field or array.shape[-2:] != (self.size, self.size):
            raise ValueError(r"The provided array does not store matrices in GF(q^d)^{n × n}")
        embedded_coefficients = self.extended_field_trace(
            np.outer(array.ravel(), self.embedded_power_basis_dual).view(self.extended_field)
        ).reshape(*array.shape[:-2], self.size**2 * self.degree)
        coefficients = self.embedded_scalars_inverse[embedded_coefficients.view(np.ndarray)]
        new_array = coefficients @ self.decomposition_coefficient_recombiner.T
        return RingArray.from_field_array(
            new_array.reshape(*array.shape[:-2], self.ring.group.order), self.ring
        )
