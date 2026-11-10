# SPDX-License-Identifier: Apache-2.0

"""Unit tests for monomials.py."""

from __future__ import annotations

import numpy as np
import pytest
import sympy
import sympy.abc

from qldpc import abstract


def test_get_coefficient_and_exponents() -> None:
    """Parse SymPy monomial expressions."""
    x = sympy.abc.x
    y = sympy.abc.y
    assert abstract.get_coefficient_and_exponents(3) == (3, [])
    assert abstract.get_coefficient_and_exponents(np.int64(3)) == (3, [])
    assert abstract.get_coefficient_and_exponents(x) == (1, [(x, 1)])
    assert abstract.get_coefficient_and_exponents(x**2) == (1, [(x, 2)])
    assert abstract.get_coefficient_and_exponents(3 * x * y**2) == (3, [(x, 1), (y, 2)])
    assert abstract.get_coefficient_and_exponents(-x / y**2) == (-1, [(x, 1), (y, -2)])
    assert abstract.get_coefficient_and_exponents(2.0 * x**2.0) == (2, [(x, 2)])

    # the factors of a non-commutative monomial keep their order
    a, b = sympy.symbols("a b", commutative=False)
    assert abstract.get_coefficient_and_exponents(b * a**2 * b) == (1, [(b, 1), (a, 2), (b, 1)])

    # coefficients must be integers, and the other factors integer powers of symbols
    for monomial in [1.5 * x, sympy.Rational(1, 2) * x]:
        with pytest.raises(ValueError, match=r"coefficient of .* is not an integer"):
            abstract.get_coefficient_and_exponents(monomial)
    for monomial in [sympy.I * x, sympy.sin(x), sympy.pi, sympy.sqrt(x), x**y, sympy.exp(x) * y]:
        with pytest.raises(ValueError, match="integer powers of symbols"):
            abstract.get_coefficient_and_exponents(monomial)


def test_iter_monomial_terms() -> None:
    """Split SymPy polynomials into their monomial terms, distributing products of sums."""
    x = sympy.abc.x
    y = sympy.abc.y

    # a sum, a product of sums, a lone monomial, a bare integer, and zero
    assert set(abstract.iter_monomial_terms(x**2 + y)) == {x**2, y}
    assert set(abstract.iter_monomial_terms((1 + x) * (1 + y))) == {sympy.Integer(1), x, y, x * y}
    assert abstract.iter_monomial_terms(2 * x * y) == (2 * x * y,)
    assert abstract.iter_monomial_terms(5) == (sympy.Integer(5),)
    assert abstract.iter_monomial_terms(0) == (sympy.Integer(0),)

    # integer-valued float exponents are read as integers, and other exponents are left alone
    assert set(abstract.iter_monomial_terms((1 + x) ** 2.0)) == {sympy.Integer(1), 2 * x, x**2}
    assert abstract.iter_monomial_terms((1 + x) ** 0.5) == ((1 + x) ** 0.5,)

    with pytest.raises(TypeError, match="not a SymPy expression"):
        abstract.iter_monomial_terms(sympy.Eq(x, y))

    # a sympy.Poly is accepted and treated the same as its expression
    assert set(abstract.iter_monomial_terms(sympy.Poly(x**2 + y, x, y))) == {x**2, y}
