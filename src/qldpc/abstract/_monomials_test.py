# SPDX-License-Identifier: Apache-2.0

"""Unit tests for _monomials.py."""

from __future__ import annotations

import sympy

from qldpc import abstract


def test_get_coefficient_and_exponents() -> None:
    """Parse SymPy monomial expressions."""
    x = sympy.abc.x
    y = sympy.abc.y
    assert abstract.get_coefficient_and_exponents(3) == (3, [])
    assert abstract.get_coefficient_and_exponents(x) == (1, [(x, 1)])
    assert abstract.get_coefficient_and_exponents(x**2) == (1, [(x, 2)])
    assert abstract.get_coefficient_and_exponents(3 * x * y**2) == (3, [(x, 1), (y, 2)])


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

    # a sympy.Poly is accepted and treated the same as its expression
    assert set(abstract.iter_monomial_terms(sympy.Poly(x**2 + y, x, y))) == {x**2, y}
