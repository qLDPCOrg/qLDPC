"""Unit tests for _monomials.py.

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
