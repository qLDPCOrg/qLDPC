"""Helpers for parsing SymPy monomial and polynomial expressions.

These utilities are shared by the group and group-algebra layers to turn user-supplied SymPy
expressions into their constituent monomials, coefficients, and exponents.  They are pure symbolic
bookkeeping with no dependence on the group machinery, so they live in their own module.

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

import numpy as np
import sympy


def iter_monomial_terms(polynomial: sympy.Basic | int | np.int_) -> tuple[sympy.Expr, ...]:
    """Split a SymPy polynomial into its monomial terms, distributing any products of sums.

    Accepts a sympy.Expr, a sympy.Poly, or a (Python or NumPy) integer, and always returns a tuple
    of single-term monomials.  For example, (1 + x) * (1 + y) becomes (1, x, y, x*y), while a lone
    monomial or constant such as 2 * x or 5 becomes a one-element tuple.
    """
    if isinstance(polynomial, sympy.Poly):
        polynomial = polynomial.as_expr()
    # sympify integers into SymPy objects, then expand so that make_args sees a sum of monomials
    return sympy.Add.make_args(sympy.sympify(polynomial).expand())


def get_coefficient_and_exponents(
    monomial: sympy.Integer | sympy.Symbol | sympy.Pow | sympy.Mul | int | np.int_,
) -> tuple[int, list[tuple[sympy.Symbol, int]]]:
    """Extract the coefficients and exponents in a SymPy monomial expression.

    For example, this method takes 5 * x**3 * y**2 to (5, [(x, 3), (y, 2)]).
    """
    if isinstance(monomial, (sympy.Integer, int, np.int_)):
        return int(monomial), []
    coeff, monomial = monomial.as_coeff_Mul()
    exponents = []
    if isinstance(monomial, sympy.Symbol):
        exponents.append((monomial, 1))
    elif isinstance(monomial, sympy.Pow):
        base, exponent = monomial.as_base_exp()
        exponents.append((base, exponent))
    elif isinstance(monomial, sympy.Mul):
        for factor in monomial.args:
            base, exponent = factor.as_base_exp()
            exponents.append((base, exponent))
    return int(coeff), exponents
