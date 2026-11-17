# SPDX-License-Identifier: Apache-2.0

"""Helpers for parsing SymPy monomial and polynomial expressions.

These utilities turn user-supplied SymPy expressions into their constituent monomials, coefficients,
and exponents, for group algebras and two-block codes.  They are pure symbolic bookkeeping with no
dependence on the group machinery, so they live in their own module.
"""

from __future__ import annotations

import numpy as np
import sympy


def iter_monomial_terms(
    polynomial: sympy.Expr | sympy.Poly | int | np.integer,
) -> tuple[sympy.Expr, ...]:
    """Split a SymPy polynomial into its monomial terms, distributing any products of sums.

    Accepts a sympy.Expr, a sympy.Poly, or a (Python or NumPy) integer, and always returns a tuple
    of single-term monomials.  For example, (1 + x) * (1 + y) becomes (1, x, y, x*y), while a lone
    monomial or constant such as 2 * x or 5 becomes a one-element tuple.  Integer-valued float
    exponents are read as integers, so (1 + x)**2.0 is expanded as (1 + x)**2.  Raise a TypeError if
    SymPy reads the input as something other than an expression, such as an equation.
    """
    # expand so that make_args sees a sum of monomials
    return sympy.Add.make_args(_read_polynomial(polynomial).expand())


def get_coefficient_and_exponents(
    monomial: sympy.Integer | sympy.Symbol | sympy.Pow | sympy.Mul | int | np.int_,
) -> tuple[int, list[tuple[sympy.Symbol, int]]]:
    """Extract the coefficient and exponents in a SymPy monomial expression.

    For example, this method takes 5 * x**3 * y**-2 to (5, [(x, 3), (y, -2)]).  The factors are
    listed in the order of the monomial, which matters for non-commutative symbols.  Integer-valued
    floats are read as integers.  Raise a ValueError unless the monomial is an integer times a
    product of integer powers of symbols.
    """
    coeff, content = sympy.sympify(monomial).as_coeff_Mul()
    if not _is_integer(coeff):
        raise ValueError(f"The coefficient of {monomial} is not an integer")
    exponents = []
    for factor in sympy.Mul.make_args(content):
        if factor == 1:
            continue
        base, exponent = factor.as_base_exp()
        if not (isinstance(base, sympy.Symbol) and _is_integer(exponent)):
            raise ValueError(f"{monomial} is not a product of integer powers of symbols")
        exponents.append((base, int(exponent)))
    return int(coeff), exponents


def _read_polynomial(polynomial: sympy.Expr | sympy.Poly | int | np.integer) -> sympy.Expr:
    """Read a polynomial as a SymPy expression, with integer-valued float exponents as integers."""
    if isinstance(polynomial, sympy.Poly):
        polynomial = polynomial.as_expr()
    expression = sympy.sympify(polynomial)
    if not isinstance(expression, sympy.Expr):
        raise TypeError(
            f"Cannot read {polynomial} as a polynomial, because it is not a SymPy expression"
        )
    # SymPy expands powers of sums only with integer exponents
    return expression.replace(
        lambda node: node.is_Pow and _is_integer(node.exp) and not node.exp.is_Integer,
        lambda node: sympy.Pow(node.base, int(node.exp)),
    )


def _is_integer(value: object) -> bool:
    """Is a number, such as a SymPy exponent, an integer or an integer-valued float?"""
    number = sympy.sympify(value)
    return bool(number.is_Integer or (number.is_Float and float(number).is_integer()))
