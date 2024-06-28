from dataclasses import dataclass

from .expr import Expr, Symbol


@dataclass
class Equation:
    lhs: Expr
    rhs: Expr


def solve(equation: Equation, x: Symbol):
    # Ensure the equation is set to 0 (ax + b = 0)
    expr = equation.lhs - equation.rhs

    # Initialize coefficients
    coeff_a = 0
    coeff_b = 0

    # Iterate over the terms of the expression
    for term in expr.as_terms():
        if (term / x).symbolless:
            coeff_a += term / x
        elif not term.contains(x):
            coeff_b += term
        else:
            raise NotImplementedError("can only solve linear eqs")

    # Calculate the solution
    if coeff_a == 0:
        if coeff_b == 0:
            return "Infinite solutions (identity equation)."
        else:
            return "No solution (contradictory equation)."

    x_solution = -coeff_b / coeff_a
    return x_solution
