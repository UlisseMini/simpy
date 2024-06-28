"""latex is deployed to production now so I'll test it somewhat more rigorously."""

from simpy import *
from simpy.debug.test_utils import *


def assert_latex(expr: Expr, expected_latex: str):
    result = latex(expr)
    assert result.replace(" ", "") == expected_latex.replace(" ", "")


def test_no_unnecessary_curly_brackets():
    assert_latex(Rat(1, 2), "\\frac12")
    assert latex(x / 2) == "\\frac x2"  # for this one you have to make sure there is space
    assert_latex((x + 2) / 3, "\\frac{x+2}3")

    assert_latex(x**2, "x^2")
    assert_latex((2 + x) ** (log(x)), "(x+2)^{\\ln(x)}")


def test_power_groups():
    assert_latex(x ** (y + 3), "x^{y+3}")
    assert_latex((2 + x) ** y, "(x+2)^y")
    assert_latex(x ** (-3), "x^{-3}")
    assert_latex(x ** (-e), "x^{-e}")


def test_sum_subtraction_bug():
    assert_latex((x - 5) / 2, "\\frac{x - 5}2")


def test_readme():
    # I'll test some examples in the readme I guess.
    assert_latex(sec(2 * x), "\\sec(2 \\cdot x)")
    assert_latex(atan(x), "\\tan^{-1}(x)")
    assert_latex(2 * log(abs(1 - x)) - x / 2, "2 \\cdot \\ln (| -x + 1 |) - \\frac x2")
    assert_latex(
        -log(abs(x**2 + 1)) / 2 + x * atan(x),
        "-\\frac{\\ln \\left( \\left| x^2 + 1 \\right| \\right)}2 + x \\cdot \\tan^{-1} (x)",
    )


def test_big_frac():
    assert_latex(tan((2 + x) / (x + 3)), "\\tan \left(\\frac{x + 2}{x + 3} \\right)")
