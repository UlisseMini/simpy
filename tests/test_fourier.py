# finding the fourier series of various fucking functions
from typing import Literal

import simpy as sp
from simpy.debug.test_utils import eq_float


def get_fourier_series(
    f: sp.expr.Expr, T: sp.expr.Expr, x: sp.expr.Symbol, *, settings: Literal["center", "right"] = "right"
):
    """gets the fourier series of a 1d function

    args:
        f: the function to get the fourier series of
        T: period of the function
        x: variable of the function

    returns a tuple of:
        a_0: coeff of constant
        a_n: coeff of cos terms
        b_n: coeff of sin terms

    a_0 = 1 \cdot f
    a_n = 2 * cos(2nx*pi/T) \cdot f
    b_n = 2 * sin(2nx*pi/T) \cdot f

    where \cdot is the dot product defined as
    f \cdot g = 1/T * integrate over one period(f*g)
    """
    c_n = sp.cos(2 * sp.pi * n * x / T)
    s_n = sp.sin(2 * sp.pi * n * x / T)

    bounds = (x, 0, T) if settings == "right" else (x, -T / 2, T / 2)

    a_0 = sp.integrate(f, bounds) / T
    a_n = 2 * sp.integrate(c_n * f, bounds) / T
    b_n = 2 * sp.integrate(s_n * f, bounds) / T

    # make the summation
    summation = a_0
    for i in range(1, 5):
        subs = {"n": i}
        summation += a_n.subs(subs) * c_n.subs(subs)
        summation += b_n.subs(subs) * s_n.subs(subs)

    return a_0, a_n, b_n, summation


a, n, x = sp.symbols("a n x")
n._is_int = True


def test_q3():
    """MATH 2410 Recitation #5 question 3
    Piecewise fn can't be integrated directly bc it has a variable bound.
    """

    f = sp.sin(sp.pi * x / a)
    c_n = sp.cos(2 * n * x)
    s_n = sp.sin(2 * n * x)

    a_0 = 1 / sp.pi * sp.integrate(f, (x, 0, a))
    a_n = 2 / sp.pi * sp.integrate(c_n * f, (x, 0, a))
    b_n = 2 / sp.pi * sp.integrate(s_n * f, (x, 0, a))

    # for some reason a_n is negative the one from class notes idk why
    assert a_0 == 2 * a / sp.pi**2

    expected_an = 2 * a * (1 + sp.cos(2 * a * n)) / (sp.pi**2 - (2 * a * n) ** 2)
    expected_bn = 2 * a * sp.sin(2 * a * n) / (sp.pi**2 - (2 * a * n) ** 2)

    for i in range(1, 5):
        assert eq_float(a_n.evalf({"n": i}), expected_an.evalf({"n": i}))
        assert eq_float(b_n.evalf({"n": i}), expected_bn.evalf({"n": i}))

    # T = sp.pi
    # fp = sp.expr.Piecewise((f, 0, a))
    # a0, an, bn, summation = get_fourier_series(fp, T, x)


def test_odd_function():
    """MATH 2410 Recitation #5 question 2"""

    f = (sp.pi - x) / 2
    T = 2 * sp.pi

    a_0, a_n, b_n, summation = get_fourier_series(f, T, x)
    assert a_0 == 0 == a_n
    assert b_n == 1 / n


def test_even_function():
    """MATH 2410 Recitation #5 question 1
    even piecewise function with period 4
    """
    f = sp.expr.Piecewise((0, -2, -1), (1 + x, -1, 0), (1 - x, 0, 1), var=x)
    T = 4
    a0, an, bn, summation = get_fourier_series(f, T, x, settings="center")
    # for an even function, all the b_n terms are zero
    assert a0 == 1 / 4
    expected_an = 8 * sp.sin(n * sp.pi / 4) ** 2 / (n * sp.pi) ** 2
    for i in range(1, 5):
        assert eq_float(an.evalf({"n": i}), expected_an.evalf({"n": i}))