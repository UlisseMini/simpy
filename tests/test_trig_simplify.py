from simpy import *
from simpy.debug.test_utils import *


def assert_simplified(e1: Expr, e2: Expr):
    assert_eq_strict(e1.simplify(), e2)


def test_simple():
    assert_simplified(1 / sin(x), csc(x))
    assert_simplified(1 / cos(x), sec(x))
    assert_simplified(sin(x) / cos(x), tan(x))


def test_simple_2():
    assert_simplified(2 * sin(x) / cos(x), 2 * tan(x))
    assert_simplified(-sin(x) / cos(x), -tan(x))
    assert_simplified(2 / sin(x), 2 * csc(x))
    assert_simplified(-1 / sin(x), -csc(x))


"""Pythagorean simplifications"""


def test_simplify_sin2x_plus_cos2x():
    expr = sin(x) ** 2 + cos(x) ** 2 + 3
    assert_simplified(expr, 4)

    # Test when sin^2(...) -> ... is more complicated than just 'x'
    expr = sin(x - 2 * y) ** 2 + 3 + cos(x - 2 * y) ** 2 + y**2 + cos(x)
    assert_simplified(expr, 4 + y**2 + cos(x))

    # Test when sin^2(...) and cos^2(...) share a common factor
    expr = 2 * y * sin(x**3) ** 2 + 2 * y * cos(x**3) ** 2
    assert_simplified(expr, 2 * y)


def test_simplify_one_minus_sin_squared():
    # this one used to fail when I only allowed the entire sum to be 1 - sin^2(...)
    expr = sin(x) ** 2 - 5 * cos(x) ** 2 - 1
    assert_simplified(expr, -6 * cos(x) ** 2)


def test_one_plus_tan_squared():
    expr = 1 + tan(x + y) ** 2
    assert_simplified(expr, sec(x + y) ** 2)

    expr = 2 + 2 * tan(x + y) ** 2
    assert_simplified(expr, 2 * sec(x + y) ** 2)


def test_one_minus_sin_squared_up_to_sum():
    expr = 1 - sin(x) ** 2 + x
    assert_simplified(expr, x + cos(x) ** 2)


"""Other things

These didn't pass when I didn't have 'heuristic simplifications'
"""


def test_pts():
    # applying product-to-sum makes this simpler
    e1 = 2 * cos(x) * sin(2 * x) / 3 - sin(x) * cos(2 * x) / 3
    e2 = sin(3 * x) / 6 + sin(x) / 2
    assert_simplified(e1, e2)


def test_sectan():
    # need to replace sec^2(x) with tan^2(x) + 1
    e1 = 1 / (4 * cos(x) ** 4) - 1 / (3 * cos(x) ** 6) + 1 / (8 * cos(x) ** 8)
    e2 = tan(x) ** 6 / 6 + tan(x) ** 8 / 8
    assert_eq_plusc(e1, e2)


def test_csc_squared_deriv():
    # need to write in terms of a common denominator
    e1 = sin(x) / cos(x) - csc(x) / cos(x)
    # this shit takes 2 rounds of simplification: first to -cos(x)/sin(x) then to -cot(x)
    # make sure that both rounds are done in the simplify function.
    e2 = -cot(x)
    assert_simplified(e1, e2)


def test_complicated_pts():
    # old solution of sin(x) ** 2 * cos(x) ** 3 simpy found when going 27 levels deep
    # needs to compound angle to the end or pts to the end.
    e1 = (
        sin(x) * cos(4 * x) / 120
        + sin(x) * cos(2 * x) / 12
        - cos(x) * sin(4 * x) / 30
        - cos(x) * sin(2 * x) / 6
        - sin(x) ** 3 / 6
        + 3 * sin(x) / 8
    )
    e2 = sin(x) ** 3 / 3 - sin(x) ** 5 / 5
    assert_eq_value(e1, e2)


def test_complicated_pts_2():
    """need to have sin(2*x)^3/48 + sin(6*x)/192 - sin(2*x)/64 simplify to 0.
    need to do expand in simplify when that makes it simpler.
    """

    e1 = sin(2 * x) ** 3 / 48 + 3 * sin(4 * x) / 64 - sin(2 * x) / 4 + 5 * x / 16
    e2 = (9 * sin(4 * x) - sin(6 * x) - 45 * sin(2 * x) + 60 * x) / 192
    assert_eq_value(e1, e2)
