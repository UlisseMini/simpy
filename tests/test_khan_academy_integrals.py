"""
LOL im gonna take a bunch of integral questions from https://www.khanacademy.org/math/integral-calculus/ic-integration/ic-integration-proofs/test/ic-integration-unit-
and make sure simpy can do them
"""

from simpy.debug.test_utils import assert_definite_integral, assert_eq_plusc, assert_eq_value, assert_integral, x, y
from simpy.expr import *
from simpy.integration import *


def test_ex():
    integrand = 6 * e**x
    ans = integrate(integrand, (x, 6, 12))
    assert_eq_plusc(ans, 6 * e**12 - 6 * e**6)


def test_xcosx():
    """Uses integration by parts"""
    integrand = x * cos(x)
    ans = integrate(integrand, (x, 3 * pi / 2, pi))
    assert_eq_plusc(ans, 3 * pi / 2 - 1)


def test_partial_fractions():
    integrand = (x + 8) / (x * (x + 6))
    expected_ans = Fraction(4, 3) * log(abs(x)) - Fraction(1, 3) * log(abs(x + 6))
    assert_integral(integrand, expected_ans)

    integrand = (18 - 12 * x) / (4 * x - 1) / (x - 4)
    expected_ans = -log(abs(4 * x - 1)) - 2 * log(abs(x - 4))
    assert_integral(integrand, expected_ans)
    integrand = (2 * x + 3) / (x - 3) / (x + 3)
    expected_ans = 3 * log(abs(x - 3)) / 2 + log(abs(x + 3)) / 2
    assert_integral(integrand, expected_ans)
    integrand = (x - 2) / (2 * x + 1) / (x + 3)
    expected_ans = -log(abs(2 * x + 1)) / 2 + log(abs(x + 3))
    assert_integral(integrand, expected_ans)


def test_integration_by_parts():
    integrand = x * e ** (-x)
    expected = -(e ** (-x)) * (x + 1)
    assert_integral(integrand, expected)

    integrand = log(x) / x**2
    expected = -log(x) / x - 1 / x
    assert_integral(integrand, expected)

    ans = integrate(x * sqrt(x - y), (x, 0, y))
    assert ans == Fraction(4, 15) * (-y) ** Fraction(5, 2)

    integrand = x * e ** (4 * x)
    assert_definite_integral(integrand, (0, 2), Fraction(7, 16) * e**8 + Fraction(1, 16))

    assert_definite_integral(-x * cos(x), (pi / 2, pi), 1 + pi / 2)

    # Challenge questions
    integrand = e**x * sin(x)
    expected = e**x / 2 * (sin(x) - cos(x))
    assert_integral(integrand, expected)

    integrand = x**2 * sin(pi * x)
    expected = -(x**2) * cos(pi * x) / pi + 2 * x * sin(pi * x) / pi**2 + 2 * cos(pi * x) / pi**3
    assert_integral(integrand, expected)


def test_arcsin():
    ans = integrate(asin(x), x)
    expected_ans = x * asin(x) + sqrt(1 - x**2)
    assert_eq_plusc(ans, expected_ans)

    ans = integrate(acos(x), x)
    expected_ans = x * acos(x) - sqrt(1 - x**2)
    assert_eq_plusc(ans, expected_ans)

    ans = integrate(atan(x), x)
    expected_ans = x * atan(x) - log(abs(1 + x**2)) / 2
    assert_eq_plusc(ans, expected_ans)


def test_sec2x_tan2x():
    """Uses either integration by parts with direct solve or generic sin/cos usub"""
    integrand = sec(2 * x) * tan(2 * x)
    ans = integrate(integrand, (x, 0, pi / 6))
    assert ans == Fraction(1, 2)


def test_misc():
    assert_integral(4 * sec(x) ** 2, 4 * tan(x))
    assert_integral(sec(x) ** 2 * tan(x) ** 2, tan(x) ** 3 / 3)
    assert_integral(5 / x - 3 * e**x, 5 * log(abs(x)) - 3 * e**x)
    assert_integral(sec(x), log(sec(x) + tan(x)))  # TODO: should this be abs?
    assert_integral(2 * cos(2 * x - 5), sin(2 * x - 5))
    assert_integral(3 * x**5 - x**3 + 6, 6 * x - x**4 / 4 + x**6 / 2)
    assert_integral(x**3 * e ** (x**4), (e ** (x**4) / 4))

    assert_definite_integral(8 * x / sqrt(1 - 4 * x**2), (0, Fraction(1, 4)), 2 - sqrt(3))
    assert_definite_integral(sin(4 * x), (0, pi / 4), Fraction(1, 2))
    assert_integral(exp(x) / (1 + exp(2 * x)), atan(exp(x)))


def test_usub():
    # Uses generic u-sub
    assert_definite_integral(e**x / (1 + e**x), (log(2), log(8)), log(9) - log(3))

    # ideally have this work with just log(x)
    assert_definite_integral(log(abs(x)) ** 2 / x, bounds=(1, e), expected=Rat(1, 3))


def test_csc_x_squared():
    integrand = 5 * csc(x) ** 2
    expected_ans = -5 * cot(x)
    assert_integral(integrand, expected_ans)


def test_csc_x_cot_x():
    # this one requires simpy knowing that 1/sin(x) and csc(x) are the same
    integrand = 2 * csc(x) * cot(x)
    expected = -2 * csc(x)
    assert_integral(integrand, expected)


def test_expanding_big_power():
    integrand = (2 * x - 5) ** 10
    expected_ans = (2 * x - 5) ** 11 / 22
    assert_integral(integrand, expected_ans)

    integrand = 3 * x**2 * (x**3 + 1) ** 6
    expected_ans = (1 + x**3) ** 7 / 7
    assert_integral(integrand, expected_ans)


def test_polynomial_div_integrals():
    expr = (x - 5) / (-2 * x + 2)
    expected = -x / 2 + 2 * log(abs(1 - x))
    assert_integral(expr, expected)
    assert_integral((x**3 - 1) / (x + 2), x**3 / 3 - x**2 + 4 * x - 9 * log(abs(2 + x)))
    assert_integral((x - 1) / (2 * x + 4), x / 2 - Fraction(3, 2) * log(abs(x + 2)))

    integrand = (2 * x**3 + 4 * x**2 - 5) / (x + 3)
    ans = integrate(integrand, x)
    # TODO: expected = ...


def test_complete_the_square_integrals():
    assert_integral(1 / (3 * x**2 + 6 * x + 78), atan((1 + x) / 5) / 15)
    assert_integral(1 / (x**2 - 8 * x + 65), atan((-4 + x) / 7) / 7)
    assert_integral(1 / sqrt(-(x**2) - 6 * x + 40), asin((3 + x) / 7))

    assert_definite_integral(1 / (1 + 9 * x**2), (-Rat(1, 3), Rat(1, 3)), expected=pi / 6)


def test_neg_inf():
    assert integrate(-(e**x), (-inf, 1)) == -e
    assert integrate(e ** (-x), (0, inf)) == 1


import pytest


@pytest.mark.xfail
def test_neg_inf_2():
    # This one requires recognizing that x * e**-x at infinity is 0 because the e term goes down faster
    # options:
    # - auto do it (sympy does this)
    # - make a seperate class for limits and especially evaluate a limit when one of the bounds is 0
    #   - lwk i like this better because inf feels wonky ngl and having it as an explicit limit...
    #   - like setting inf * e ** -inf is like lowkey fair ngl if we don't know? context?
    # tho i do think having inf * 2 = inf always is fair or smtn. linear shit doesn't change scaling.
    # but don't combine powers by default or something.
    for n in range(4):
        assert integrate(x**n * e ** (-x), (0, inf)) == math.factorial(n)


def test_bigger_power_trig():
    # uses product-to-sum on bigger powers:
    expr = sin(x) ** 4
    expected = (sin(4 * x) - 8 * sin(2 * x) + 12 * x) / 32
    assert_integral(expr, expected)


def test_bigger_power_trig_2():
    # both e1 and e2 are correct answers.
    e1 = sin(2 * x) ** 3 / 48 + 3 * sin(4 * x) / 64 - sin(2 * x) / 4 + 5 * x / 16
    e2 = (9 * sin(4 * x) - sin(6 * x) - 45 * sin(2 * x) + 60 * x) / 192
    assert_integral(sin(x) ** 6, (e1, e2))


def test_rewrite_pythag():
    expr = sin(x) ** 2 * cos(x) ** 3
    # this one still takes ~.9s to complete, which is quite long & much longer than any other integral in our tests as of 05/10.
    expected_ans = sin(x) ** 3 / 3 - sin(x) ** 5 / 5
    assert_integral(expr, expected_ans)


def test_rewrite_pythag_2():
    assert_integral(sin(x) ** 3, cos(x) ** 3 / 3 - cos(x))
    assert_integral(cos(x) ** 5, sin(x) ** 5 / 5 - 2 * sin(x) ** 3 / 3 + sin(x))


def test_tan_x_4():
    # this would take forever if i don't have node.add_child
    # when sin^4x/cos^4x on the first one, it never goes onto the inversetrigusub.
    ans = integrate(tan(x) ** 4, (0, pi / 4))
    assert_eq_value(ans, pi / 4 - Fraction(2, 3))


def test_more_complicated_trig():
    expr = tan(x) ** 5 * sec(x) ** 4
    expected_ans = tan(x) ** 6 / 6 + tan(x) ** 8 / 8
    assert_integral(expr, expected_ans)


def test_abs():
    expr = abs(-2 * x + 4)
    assert_definite_integral(expr, bounds=(-2, 4), expected=20)


def test_piecewise():
    expr = Piecewise((9 * sqrt(x), 0, inf), (-2 * x, -inf, 0), var=x)
    assert_definite_integral(expr, bounds=(-3, 1), expected=15)

    expr = Piecewise((3 * x**2 - 1, 0, inf), (6 * x - 1, -inf, 0), var=x)
    assert_definite_integral(expr, bounds=(-1, 1), expected=-4)

    expr = Piecewise((1 / x, 1, inf), (x, -inf, 1), var=x)
    assert_definite_integral(expr, bounds=(0, 3), expected=log(3) + Rat(1, 2))
