"""
LOL im gonna take a bunch of integral questions from https://www.khanacademy.org/math/integral-calculus/ic-integration/ic-integration-proofs/test/ic-integration-unit-
and make sure simpy can do them
"""
import pytest

from src.simpy.expr import *
from src.simpy.integration import *
from test_utils import (assert_definite_integral, assert_eq_plusc,
                        assert_eq_value, assert_integral, x, y)


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

    integrand = (18-12*x)/(4*x-1)/(x-4)
    expected_ans = -log(abs(4*x-1))-2*log(abs(x-4))
    assert_integral(integrand, expected_ans)
    integrand = (2*x+3)/(x-3)/(x+3)
    expected_ans = 3*log(abs(x-3))/2 + log(abs(x+3))/2
    assert_integral(integrand, expected_ans)
    integrand = (x-2)/(2*x+1)/(x+3)
    expected_ans = -log(abs(2*x+1))/2 + log(abs(x+3))
    assert_integral(integrand, expected_ans)


def test_integration_by_parts():
    integrand = x * e ** (-x)
    expected = -e ** (-x) * (x + 1)
    assert_integral(integrand, expected)

    integrand = log(x) / x ** 2
    expected = -log(x) / x - 1 / x
    assert_integral(integrand, expected)

    ans = integrate(x*sqrt(x-y), (x, 0, y))
    assert ans == Fraction(4, 15) * (-y) ** Fraction(5, 2)

    integrand =  x * e ** (4*x)
    assert_definite_integral(integrand, (0, 2), Fraction(7, 16) * e ** 8 + Fraction(1, 16))

    assert_definite_integral(-x * cos(x), (pi/2, pi), 1 + pi/2)

    # Challenge questions
    integrand = e ** x * sin(x)
    expected = e ** x / 2 * (sin(x) - cos(x))
    assert_integral(integrand, expected)

    integrand = x ** 2 * sin(pi * x) 
    expected = -x**2 * cos(pi * x) / pi + 2 * x * sin(pi*x) / pi ** 2 + 2*cos(pi*x)/pi**3
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
    integrand = sec(2*x) * tan(2*x)
    ans = integrate(integrand, (x, 0, pi/6))
    assert ans == Fraction(1, 2)

def test_misc():
    assert_integral(4 * sec(x) ** 2, 4 * tan(x))
    assert_integral(sec(x) ** 2 * tan(x) ** 2, tan(x) ** 3 / 3)
    assert_integral(5 / x - 3 * e ** x, 5 * log(abs(x)) - 3 * e ** x)
    assert_integral(sec(x), log(sec(x) + tan(x))) # TODO: should this be abs?
    assert_integral(2 * cos(2 * x - 5), sin(2 * x - 5)) 
    assert_integral(3 * x ** 5 - x ** 3 + 6, 6*x - x**4/4 + x**6/2)
    assert_integral(x ** 3 * e ** (x ** 4), (e**(x**4)/4))

    # Uses generic u-sub
    assert_definite_integral(e ** x / (1 + e ** x), (log(2), log(8)), log(9) - log(3))

    assert_definite_integral(8 * x / sqrt(1 - 4 * x ** 2), (0, Fraction(1,4)), 2 - sqrt(3))
    assert_definite_integral(sin(4*x), (0, pi/4), Fraction(1,2))

@pytest.mark.xfail
def test_csc_x_squared():
    # idk how to simplify the answer to this one.
    # requires simplifying 1/sinxcosx - tanx -> cotx ??

    # takes forever
    integrand = 5 * csc(x) ** 2
    expected_ans = -5 * cot(x)
    assert_integral(integrand, expected_ans)

@pytest.mark.xfail
def test_csc_x_cot_x():
    # this one requires simpy knowing that 1/sin(x) and csc(x) are the same
    integrand = 2 * csc(x) * cot(x)
    expected = -2 * csc(x)
    assert_integral(integrand, expected)


def test_expanding_big_power():
    integrand = (2 * x - 5) ** 10
    expected_ans = (2*x-5)**11/22
    assert_integral(integrand, expected_ans)

    integrand = 3 * x ** 2 * (x ** 3 + 1) ** 6
    expected_ans = (1 + x**3)**7/7
    assert_integral(integrand, expected_ans)


def test_polynomial_div_integrals():
    expr = (x-5) / (-2 * x + 2)
    expected =  -x/2 + 2 * log(abs(1 - x))
    assert_integral(expr, expected)
    assert_integral((x ** 3 - 1)/ (x+2), x**3/3 - x**2 + 4*x- 9*log(abs(2 + x)))
    assert_integral((x - 1)/ (2 * x + 4), x / 2 - Fraction(3, 2) * log(abs(x + 2)))
    
    integrand = (2 * x ** 3 + 4 * x ** 2 - 5)/ (x + 3)
    ans = integrate(integrand, x)
    # TODO: expected = ...

def test_complete_the_square_integrals():
    assert_integral(1/(3*x**2+6*x+78), atan((1 + x)/5)/15)
    assert_integral(1/(x**2-8*x+65), atan((-4 + x)/7)/7)
    assert_integral(1/sqrt(-x**2-6*x+40), asin((3 + x)/7))


def test_neg_inf():
    assert integrate(-e ** x, (-oo, 1)) == -e


def test_bigger_power_trig():
    # uses product-to-sum on bigger powers:
    expr = sin(x) ** 4
    expected = (sin(4*x) - 8*sin(2*x) + 12*x) / 32
    assert_integral(expr, expected)


def test_rewrite_pythag():
    expr = sin(x) ** 2 * cos(x) ** 3
    ## USED TO
    # returns the right answer, just with a tree 27 layers deep and with a very long expression
    # probs assert equality by implemeting product-to-sum and compound angle?? which one tho? we can experiment.
    # sin(x)cos(4x)/120 + sin(x)cos(2x)/12 - cos(x)sin(4x)/30 - cos(x)sin(2x)/6 - sin(x)^3/6 + 3sin(x)/8
    ## USED TO
    # ^ im proud this no longer happens <3 i didnt even optimize for it; i just made the changes to
    # changing depth first to breadth first when expression became too complicated <3
    # and this happend as a bypproduct!
    # this means that the decision was a good architectural decision; not a narrow patch that only fixes that one specific case.
    
    # this one still takes ~.9s to complete, which is quite long & much longer than any other integral in our tests as of 05/10.
    expected_ans = sin(x) ** 3 / 3 - sin(x) ** 5 / 5
    assert_integral(expr, expected_ans)

    assert_integral(sin(x)**3, cos(x)**3/3 - cos(x))
    assert_integral(cos(x)**5, sin(x)**5/5 - 2*sin(x)**3/3 + sin(x))

def test_tan_x_4():
    # this would take forever if i don't have node.add_child 
    # when sin^4x/cos^4x on the first one, it never goes onto the inversetrigusub.
    ans = integrate(tan(x)**4, (0, pi/4))
    assert_eq_value(ans, pi/4 - Fraction(2,3))


@pytest.mark.xfail
def test_more_complicated_trig():
    expr = tan(x) ** 5 * sec(x) ** 4
    expected_ans = tan(x) ** 6 / 6 + tan(x) ** 8 / 8
    assert_integral(expr, expected_ans)
    # the answer is the correct value but it does not simplify it to the expected answer
    # 1/(4cos(x)^4) - 1/(3cos(x)^6) + 1/(8cos(x)^8)
