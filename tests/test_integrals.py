from simpy.debug.test_utils import (
    assert_definite_integral,
    assert_eq_plusc,
    assert_eq_strict,
    assert_eq_value,
    assert_integral,
    x,
    y,
)
from simpy.expr import *
from simpy.integration import *

F = Fraction
x = symbols("x")


def test_basic_integrals():
    assert_definite_integral(2, (x, 5, 3), -4)
    assert_integral(x ** Fraction(7, 3), Fraction(3, 10) * x ** Fraction(10, 3))
    assert_integral(3 * x**2 - 2 * x, x**3 - x**2)
    assert_integral((x + 1) ** 2, x + x**2 + (x**3 / 3))
    assert_integral(x**12, x**13 / 13)
    assert_integral(1 / x, log(abs(x)))
    assert_definite_integral(1 / x, (x, 1, 2), log(2))
    assert_integral(y, x * y, var=x)
    assert_integral(tan(y), x * tan(y), var=x)


def test_stat_polynomials():
    I1 = integrate((x / 90 * (x - 5) ** 2 / 350), (x, 5, 6))
    I2 = integrate((F(1, 15) - F(1, 360) * (x - 6)) * (x - 5) ** 2 / 350, (x, 6, 15))
    I3 = integrate((F(1, 15) - F(1, 360) * (x - 6)) * (1 - (40 - x) ** 2 / 875), (x, 15, 30))
    assert (I1, I2, I3) == (F(23, 378000), F(2589, 56000), F(37, 224))


def test_lecture_example():
    """The integral from the MIT OCW lecture that inspired this project:
    https://ocw.mit.edu/courses/6-034-artificial-intelligence-fall-2010/resources/lecture-2-reasoning-goal-trees-and-problem-solving/

    Highly recommend watching this to understand how this program performs integrals.
    """
    expression = -5 * x**4 / (1 - x**2) ** F(5, 2)
    expected_integral = -5 * (-(x / (sqrt(1 - x**2))) + (F(1, 3) * x**3 / (1 - x**2) ** F(3, 2)) + asin(x))
    assert_integral(expression, expected_integral)


def test_x2_sqrt_1_x3():
    expression = x**2 / sqrt(1 - x**3)
    expected = -2 * sqrt(1 - x**3) / 3
    assert_integral(expression, expected)


def test_compound_angle():
    # Finds the average power of an AC circuit
    w, phi, t = symbols("w \\phi t")
    ac_power_integrand = cos(w * t - phi) * cos(w * t)
    period = 2 * pi / w
    ac_power = 1 / period * integrate(ac_power_integrand, (t, 0, period))

    expected = cos(phi) / 2
    assert_eq_plusc(ac_power, expected)


def test_sin2x():
    # tests the product-to-sum formula
    # Test integral (sin x)^2 = x / 2 - Sin(2x) / 4
    expr = sin(x) ** 2
    expected = x / 2 - sin(2 * x) / 4
    assert_integral(expr, expected)


def test_cos2x():
    # tests the product-to-sum formula
    # Test integral (cos x)^2 = Sin(2x) / 4 + x / 2
    expr = cos(x) ** 2
    expected = sin(2 * x) / 4 + x / 2
    assert_integral(expr, expected)


def test_linear_usub_with_multiple_subs():
    # Last I checked, this fails without LinearUSub
    integrand = sin(2 * x) / cos(2 * x)
    expected = -log(abs(cos(2 * x))) / 2
    expected2 = log(abs(sec(2 * x))) / 2
    # can't fucking implement this being equal. wtv.
    integral = integrate(integrand)
    assert integral == expected or integral == expected2


def test_misc():
    # This integral can either be sin^2(wt) / 2w or -cos^2(wt) / 2w depending on the method used to solve it
    w, t = symbols("w t")
    expr = sin(w * t) * cos(w * t)
    expected = sin(w * t) ** 2 / (2 * w)
    assert_eq_plusc(integrate(expr, t), expected, t)

    # ln integral
    assert_eq_strict((x * log(x) - x).diff(x), log(x))
    assert_integral(log(x), x * log(x) - x)

    integrand = log(x + 6) / x**2
    expected = log(abs(x)) / 6 - log(x + 6) / x - log(abs(x + 6)) / 6  # TODO: why does one of them not have abs?
    assert_integral(integrand, expected)

    # without the string search fix on InverseTrigUSub, this returns a wrong answer.
    integrand = sqrt(x**2 + 11)
    ans = integrate(integrand, x)
    assert ans is None

    # we used to get this wrong from a byparts error
    integrand = cos(x) * cos(2 * x)
    expected_ans = sin(3 * x) / 6 + sin(x) / 2
    assert_integral(integrand, expected_ans)


def test_integrate_with_completing_the_square():
    expr = 1 / sqrt(-(x**2) + 10 * x + 11)
    expected = asin((x - 5) / 6)
    assert_integral(expr, expected)


def integrate_polar(r: Expr, theta: Symbol, a=0, b=2 * pi) -> Expr:
    return integrate(r**2, (theta, a, b)) / 2


def test_area_between():
    theta = symbols("theta")
    r1 = 3 * sin(theta)
    r2 = 1 + sin(theta)
    bounds = (pi / 6, 5 * pi / 6)
    ans = integrate_polar(r1, theta, *bounds) - integrate_polar(r2, theta, *bounds)
    assert_eq_value(ans, pi)


def test_multiterm_pts():
    expr = sin(x) * cos(2 * x) * sin(2 * x)
    assert_integral(expr, 2 * sin(x) ** 3 / 3 - 4 * sin(x) ** 5 / 5)
