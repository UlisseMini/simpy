from fractions import Fraction

import pytest

from simpy.debug.test_utils import assert_eq_strict, unhashable_set_eq, x, y
from simpy.debug.utils import debug_repr
from simpy.expr import *
from simpy.integration import *
from simpy.regex import count


def test_equality():
    assert x == x
    assert x == Symbol("x")  # seperately created symbols with the same name should be the same
    assert not x == y
    assert x != y
    assert not x == 2 * x
    assert (x + 2) == (x + 2)  # seperately created sums should be the same
    assert (x + 2) == (2 + x)
    assert x + 2 == Symbol("x") + 2
    assert x * y == y * x

    # consts should handle equality, inequality, greater than, less than
    assert Rat(2) == 2
    assert Rat(2) != 3
    assert 2 == Rat(2)
    assert 2 < Rat(3)
    assert 2 <= Rat(2)
    assert Rat(2) == Rat(2)

    # trig functions aren't dataclasses so this is more liable to not working
    assert cos(x * 2) == cos(x * 2)


def test_floats():
    assert_eq_strict(Float(3.2), Rat(16, 5).evalf())
    assert_eq_strict(Sum([Float(2.32), Float(0.31)]), Float(2.63))
    assert_eq_strict(Prod([Float(2.32), Float(0.31)]), Float(0.7192))


def test_defaults():
    assert Sum([]) == 0
    assert Prod([]) == 1

    assert_eq_strict(x * 0, 0)
    assert_eq_strict(x * 2, 2 * x)
    assert_eq_strict(x**2, x * x)
    assert_eq_strict(x * 2 - 2 * x, 0)
    assert_eq_strict(((x + 1) ** 2 - (x + 1) * (x + 1)), 0)


def test_sum_combines_like_terms():
    assert_eq_strict(x + x, 2 * x)
    assert_eq_strict(x + x + x, 3 * x)
    assert_eq_strict(3 * (x + 2) + 2 * (x + 2), 5 * (x + 2))  # like terms that is a sum
    assert_eq_strict(2 * x * y + 3 * x * y, 5 * x * y)  # like terms with multiple factors
    assert_eq_strict(0.2 * x * y + 0.8 * x * y, x * y)  # like terms with multiple factors

    # not sure if we want this to be the case but wtv, make sure behavior is well-defined anyways
    # this check is useful it caught a bug when i mutated things for speed
    assert_eq_strict(2 * x + 0.2 * x, (Rat(2) + Float(0.2)) * x)

    # when it sums to 1
    assert_eq_strict(3 * x - 2 * x, x)


def test_prod_combines_like_terms_prod():
    assert_eq_strict(x**2, x * x)
    assert_eq_strict(x**3, x * x * x)

    # More complicated example
    # this wasn't working when the denominator "power" wasn't flattening in simplification.
    expr = (2 * sin(x) * cos(x) ** 2) / (sin(x) * cos(x) ** 2)
    assert expr == 2, f"expected 2, got {expr} // debug repr: {debug_repr(expr)}"


def test_basic_power_simplification():
    assert_eq_strict(x**0, 1)
    assert_eq_strict(x**1, x)
    assert_eq_strict(Rat(2) ** 2, 4)
    assert_eq_strict(sqrt(4), 2)
    assert_eq_strict(sqrt(x) ** 2, x)
    assert_eq_strict(2 / sqrt(2), sqrt(2))


def test_expandable():
    assert not (x / (x + 2)).expandable()
    assert ((x + 2) / x).expandable()
    assert log(x * (x + 2)).expandable()
    assert (log(x * (x + 2)) * 3).expandable()

    # This one requires seeing a sum in the denominator
    assert (y / (x * (x + 6))).expandable()


def test_expand_prod():
    # make sure an expandable denominator gets expanded
    assert_eq_strict((1 / (x * (x + 6))).expand(), 1 / (x**2 + x * 6))  # this can be converted to a power
    assert_eq_strict((y / (x * (x + 6))).expand(), y / (x**2 + x * 6))
    # make sure that a numberator with a single sum gets expanded
    assert_eq_strict(((2 + x) / sin(x)).expand(), (2 / sin(x) + x / sin(x)))


def test_expand_neg_power():
    expr = (cos(2 * x) + 1) ** 2 - 8 / (3 * (cos(2 * x) + 1) ** 3) + 2 / (cos(2 * x) + 1) ** -4
    assert expr.expandable()


def test_flatten():
    assert_eq_strict(x + (2 + y), x + 2 + y)

    # Test nested flatten
    expr = x**5 + ((3 + x) + 2 * y)
    expected_terms = [x**5, 3, x, 2 * y]
    assert unhashable_set_eq(expr.terms, expected_terms)


def test_regex():
    assert count(Rat(2), x) == 0
    assert count(tan(x + 1) ** 2 - 2 * x, x) == 2


def test_nesting():
    assert nesting(x**2, x) == 2
    assert nesting(x * y**2, x) == 2
    assert nesting(x * (1 / y**2 * 3), x) == 2


def test_repr():
    # New repr standards!!!
    expr = 1 - x**2
    assert expr.__repr__() == "-x^2 + 1"
    assert repr(x - 2) == "x - 2"
    assert (2 * x).__repr__() == "2*x"
    assert (2 * (2 + x)).__repr__() == "2*(x + 2)"
    assert (2 / (2 + x)).__repr__() == "2/(x + 2)"
    assert repr(2 * (2 + x) ** (-2)) == repr(2 / (2 + x) ** 2) == "2/(x + 2)^2"
    assert repr(Rat(-1) ** Fraction(1, 4)) == "(-1)^(1/4)"
    assert sqrt(3).__repr__() == "sqrt(3)"
    assert repr(1 / sqrt(1 - x**2)) == "1/sqrt(-x^2 + 1)"
    assert repr(sqrt(1 / x)) == "1/sqrt(x)"
    assert repr(x ** -Fraction(1, 2)) == "1/sqrt(x)"

    # make sure denominator is bracketed
    assert repr(sin(x) / (2 * x)) == "sin(x)/(2*x)"
    # make sure products with negative consts and dividing by consts are treated better
    assert repr(x / 2) == "x/2"
    assert repr(x * Fraction(1, 2)) == "x/2"
    assert repr(3 - 2 * x) == "-2*x + 3"
    # make sure polynomials show up in the correct order
    poly = 3 * x**4 / 2 + x**5 + 2 * x + 3 - x**2
    assert repr(poly) == "x^5 + 3*x^4/2 - x^2 + 2*x + 3"

    # when there are multiple symboless terms, make sure their order is logical
    assert repr(2 + log(2)) == "ln(2) + 2"
    assert repr(log(4) + Rat(9, 16) - log(2)) == "-ln(2) + ln(4) + 9/16"
    assert repr(Rat(2, 3) ** Rat(2, 3) * -1) == "-(2/3)^(2/3)"
    # assert repr(Float(2.2) * Rat(3) * x) == "3*2.2*x"
    # make sure consts show up before pi
    assert repr(pi * 2) == "2*pi"
    assert repr(pi * -2) == "-2*pi"


def test_neg_power():
    expr = Rat(-1) ** Fraction(5, 2)  # this is i. ig it should just stay this way & not simplify.
    assert debug_repr(expr) == "Power(-1, 5/2)"


@pytest.mark.xfail
def test_circular_repr():
    expr = Rat(-1) ** Fraction(5, 2) * -2
    repr(expr)
    expr = Prod([-Fraction(4, 5), Rat(-1) ** Fraction(5, 2)])
    repr(expr)
    expr = Prod([-Fraction(4, 5), Rat(-1) ** Fraction(5, 2) * x ** Fraction(5, 2)])
    repr(expr)


def test_factor():
    # Simple example
    x = symbols("x")
    expr = 6 * x + x**2
    expected = x * (x + 6)
    assert expr.factor() == expected

    # More complex example
    c3, r3, c4, r4, a, b = symbols("c_3 r_3 c_4 r_4 a b")

    em_hw_expr = (
        c3 * r3 / (sqrt(a) * sqrt(b))
        - c3**2 * r3**2 / (c4 * r4 * sqrt(b) * a ** Fraction(3 / 2))
        - c4 * r4 / (sqrt(a) * b ** Fraction(3 / 2))
    )
    factored = em_hw_expr.factor()
    expected_factored = 1 / (sqrt(a) * sqrt(b)) * (c3 * r3 - c3**2 * r3**2 / (c4 * r4 * a) - c4 * r4 / b)
    assert_eq_strict(factored, expected_factored)


def test_some_constructor_simplification():
    r1, r2, w, v = symbols("r_1 r_2 \\omega v_0")

    # I believe r1 and r2 are resistances so they are strictly positive
    r1._strictly_positive = True
    r2._strictly_positive = True

    # I think r2 > r1 always so this is also strictly positive?
    # TODO check the problem again
    intermediate = r2 / r1 - 1
    intermediate._strictly_positive = True

    c = 1 / (r2 * w) * sqrt(intermediate)
    l = -r1 / (r2 * w) * sqrt(intermediate)
    r_s = r1
    x_s = w * l
    # z_s = r_s + x_s * 1j
    r_l = r2 / (1 + w**2 * c**2 * r2**2)
    x_l = w * c * r2 / (1 + w**2 * c**2 * r2**2)
    # z_l = r_l + x_l * 1j # simpy doesn't support complex numbers i think.

    z_eq = sqrt((r_s + r_l) ** 2 + (x_s + x_l) ** 2)
    i_0 = v / z_eq
    pload = Fraction(1, 2) * r_l * i_0**2

    assert_eq_strict(pload, v**2 / (8 * r1))  # the power dissipated through the load when z_l = conjugate(z_s)


def test_factor_const():
    expr = 2 - 2 * x
    factored = expr.factor()
    expected = 2 * (1 - x)
    assert_eq_strict(expected, factored)


def test_strict_const_power_simplification():
    """TBH I feel like I made this overly complicated LOL but whatever we live with this."""
    half = Rat(Fraction(1, 2))
    neg_half = Rat(Fraction(-1, 2))

    # All permutations of (3/4)^(1/2)
    expr = Power(Rat(Fraction(3, 4)), half)
    expected = Prod([Power(3, half), half])
    assert_eq_strict(expr, expected)

    expr = Power(Rat(Fraction(4, 3)), half)
    expected = Prod([Rat(2), Power(3, neg_half)])
    assert_eq_strict(expr, expected)

    expr = Power(Rat(Fraction(3, 4)), neg_half)
    expected = Prod([Power(3, neg_half), Rat(2)])
    assert_eq_strict(expr, expected)

    expr = Power(Rat(Fraction(4, 3)), neg_half)
    expected = Prod([half, Power(3, half)])
    assert_eq_strict(expr, expected)

    # If the base is negative, it cannot square root.
    expr = Power(Rat(Fraction(-4, 3)), half)
    expected = Power(Rat(Fraction(-4, 3)), half)
    assert_eq_strict(expr, expected)

    # but it can cube root
    expr = Power(Rat(Fraction(-8, 3)), Fraction(1, 3))
    expected = Prod([Rat(-2), Power(3, Fraction(-1, 3))])
    assert_eq_strict(expr, expected)
    expr = Power(Rat(Fraction(-8, 3)), Fraction(-1, 3))
    expected = Prod([neg_half, Power(3, Fraction(1, 3))])
    assert_eq_strict(expr, expected)

    # All permutations of (36)^(1/2)
    expr = Power(Rat(Fraction(1, 36)), neg_half)
    expected = Rat(6)
    assert_eq_strict(expr, expected)

    expr = Power(Rat(36), neg_half)
    expected = Rat(Fraction(1, 6))
    assert_eq_strict(expr, expected)

    expr = Power(Rat(36), half)
    expected = Rat(6)
    assert_eq_strict(expr, expected)

    expr = Power(Rat(Fraction(1, 36)), half)
    expected = Rat(Fraction(1, 6))
    assert_eq_strict(expr, expected)

    # Nothing should happen when it's unsimplifiable
    # (this might change in the future bc maybe i want standards for frac^(neg x) vs reciprocal_frac^abs(neg x))
    expr = Power(Rat(2), neg_half)
    assert debug_repr(expr) == "Power(2, -1/2)"

    # This is the expression that caused me problems!!
    # When I was doing it wrong (raising numerator/denominator of Fraction to exponent seperately even when it was 1),
    # Power(Power(2, 1/2), -1) gets simplified to Prod(Power(Power(2, 1/2), -1), 1)
    # Which is the most bullshit ever.
    expr = Power(Power(Rat(2), half), -1)
    expected = Power(Rat(2), neg_half)
    assert_eq_strict(expr, expected)


def test_fractional_power_beauty_standards():
    """Make sure Power(1/a, neg x) simplifies to Power(a, x) --- condition [A]
    Make sure Power(a/b), neg x simplifies to Power(b/a, x) [B]

    More controversially:
    Power(1/a, x) should simplify to Power(a, neg x) [C]

    This keeps the repr standard consistent with Power(a, neg x) which is equal.
    """
    f53 = Rat(Fraction(5, 3))
    f35 = Rat(Fraction(3, 5))
    half = Rat(Fraction(1, 2))
    neg_half = Rat(Fraction(-1, 2))

    assert_eq_strict(f53**half, f35**neg_half)  # [B]
    assert_eq_strict(f35**half, f53**neg_half)  # [B]

    assert repr(Power(half, neg_half)) == "sqrt(2)"  # [A]
    assert_eq_strict(Power(half, neg_half), Power(Rat(2), half))  # [A]
    f17 = Rat(Fraction(1, 7))
    neg_f17 = Rat(Fraction(-1, 7))
    assert repr(Power(half, neg_f17)) == "2^(1/7)"  # [A]
    assert repr(Power(f17, neg_f17)) == "7^(1/7)"  # [A]

    # More controversially, [C]:
    # Both of the following should be represented as 1/3^(1/7)
    e1 = Power(Rat(Fraction(1, 3)), f17)
    e2 = Power(Rat(3), neg_f17)
    assert repr(e1) == repr(e2) == "1/3^(1/7)"
    assert_eq_strict(e1, e2)


def test_singlefuncs_auto_simplify_special_values():
    assert_eq_strict(log(1), 0)
    assert_eq_strict(log(e**x), x)
    assert_eq_strict(sin(3 * pi), 0)
    assert_eq_strict(cos(5 * pi), -1)
    assert_eq_strict(csc(pi / 2), 1)
    assert_eq_strict(cot(pi / 4), 1)
    assert_eq_strict(cos(-x), cos(x))
    assert_eq_strict(sin(-x), -sin(x))
    assert_eq_strict(tan(-x), -tan(x))
    assert_eq_strict(sin(acos(x + y)), sqrt(1 - (x + y) ** 2))
    assert_eq_strict(csc(acos(x + y)), 1 / sqrt(1 - (x + y) ** 2))
    assert_eq_strict(sec(acos(x + y)), 1 / (x + y))
    assert_eq_strict(atan(cot(3 * x + 2)), 1 / (3 * x + 2))


def test_trigfuncs_auto_simplify_plus_2pis():
    assert_eq_strict(cos(x + 2 * pi), cos(x))
    assert_eq_strict(sin(x + e**y + 4 * pi), sin(x + e**y))

    # Period of tan is pi
    assert_eq_strict(tan(x + pi), tan(x))

    # Inverses aren't periodic
    assert_eq_strict(asin(x + 2 * pi), asin(x + 2 * pi))


def test_trigfuncs_auto_simplify_more_complex_negs():
    assert_eq_strict(cos(-x - 2), cos(x + 2))


@pytest.mark.parametrize(
    ["cls", "func"],
    [
        [sin, math.sin],
        [cos, math.cos],
        [tan, math.tan],
    ],
)
def test_trigfunctions_special_values_are_correct(cls: Type[TrigFunction], func):
    import numpy as np

    for k in TrigFunction._SPECIAL_KEYS:
        num = Fraction(k)
        v1 = cls(num * pi).evalf().value
        v2 = func(num * math.pi)
        try:
            np.testing.assert_almost_equal(v1, v2)
        except AssertionError:
            if cls == tan and (v1 == inf or v1 == -inf):
                assert v2 > 1e10 or v2 < -1e-10
            else:
                raise AssertionError


def test_is_subtraction():
    assert x.is_subtraction is False
    assert (-x).is_subtraction is True
    assert Rat(0).is_subtraction is False
    assert Rat(-2).is_subtraction is True
    assert (e**x).is_subtraction is False
    assert (-(e**-x)).is_subtraction is True
    assert (Rat(-3) ** 3).is_subtraction is True
    assert (Rat(Fraction(-3, 2)) ** Fraction(1, 3)).is_subtraction is True
    assert (Rat(Fraction(-3, 2)) ** Fraction(-2, 3)).is_subtraction is True
    assert (Rat(Fraction(-3, 2)) ** Fraction(1, 2)).is_subtraction is False


def test_power_abs():
    assert sqrt(x**2) == abs(x)
    assert sqrt(x) ** 2 == x
    assert (x**6) ** Rat(1, 6) == abs(x)
    assert (x**3) ** Rat(1, 3) == x


def test_trigfunctions_plusminuspi():
    assert sin(x + pi) == -sin(x)
    assert cos(x + pi) == -cos(x)
    assert sin(x - pi) == -sin(x)
    assert cos(x - pi) == -cos(x)
