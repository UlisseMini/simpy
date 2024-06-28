from simpy.debug.test_utils import *
from simpy.expr import *


def test_infinity_basic_ops():
    assert 0 < inf
    assert Rat(1) == 1
    assert inf**1 == inf
    assert inf == inf + 1
    assert sqrt(inf) == inf
    assert inf == 2 * inf
    assert 192330 < inf
    assert inf > -inf
    assert -inf < -2039
    assert -inf == -inf - 1
    assert -inf == -2 * inf


def test_nums():
    assert e > 2
    assert pi <= 4


def test_combine_float_rat():
    assert Rat(1, 2) + 0.5 == 1
    assert debug_repr(0.5 + Rat(1, 2), pedantic="always") == debug_repr(Float(1.0), pedantic="always")
    assert Rat(1, 3) * 0.2 == 0.2 / 3


def test_combining_in_sum():
    assert eq_float(0.2 * 3 * x, 0.6 * x)
    assert eq_float(0.2 * 3 * x + 0.2 * 5 * x, 1.6 * x)


def test_accumulate():
    assert accumulate(Rat(2), Rat(3), Rat(4)) == Rat(9)
    assert accumulate(Float(2.2), Rat(3), Rat(4)) == Float(9.2)
