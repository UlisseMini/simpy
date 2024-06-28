from simpy import *
from simpy.debug.test_utils import *
from simpy.simplify.product_to_sum import *


def test_sin_x_3():
    """Has been manually calculated <3"""
    assert product_to_sum_unit(sin(x) ** 3) == 3 * sin(x) / 4 - sin(3 * x) / 4


def test_cos_x_5():
    """Has been manually calculated <3"""
    assert product_to_sum_unit(cos(x) ** 5) == cos(5 * x) / 16 + 5 * cos(3 * x) / 16 + 5 * cos(x) / 8


def test_multiterm_prod():
    assert product_to_sum_unit(sin(x) * cos(x) * sin(2 * x)) == sin(x) ** 2 / 2 + sin(3 * x) * sin(x) / 2


def test_power_in_prod():
    assert product_to_sum_unit(sin(x) ** 3 * cos(x)) == sin(2 * x) / 4 - sin(4 * x) / 8
    assert product_to_sum_unit(sin(x) ** 3 * cos(x) ** 2) == sin(x) / 8 + sin(3 * x) / 16 - sin(5 * x) / 16
