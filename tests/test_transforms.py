"""Unsure if this file is needed // maybe combine with test_integrals

Ex doesn't rlly make sense to put test_complete_the_square not next to test_integrate_with_complete_the_square

Don't want to disconnect code structure from meaning too much altho it's easy to get caught up in code structure :)
"""

import numpy as np

from simpy.debug.test_utils import assert_eq_strict, x
from simpy.expr import Rat
from simpy.transforms import CompleteTheSquare, Node, PolynomialDivision, PullConstant, to_const_polynomial


def test_pullconstant():
    expr = 2 * x**3
    test_node = Node(expr, x)
    transform = PullConstant()
    assert transform.check(test_node)
    transform.forward(test_node)
    assert_eq_strict(test_node.child.expr, x**3)


def test_to_polynomial():
    expr = 6 * x + x**2
    assert np.array_equal(to_const_polynomial(expr, x), np.array([Rat(0), Rat(6), Rat(1)]))


def test_polynomial_division():
    expr = x**4 * (1 + x**2) ** -1

    test_node = Node(expr, x)
    tr = PolynomialDivision()
    assert tr.check(test_node)

    tr.forward(test_node)
    assert_eq_strict(test_node.children[0].expr, x**2 - 1 + 1 / (1 + x**2))

    # 2nd test
    expr = x**3 / (1 - x**2)

    test_node = Node(expr, x)
    tr = PolynomialDivision()
    assert tr.check(test_node)

    tr.forward(test_node)
    ans = test_node.children[0].expr


def test_complete_the_square():
    quadratic = -(x**2) + 10 * x + 11
    test_node = Node(quadratic, x)
    tr = CompleteTheSquare()
    tr.forward(test_node)
    ans = test_node.child.expr
    expected = 36 * (-((x - 5) ** 2) / 36 + 1)
    assert_eq_strict(ans, expected)
