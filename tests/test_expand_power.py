from simpy.combinatorics import generate_permutations
from simpy.debug.test_utils import assert_eq_strict, unhashable_set_eq
from simpy.expr import Prod, Sum, symbols


def test_expand_power():
    x = symbols("x")
    power = (x**3 + 1) ** 6
    prod = Prod([Sum([x**3, 1])] * 6)

    assert_eq_strict(power.expand(), prod.expand())


def test_multinomial_powers():

    # Example usage
    i = 3  # Number of terms
    n = 3  # Sum of terms
    result = generate_permutations(i, n)
    expected_result = [
        [0, 0, 3],
        [0, 1, 2],
        [0, 2, 1],
        [0, 3, 0],
        [1, 0, 2],
        [1, 1, 1],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
        [3, 0, 0],
    ]

    # lmao this is a bit sad but it works
    a = [repr(el) for el in result]
    b = [repr(el) for el in expected_result]
    unhashable_set_eq(a, b)
