import simpy as sp
from simpy.equation import Equation, solve


def test_simple_linear():
    x = sp.symbols("x")
    equation = Equation(-2 * x + 4, 0)
    solution = solve(equation, x)
    assert solution == 2
