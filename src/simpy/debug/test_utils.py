from typing import Optional, Tuple, Union

from simpy.debug.utils import debug_repr
from simpy.expr import Expr, Float, Symbol, TrigFunctionNotInverse, cast, log, symbols
from simpy.integration import integrate
from simpy.simplify import expand_logs, trig_simplify

x, y = symbols("x y")


@cast
def assert_integral(integrand: Expr, expected: Expr, var: Optional[Symbol] = None, **kwargs):
    assert_eq_plusc(integrate(integrand, var, **kwargs), expected)
    # assert_eq_value(diff(expected, var), integrand) # this might be nice; idk. catch some potential diff errors.


@cast
def assert_definite_integral(integrand: Expr, bounds: tuple, expected: Expr):
    assert_eq_strict(integrate(integrand, bounds), expected)


@cast
def assert_eq_plusc(a: Expr, b: Union[Expr, Tuple[Expr, ...]], *vars):
    """Assert a and b are equal up to a constant, relative to vars.
    If no vars are given, then
    """
    if isinstance(b, tuple):
        assert any(_assert_eq_plusc(a, b_, *vars)[0] for b_ in b)
        return
    success, diff = _assert_eq_plusc(a, b, *vars)
    assert success, f"diff = {diff}"


def _assert_eq_plusc(a, b, *vars) -> Tuple[bool, Expr]:
    diff = a - b
    if len(diff.symbols()) == 0 or vars and all(var not in diff.symbols() for var in vars):
        return True, None
    diff = simplify_to_same_standard(diff)
    if not vars:
        return len(diff.symbols()) == 0, diff
    else:
        return all(var not in diff.symbols() for var in vars), diff


def simplify_to_same_standard(expr: Expr) -> Expr:
    if expr.expandable():
        expr2 = expr.expand()
    else:
        expr2 = expr

    # if trig simplify has hit then it's always good :thumbsup:
    if expr2.has(TrigFunctionNotInverse):
        expr2, is_trig_hit = trig_simplify(expr2)
    if expr2.has(log):
        expr2 = expand_logs(expr2)

    return expr2


@cast
def assert_eq_value(a: Expr, b: Expr):
    """Tests that the values of a & b are the same in spirit, regardless of how they are represented with the Expr data structures."""
    if a == b:
        return
    diff = simplify_to_same_standard(a - b)
    assert diff == 0, f"a != b, {simplify_to_same_standard(a)} != {simplify_to_same_standard(b)}"


@cast
def assert_eq_strict(a: Expr, b: Expr):
    """Tests that the structure of a and b exprs are the same, not just their values or reprs.
    // This does the same thing as assert debug_repr(a) == debug_repr(b) I believe.
    """
    assert a == b, f"STRICT: {a} != {b}, \n\tDebug repr: {debug_repr(a)} != {debug_repr(b)}"


def unhashable_set_eq(a: list, b: list) -> bool:
    """Does equivalent of set(a) == set(b) but for lists containing unhashable types"""
    for el in a:
        if el not in b:
            return False
    for el in b:
        if el not in a:
            return False
    return True


def eq_float(e1: Expr, e2: Expr, atol=1e-6):
    if type(e1) != type(e2):
        return False
    if isinstance(e1, Float) and isinstance(e2, Float):
        return abs(e1.value - e2.value) < atol

    if not len(e1.children()) == len(e2.children()):
        return False
    return all(eq_float(c1, c2) for c1, c2 in zip(e1.children(), e2.children()))
