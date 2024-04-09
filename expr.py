# TODO: tasks for the future:
# - method to convert from our expression to sympy for testing

import itertools
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from fractions import Fraction
from functools import reduce
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union


def _cast(x):
    if type(x) == int or isinstance(x, Fraction):
        return Const(x)
    if type(x) == float and int(x) == x:  # silly patch
        return Const(int(x))
    elif isinstance(x, Expr):
        return x
    elif isinstance(x, dict):
        return {k: _cast(v) for k, v in x.items()}
    elif isinstance(x, tuple):
        return tuple(_cast(v) for v in x)
    else:
        raise NotImplementedError(f"Cannot cast {x} to Expr")


def cast(func):
    def wrapper(*args) -> "Expr":
        return func(*[_cast(a) for a in args])

    return wrapper


class Expr(ABC):
    def __post_init__(self):
        # if any field is an Expr, cast it
        # note: does not cast List[Expr]
        for field in fields(self):
            if field.type is Expr:
                setattr(self, field.name, _cast(getattr(self, field.name)))

    # should be overwritten in subclasses
    def simplify(self):
        return self

    @cast
    def __add__(self, other):
        return Sum([self, other])

    @cast
    def __radd__(self, other):
        return Sum([other, self])

    @cast
    def __sub__(self, other):
        return self + (-1 * other)

    @cast
    def __rsub__(self, other):
        return other + (-1 * self)

    @cast
    def __mul__(self, other):
        return Prod([self, other])

    @cast
    def __rmul__(self, other):
        return Prod([other, self])

    @cast
    def __pow__(self, other):
        return Power(self, other)

    @cast
    def __rpow__(self, other):
        return Power(other, self)

    @cast
    def __div__(self, other):
        return Prod([self, Power(other, -1)])

    @cast
    def __truediv__(self, other):
        return Prod([self, Power(other, -1)])

    @cast
    def __rdiv__(self, other):
        return Prod([other, Power(self, -1)])

    @cast
    def __rtruediv__(self, other):
        return Prod([other, Power(self, -1)])

    def __neg__(self):
        return -1 * self

    @cast
    def __eq__(self, other):
        return self.__repr__() == other.__repr__()

    @cast
    def __ne__(self, other):
        return not (self == other)

    # should be overloaded if necessary
    def expandable(self) -> bool:
        return False

    # overload if necessary
    def expand(self) -> "Expr":
        raise NotImplementedError(f"Cannot expand {self}")

    @cast
    @abstractmethod
    def evalf(self, subs: Dict[str, "Const"]):
        raise NotImplementedError(f"Cannot evaluate {self}")

    @abstractmethod
    def children(self) -> List["Expr"]:
        raise NotImplementedError(f"Cannot get children of {self.__class__.__name__}")

    def contains(self: "Expr", var: "Symbol"):
        is_var = isinstance(self, Symbol) and self.name == var.name
        return is_var or any(e.contains(var) for e in self.children())

    # should be overloaded
    def simplifable(self) -> bool:
        return False

    # @abstractmethod
    def diff(self, var: "Symbol"):
        raise NotImplementedError(
            f"Cannot get the derivative of {self.__class__.__name__}"
        )

    def symbols(self) -> List["Symbol"]:
        # I hate this syntax
        str_set = set([symbol.name for e in self.children() for symbol in e.symbols()])
        return [Symbol(name=s) for s in str_set]


@dataclass
class Associative:
    terms: List[Expr]

    def _flatten(self) -> "Associative":
        new_terms = []
        for t in self.terms:
            new_terms += t._flatten().terms if isinstance(t, self.__class__) else [t]
        return self.__class__(new_terms)

    def children(self) -> List["Expr"]:
        return self.terms

    def _sort(self) -> "Associative":
        def _key(term: Expr) -> str:
            n = nesting(term)
            power = deconstruct_power(term)[1]
            return f"{n} {power} {term.__repr__()}"
            # the idea is you sort first by nesting, then by power, then by the term alphabetical

        return self.__class__(sorted(self.terms, key=_key))

    @abstractmethod
    def simplify(self) -> "Associative":
        return self.__class__([t.simplify() for t in self.terms])._flatten()
        # sort at the end


@dataclass
class Const(Expr):
    value: Fraction

    def __post_init__(self):
        assert (
            isinstance(self.value, Fraction) or type(self.value) == int
        ), f"got value={self.value} not allowed Const"
        self.value = Fraction(self.value)

    def __repr__(self):
        return str(self.value) if self.value.denominator == 1 else f"({self.value})"

    @cast
    def __eq__(self, other):
        return (isinstance(other, Const) and self.value == other.value) or (
            isinstance(other, Union[int, Fraction]) and self.value == other
        )

    @cast
    def __ge__(self, other):
        if not isinstance(other, Const):
            return NotImplemented
        return self.value > other.value

    @cast
    def __lt__(self, other):
        if not isinstance(other, Const):
            return NotImplemented
        return self.value < other.value

    @cast
    def evalf(self, subs: Dict[str, "Const"]):
        return self

    def diff(self, var):
        return Const(0)

    def children(self) -> List["Expr"]:
        return []


@dataclass
class Symbol(Expr):
    name: str

    def __repr__(self):
        return self.name

    @cast
    def evalf(self, subs: Dict[str, "Const"]):
        return subs.get(self.name, self)

    def diff(self, var):
        return Const(1) if self == var else Const(0)

    def __eq__(self, other):
        return isinstance(other, Symbol) and self.name == other.name

    def children(self) -> List["Expr"]:
        return []

    def symbols(self) -> List["Expr"]:
        return [self]


@dataclass
class Sum(Associative, Expr):
    def simplify(self):
        # TODO: this currently would not combine terms like (2+x) and (x+2)

        # simplify subexprs and flatten sub-sums
        s = super().simplify()

        # accumulate all constants
        const = sum(t.value for t in s.terms if isinstance(t, Const))

        # return immediately if there are no non constant items
        non_constant_terms = [t for t in s.terms if not isinstance(t, Const)]
        if len(non_constant_terms) == 0:
            return Const(const)

        # otherwise, bring the constant to the front (if != 1)
        s = Sum(([] if const == 0 else [Const(const)]) + non_constant_terms)

        # accumulate all like terms
        new_terms = []
        for i, term in enumerate(s.terms):
            if term is None:
                continue

            new_coeff, non_const_factors1 = _deconstruct_prod(term)

            # check if any later terms are the same
            for j in range(i + 1, len(s.terms)):
                if s.terms[j] is None:
                    continue

                term2 = s.terms[j]
                coeff2, non_const_factors2 = _deconstruct_prod(term2)

                if non_const_factors1 == non_const_factors2:
                    new_coeff += coeff2
                    s.terms[j] = None

            new_terms.append(Prod([new_coeff] + non_const_factors1).simplify())

        # if Const(1) in new_terms and Prod([-1, Power(TrigFunction(anysymbol, "sin"), 2)]) in new_terms:
        # get rid of 1-term sums
        if len(new_terms) == 1:
            return new_terms[0]

        new_sum = Sum(new_terms)

        if contains_cls(new_sum, TrigFunction):
            # I WANT TO DO IT so that it's more robust.
            # - what if the matched query is not a symbol but an expression?
            # - ~~do something to check for sin^2x + cos^2x = 1 (and allow for it if sum has >2 terms)~~
            # - ordering
            # - what if there is a constant (or variable) common factor? (i think for this i'll have to implement a .factor method)

            pythagorean_trig_identities: Dict[str, Callable[[Expr], Expr]] = {
                r"1 \+ tan\((\w+)\)\^2": lambda x: Sec(x) ** 2,
                r"1 \+ cot\((\w+)\)\^2": lambda x: Csc(x) ** 2,
                r"1 - sin\((\w+)\)\^2": lambda x: Cos(x) ** 2,
                r"1 - cos\((\w+)\)\^2": lambda x: Sin(x) ** 2,
                r"1 - tan\((\w+)\)\^2": lambda x: Const(1) / (Tan(x) ** 2),
                r"1 - cot\((\w+)\)\^2": lambda x: Const(1) / (Cot(x) ** 2),
            }

            for pattern, replacement_callable in pythagorean_trig_identities.items():
                match = re.search(pattern, new_sum.__repr__())
                result = match.group(1) if match else None

                if result and len(new_sum.terms) == 2:
                    other = replacement_callable(Symbol(result)).simplify()
                    return other

            # fuckit just gonna let the insides be anything and not check for paranthesis balance
            # because im asserting beginning and end of string conditions.
            other_table = [
                (r"^sin\((.+)\)\^2$", r"^cos\((.+)\)\^2$", Const(1)),
                (r"^sec\((.+)\)\^2$", r"^-tan\((.+)\)\^2$", Const(1)),
            ]
            for pattern1, pattern2, value in other_table:
                match1 = []
                match2 = []
                for t in new_sum.terms:
                    m1 = re.search(pattern1, t.__repr__())
                    m2 = re.search(pattern2, t.__repr__())
                    if m1:
                        match1.append(m1)
                    if m2:
                        match2.append(m2)

                if len(match1) == 0 or len(match2) == 0:
                    continue

                r1 = [m.group(1) for m in match1]
                r2 = [m.group(1) for m in match2]
                for m in r1:
                    for n in r2:
                        if m == n:
                            new_terms = [value] + [
                                t
                                for t in new_sum.terms
                                if t.__repr__() != f"sin({m})^2"
                                and t.__repr__() != f"cos({m})^2"
                            ]
                            return Sum(new_terms).simplify()

        return new_sum._sort()

    @cast
    def evalf(self, subs: Dict[str, "Const"]):
        return Sum([t.evalf(subs) for t in self.terms]).simplify()

    def diff(self, var):
        return Sum([diff(e, var) for e in self.terms])

    def __repr__(self):
        ongoing_str = "("
        for i, term in enumerate(self.terms):
            if i == 0:
                ongoing_str += f"{term}"
            elif isinstance(term, Prod) and term.is_subtraction:
                ongoing_str += f" - {Prod(term.terms[1:]).simplify()}"
            else:
                ongoing_str += f" + {term}"

        return ongoing_str + ")"


def _deconstruct_prod(expr: Expr) -> Tuple[Const, List[Expr]]:
    # 3*x^2*y -> (3, [x^2, y])
    # turns smtn into a constant and a list of other terms
    # assume expr is simplified
    if isinstance(expr, Prod):
        # simplifying the product puts the constants at the front
        non_const_factors = (
            expr.terms[1:] if isinstance(expr.terms[0], Const) else expr.terms
        )
        coeff = expr.terms[0] if isinstance(expr.terms[0], Const) else Const(1)
    else:
        non_const_factors = [expr]
        coeff = Const(1)
    return (coeff, non_const_factors)


def deconstruct_power(expr: Expr) -> Tuple[Expr, Const]:
    # x^3 -> (x, 3). x -> (x, 1). 3 -> (3, 1)
    if isinstance(expr, Power):
        return (expr.base, expr.exponent)
    else:
        return (expr, Const(1))


@dataclass
class Prod(Associative, Expr):
    def __repr__(self):
        # special case for subtraction:
        if self.is_subtraction:
            if len(self.terms) == 2:
                return "-" + repr(self.terms[1])
            else:
                return "-" + Prod(self.terms[1:]).__repr__()

        is_division = False
        denominator = []
        denominator_ = []
        for term in self.terms:
            if isinstance(term, Power):
                if isinstance(term.exponent, Const) and term.exponent.value < 0:
                    is_division = True
                    denominator.append(term)
                    denominator_.append(Power(term.base, -term.exponent))

        if is_division:
            numerator = [term for term in self.terms if term not in denominator]
            numerator_str = repr(Prod(numerator).simplify())
            denominator_str = repr(Prod(denominator_).simplify())
            return numerator_str + "/" + denominator_str

        return "(" + "*".join(map(repr, self.terms)) + ")"

    @property
    def numerator_denominator(self) -> Tuple["Expr", "Expr"]:
        denominator = [1]
        numerator = [1]
        for term in self.terms:
            b, x = deconstruct_power(term)
            if isinstance(x, Const) and x.value < 0:
                denominator.append(Power(b, -x))
            else:
                numerator.append(term)
        return [Prod(numerator).simplify(), Prod(denominator).simplify()]

    @property
    def is_subtraction(self):
        return self.terms[0] == Const(-1)

    def simplify(self):
        # simplify subexprs and flatten sub-products
        new = super().simplify()

        # accumulate all like terms
        terms = []
        for i, term in enumerate(new.terms):
            if term is None:
                continue

            base, expo = deconstruct_power(term)

            # other terms with same base
            for j in range(i + 1, len(new.terms)):
                if new.terms[j] is None:
                    continue
                other = new.terms[j]
                base2, expo2 = deconstruct_power(other)
                if base2 == base:  # TODO: real expr equality
                    expo += expo2
                    new.terms[j] = None

            terms.append(Power(base, expo).simplify())

        new.terms = terms

        # Check for zero
        if any(t == 0 for t in new.terms):
            return Const(0)

        # accumulate constants to the front
        const = reduce(
            lambda x, y: x * y, [t.value for t in new.terms if isinstance(t, Const)], 1
        )

        # return immediately if there are no non constant items
        non_constant_terms = [t for t in new.terms if not isinstance(t, Const)]
        if len(non_constant_terms) == 0:
            return Const(const)

        # otherwise, bring the constant to the front (if != 1)
        new.terms = ([] if const == 1 else [Const(const)]) + non_constant_terms

        return new.terms[0] if len(new.terms) == 1 else new._sort()

    @cast
    def expandable(self) -> bool:
        # a product is expandable if it contains any sums
        return any(isinstance(t, Sum) for t in self.terms) or any(
            t.expandable() for t in self.terms
        )

    def expand(self):
        # expand sub-expressions
        self = self._flatten()
        self = Prod([t.expand() if t.expandable() else t for t in self.terms])

        # expand sums that are left
        sums = [t for t in self.terms if isinstance(t, Sum)]
        other = [t for t in self.terms if not isinstance(t, Sum)]

        if not sums:
            return self

        # for every combination of terms in the sums, multiply them and add
        # (using itertools)
        expanded = []
        for terms in itertools.product(*[s.terms for s in sums]):
            expanded.append(Prod(other + list(terms)).simplify())

        return Sum(expanded).simplify()

    @cast
    def evalf(self, subs: Dict[str, "Const"]):
        return Prod([t.evalf(subs) for t in self.terms]).simplify()

    def diff(self, var):
        return Sum(
            [
                Prod([diff(e, var)] + [t for t in self.terms if t is not e])
                for e in self.terms
            ]
        )


@dataclass
class Power(Expr):
    base: Expr
    exponent: Expr

    def __repr__(self):
        # special case for sqrt
        if self.exponent == Const(Fraction(1, 2)):
            return _repr(self.base, "sqrt")
        if self.exponent == Const(Fraction(-1, 2)):
            return f"{_repr(self.base, 'sqrt')}^-1"

        return f"{self.base}^{self.exponent}"

    def simplify(self):
        x = self.exponent.simplify()
        b = self.base.simplify()
        if x == 0:
            return Const(1)
        elif x == 1:
            return b
        elif isinstance(b, Const) and isinstance(x, Const):
            return Const(b.value**x.value)
        elif isinstance(b, Power):
            return Power(b.base, x * b.exponent).simplify()
        elif isinstance(b, Prod):
            # when you construct this new power entity you have to simplify it.
            # because what if the term raised to this exponent can be simplified?
            # ex: if you have (ab)^n where a = c^m
            return Prod([Power(term, x).simplify() for term in b.terms])
        else:
            return Power(self.base.simplify(), x)

    def expandable(self) -> bool:
        return (
            isinstance(self.exponent, Const)
            and self.exponent.value.denominator == 1
            and self.exponent.value >= 1
            and isinstance(self.base, Sum)
        )

    def expand(self) -> Expr:
        assert self.expandable(), f"Cannot expand {self}"
        return Prod([self.base] * self.exponent.value.numerator).expand()

    @cast
    def evalf(self, subs: Dict[str, "Const"]):
        return Power(self.base.evalf(subs), self.exponent.evalf(subs)).simplify()

    def children(self) -> List["Expr"]:
        return [self.base, self.exponent]

    def diff(self, var) -> Expr:
        if self.exponent.contains(var):
            # return self * (self.exponent * Log(self.base)).diff(var) idk if this is right and im lazy rn
            raise NotImplementedError(
                "Power.diff not implemented for exponential functions."
            )
        return self.exponent * self.base ** (self.exponent - 1) * self.base.diff(var)


@dataclass
class SingleFunc(Expr):
    inner: Expr

    @property
    @abstractmethod
    def _label(self) -> str:
        raise NotImplementedError("Label not implemented")

    def children(self) -> List["Expr"]:
        return [self.inner]

    def simplify(self):
        inner = self.inner.simplify()
        return self.__class__(inner)

    def __repr__(self) -> str:
        return _repr(self.inner, self._label)


def _repr(inner: Expr, label: str) -> str:
    inner_repr = inner.__repr__()
    if inner_repr[0] == "(" and inner_repr[-1] == ")":
        return f"{label}{inner_repr}"
    return f"{label}({inner_repr})"


class Log(SingleFunc):
    inner: Expr

    @property
    def _label(self):
        return "ln"

    @cast
    def evalf(self, subs: Dict[str, "Const"]):
        inner = self.inner.evalf(subs)
        # TODO: Support floats in .evalf
        # return Const(math.log(inner.value)) if isinstance(inner, Const) else Log(inner)
        return Log(inner)

    def simplify(self):
        inner = self.inner.simplify()
        if inner == 1:
            return Const(0)

        return Log(inner)

    def diff(self, var):
        return self.inner.diff(var) / self.inner


@cast
def sqrt(x: Expr) -> Expr:
    return x ** Const(Fraction(1, 2))


double_trigfunction_simplification_dict: Dict[str, Callable[[Expr], Expr]] = {
    "sin acos": lambda x: sqrt(1 - x**2),
    "sin atan": lambda x: x / sqrt(1 + x**2),
    "cos asin": lambda x: sqrt(1 - x**2),  # same as sin acos
    "cos atan": lambda x: 1 / sqrt(1 + x**2),
    "tan asin": lambda x: x / sqrt(1 - x**2),
    "tan acos": lambda x: sqrt(1 - x**2) / x,
    # Arcsecant
    "sin asec": lambda x: sqrt(x**2 - 1) / x,  # Since sin(asec(x)) = sqrt(x^2 - 1) / x
    "tan asec": lambda x: sqrt(x**2 - 1),  # tan(asec(x)) = sqrt(x^2 - 1)
    # Arccosecant
    "cos acsc": lambda x: sqrt(1 - 1 / x**2),  # cos(acsc(x)) = sqrt(1 - 1/x^2)
    "tan acsc": lambda x: 1 / sqrt(x**2 - 1),  # tan(acsc(x)) = 1/sqrt(x^2 - 1)
    # Arccotangent
    "sin acot": lambda x: 1 / sqrt(1 + x**2),  # sin(acot(x)) = 1/sqrt(1 + x^2)
    "cos acot": lambda x: x / sqrt(1 + x**2),  # cos(acot(x)) = x/sqrt(1 + x^2)
}

reciprocal_chart: Dict[str, str] = {
    "sin": "csc",
    "cos": "sec",
    "tan": "cot",
    "csc": "sin",
    "sec": "cos",
    "cot": "tan",
}


class TrigFunction(SingleFunc):
    inner: Expr
    function: Literal["sin", "cos", "tan", "sec", "csc", "cot"]
    is_inverse: bool = False

    # have to have __init__ here bc if i use @dataclass on TrigFunction
    # repr no longer inherits from SingleFunc
    def __init__(self, inner, function, is_inverse=False):
        super().__init__(inner)
        self.function = function
        self.is_inverse = is_inverse

    @property
    def _label(self):
        return f"{'a' if self.is_inverse else ''}{self.function}"

    def simplify(self):
        inner = self.inner.simplify()

        # things like sin(cos(x)) cannot be more simplified.
        if isinstance(inner, TrigFunction) and inner.is_inverse != self.is_inverse:
            # asin(sin(x)) -> x
            if inner.function == self.function:
                return inner.inner

            if not self.is_inverse:
                if inner.function == reciprocal_chart[self.function]:
                    return (1 / inner.inner).simplify()

                if self.function in ["sin", "cos", "tan"]:
                    callable_ = double_trigfunction_simplification_dict[
                        f"{self.function} {inner._label}"
                    ]
                    return callable_(inner.inner).simplify()

                else:
                    callable_ = double_trigfunction_simplification_dict[
                        f"{reciprocal_chart[self.function]} {inner._label}"
                    ]
                    return (1 / callable_(inner.inner)).simplify()

            # not supporting stuff like asin(cos(x)) sorry.

        return self.__class__(inner)


class Sin(TrigFunction):
    def __init__(self, inner):
        super().__init__(inner, function="sin")


class Cos(TrigFunction):
    def __init__(self, inner):
        super().__init__(inner, function="cos")


class Tan(TrigFunction):
    def __init__(self, inner):
        super().__init__(inner, function="tan")


class Csc(TrigFunction):
    def __init__(self, inner):
        super().__init__(inner, function="csc")


class Sec(TrigFunction):
    def __init__(self, inner):
        super().__init__(inner, function="sec")


class Cot(TrigFunction):
    def __init__(self, inner):
        super().__init__(inner, function="cot")


class ArcSin(TrigFunction):
    def __init__(self, inner):
        super().__init__(inner, function="sin", is_inverse=True)


class ArcCos(TrigFunction):
    def __init__(self, inner):
        super().__init__(inner, function="cos", is_inverse=True)


class ArcTan(TrigFunction):
    def __init__(self, inner):
        super().__init__(inner, function="tan", is_inverse=True)


def symbols(symbols: str):
    symbols = [Symbol(name=s) for s in symbols.split(" ")]
    return symbols if len(symbols) > 1 else symbols[0]


@cast
def diff(expr: Expr, var: Symbol) -> Expr:
    if hasattr(expr, "diff"):
        return expr.diff(var)
    else:
        raise NotImplementedError(f"Differentiation of {expr} not implemented")


def nesting(expr: Expr, var: Optional[Symbol] = None) -> int:
    """
    Compute the nesting amount (complexity) of an expression
    If var is provided, only count the nesting of the subexpression containing var

    >>> nesting(x**2, x)
    2
    >>> nesting(x * y**2, x)
    2
    >>> nesting(x * (1 / y**2 * 3), x)
    2
    """

    if var is not None and not expr.contains(var):
        return 0

    # special case
    if isinstance(expr, Prod) and expr.terms[0] == Const(-1) and len(expr.terms) == 2:
        return nesting(expr.terms[1], var)

    if isinstance(expr, Symbol) and (var is None or expr.name == var.name):
        return 1
    elif len(expr.children()) == 0:
        return 0
    else:
        return 1 + max(nesting(sub_expr, var) for sub_expr in expr.children())


def contains_cls(expr: Expr, cls) -> bool:
    if isinstance(expr, cls) or issubclass(expr.__class__, cls):
        return True

    return any([contains_cls(e, cls) for e in expr.children()])


@cast
def count(expr: Expr, query: Expr) -> int:
    if isinstance(expr, query.__class__) and expr == query:
        return 1
    return sum(count(e, query) for e in expr.children())