"""RULES OF EXPRs:

1. Exprs shall NOT be mutated in place after __post_init__.
For example, if I put a Const into a numpy array, I don't want to have to copy it. i can trust that its value stays the same forever.

Note on equality: if you call (expr1 == expr2), it returns true/false based on **the structure of the expr is the same.** rather than based on their values
being equal. if you wanna check equality of values, try simplifying, expanding, etc.

This currently operates on fractions only. I personally love keeping it rational and precise
so I don't really mind the lack of float support, but I can see why you'd be annoyed.
~~but i spent so much time making the fractional arithmetic perfect -- see Power.__new__() for instance

"""

import inspect
import itertools
import math
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from fractions import Fraction
from functools import cmp_to_key, reduce
from typing import Callable, Dict, List, Literal, NamedTuple, Optional, Tuple, Type, Union

from .combinatorics import generate_permutations, multinomial_coefficient


def nesting(expr: "Expr", var: Optional["Symbol"] = None) -> int:
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
    stri = var.name if var else ""
    if stri not in expr._nesting_cache:
        expr._nesting_cache[stri] = _nesting(expr, var)
    return expr._nesting_cache[stri]


def _nesting(expr: "Expr", var: "Symbol") -> int:
    if var is not None and not expr.contains(var):
        return 0

    # special case
    if isinstance(expr, Prod) and expr.terms[0] == Rat(-1) and len(expr.terms) == 2:
        return __nesting(expr.terms[1], var)
    if isinstance(expr, Prod):
        return 1 + max(__nesting(sub_expr, var) for sub_expr in expr.children())

    if isinstance(expr, Symbol) and (var is None or expr.name == var.name):
        return 1
    elif len(expr.children()) == 0:
        return 0
    else:
        return 1 + max(__nesting(sub_expr, var) for sub_expr in expr.children())


def __nesting(e, v) -> int:
    if isinstance(e, Power) and e.exponent == -1:
        return _nesting(e.base, v)
    return _nesting(e, v)


def _cast(x):
    """Cast x to an Expr if possible."""
    # Check simple cases first and return immediately if possible
    if x is None or x is True or x is False or isinstance(x, Expr):
        return x

    # Check for numerical types, reuse constant instances if possible
    if isinstance(x, (Fraction, float, int)):
        return Const(x)

    # Handling collections with a recursive approach
    if isinstance(x, dict):
        # Consider using a loop or helper function if performance is still an issue
        return {k: _cast(v) for k, v in x.items()}
    elif isinstance(x, tuple):
        return tuple(_cast(v) for v in x)
    elif isinstance(x, list):
        return [_cast(v) for v in x]

    # Class-based checks last since they are least likely and most expensive
    if inspect.isclass(x) and issubclass(x, Expr):
        return x

    raise NotImplementedError(f"Cannot cast {x} to Expr")


def cast(func):
    """Decorator to cast all arguments to Expr."""

    def wrapper(*args, **kwargs) -> "Expr":
        return func(*map(_cast, args), **{k: _cast(v) for k, v in kwargs.items()})

    return wrapper


class Expr(ABC):
    """Base class for all expressions."""

    _fields_already_casted = False  # class attribute

    # These should never change per instance.
    _symbols_cache = None
    _nesting_cache: Dict[str, int] = None
    _expandable_cache = None
    _expand_cache = None

    def __post_init__(self):
        # if any field is an Expr, cast it
        if not self._fields_already_casted:
            for field in fields(self):
                if field.type is Expr:
                    setattr(self, field.name, _cast(getattr(self, field.name)))

        self._nesting_cache = {}

        # Experimental & not guaranteed to be robust so it's private API for now.
        self._strictly_positive = False
        self._is_int = False

    def simplify(self) -> "Expr":
        from .simplify import simplify

        return simplify(self)

    @cast
    def __add__(self, other) -> "Expr":
        return Sum([self, other])

    @cast
    def __radd__(self, other) -> "Expr":
        return Sum([other, self])

    @cast
    def __sub__(self, other) -> "Expr":
        return self + (-1 * other)

    @cast
    def __rsub__(self, other) -> "Expr":
        return other + (-1 * self)

    @cast
    def __mul__(self, other) -> "Expr":
        return Prod([self, other])

    @cast
    def __rmul__(self, other) -> "Expr":
        return Prod([other, self])

    @cast
    def __pow__(self, other) -> "Expr":
        return Power(self, other)

    @cast
    def __rpow__(self, other) -> "Expr":
        return Power(other, self)

    @cast
    def __div__(self, other) -> "Expr":
        return Prod([self, Power(other, -1)])

    @cast
    def __truediv__(self, other) -> "Expr":
        return Prod([self, Power(other, -1)])

    @cast
    def __rdiv__(self, other) -> "Expr":
        return Prod([other, Power(self, -1)])

    @cast
    def __rtruediv__(self, other) -> "Expr":
        return Prod([other, Power(self, -1)])

    def __neg__(self) -> "Expr":
        return -1 * self

    def __abs__(self) -> "Expr":
        return Abs(self)

    def expandable(self) -> bool:
        if self._expandable_cache is None:
            self._expandable_cache = self._expandable()
        return self._expandable_cache

    def expand(self) -> "Expr":
        if self._expand_cache is None:
            self._expand_cache = self._expand()
        return self._expand_cache

    # should be overloaded if necessary
    def _expandable(self) -> bool:
        if not self.children():
            return False
        return any(c.expandable() for c in self.children())

    # overload if necessary
    def _expand(self) -> "Expr":
        """Subclasses: this function should raise AssertionError if self.expandable() is false."""
        if self.expandable():
            raise NotImplementedError(f"Expansion of {self} not implemented")
        raise AssertionError(f"Cannot expand {self}")

    @cast
    @abstractmethod
    def subs(self, subs: Dict[str, "Expr"]) -> "Expr":
        """Substitute variables with expressions."""
        pass

    @abstractmethod
    def _evalf(self, subs: Dict[str, "Expr"]) -> "Expr":
        return self.subs(subs)

    @cast
    def evalf(self, subs: Optional[Dict[str, "Expr"]] = None) -> "Expr":
        """Evaluate the expression to a float."""
        if subs is None:
            subs = {}
        return self._evalf(subs)

    @abstractmethod
    def children(self) -> List["Expr"]:
        raise NotImplementedError(f"Cannot get children of {self.__class__.__name__}")

    def contains(self: "Expr", var: "Symbol") -> bool:
        is_var = isinstance(self, Symbol) and self.name == var.name
        return is_var or any(e.contains(var) for e in self.children())

    def has(self, cls: Type["Expr"]) -> bool:
        from .regex import contains_cls

        return contains_cls(self, cls)

    @abstractmethod
    def diff(self, var: "Symbol") -> "Expr":
        raise NotImplementedError(f"Cannot get the derivative of {self.__class__.__name__}")

    def _symbols(self) -> List["Symbol"]:
        str_set = {symbol.name for e in self.children() for symbol in e.symbols()}
        return [Symbol(name=s) for s in str_set]

    def symbols(self) -> List["Symbol"]:
        """Get all symbols in the expression."""
        if self._symbols_cache is None:
            self._symbols_cache = self._symbols()
        return self._symbols_cache

    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError(f"Cannot represent {self.__class__.__name__}")

    @abstractmethod
    def latex(self) -> str:
        raise NotImplementedError(f"Cannot convert {self.__class__.__name__} to latex")

    @cast
    def __mod__(self, other) -> "Rat":
        return NotImplemented

    @property
    def is_subtraction(self) -> bool:
        """Returns True if the expression would be a subtraction when printed in a sum."""
        return False

    @property
    def is_int(self) -> bool:
        """Returns True if the expression is an integer."""
        return False

    @property
    def symbolless(self) -> bool:
        return len(self.symbols()) == 0

    def as_terms(self):
        if isinstance(self, Sum):
            return self.terms
        return [self]


@dataclass
class Associative:
    """The children's __new__ must handle sorting & flattening."""

    terms: List[Expr]

    def __post_init__(self):
        assert len(self.terms) >= 2
        # Calls super because: If we have Sum(Associative, Expr) and Sum.__post_init__()
        # will call Associative.__post_init__(). the super in Associative will call Expr.__post_init__()
        # (This is according to chatgpt and I haven't confirmed it yet.)
        super().__post_init__()
        # self._flatten()
        # self._sort()

    @classmethod
    def _flatten_terms(cls, terms: List[Expr]):
        """Utility function for flattening a list."""
        new_terms = []
        for t in terms:
            if isinstance(t, cls):
                t._flatten()
                new_terms += t.terms
            else:
                new_terms += [t]
        return new_terms

    def _flatten(self) -> None:
        """Put any sub-products/sub-sums into the parent
        ex: (x * 3) * y -> x * 3 * y
        Can go down as many levels deep as necessary.
        """
        # no need to flatten in simplify because we can then always assume ALL Prod and Sum class objects are flattened
        self.terms = self._flatten_terms(self.terms)

    def children(self) -> List["Expr"]:
        return self.terms

    @classmethod
    def _sort_terms(cls, terms) -> None:
        def _nesting_without_factor(expr: "Expr") -> int:
            """Like the public nesting except constant multipliers don't count
            to prevent fucky reprs for things like
            1 + x^3 + 3*x^2
            """
            from .regex import Any_, get_anys

            if hasattr(expr, "_nesting_without_factor"):
                return expr._nesting_without_factor

            else:
                expr2 = remove_const_factor(expr)
                if isinstance(expr2, (Symbol, Any_)):
                    ans = 1
                elif len(expr2.symbols()) == 0 and len(get_anys(expr2)) == 0:
                    ans = 0
                else:
                    ans = 1 + max(_nesting_without_factor(sub_expr) for sub_expr in expr2.children())
                expr._nesting_without_factor = ans
                return expr._nesting_without_factor

        def _symboless_nest(expr: Expr) -> int:
            if not expr.children():
                return 1 if isinstance(expr, Rat) else 2
            return 2 + max(_symboless_nest(sub_expr) for sub_expr in expr.children())

        def _symboless_compare(a: Expr, b: Expr) -> int:
            """a and b both don't have any symbols.

            Order is first the rat and then float and then sort by nesting including
            """
            na = _symboless_nest(a)
            nb = _symboless_nest(b)
            if na != nb:
                return na - nb
            else:
                return 1 if a.__repr__() > b.__repr__() else -1

        def _compare(a: Expr, b: Expr) -> int:
            """Returns -1 if a < b, 0 if a == b, 1 if a > b.
            The idea is you sort first by nesting, then by power, then by the term alphabetical
            """

            def _deconstruct_const_power(expr: Expr) -> Rat:
                expr = remove_const_factor(expr)
                if isinstance(expr, Power) and isinstance(expr.exponent, Rat):
                    return expr.exponent
                return Rat(1)

            na = _nesting_without_factor(a)
            nb = _nesting_without_factor(b)
            if na == 0 and nb == 0:
                return _symboless_compare(a, b)

            n = na - nb
            if n != 0:
                return n

            power = _deconstruct_const_power(a).value - _deconstruct_const_power(b).value
            if power != 0:
                return power
            return 1 if a.__repr__() > b.__repr__() else -1

        key = cmp_to_key(_compare)
        return sorted(terms, key=key)

    def _evalf(self, subs):
        return self.__class__([term.evalf(subs) for term in self.terms])

    def __iter__(self):
        return iter(self.terms)

    def __len__(self):
        return len(self.terms)


class Num(ABC):
    """Base class -- all numbers.

    all subclasses must implement value
    """

    value = None
    _fields_already_casted = True

    def __post_init__(self):
        super().__post_init__()

    def diff(self, var) -> "Rat":
        return Rat(0)

    def children(self) -> List["Expr"]:
        return []

    @cast
    def subs(self, subs: Dict[str, Expr]):
        return self

    def _evalf(self, subs: Dict[str, Expr]):
        return Float(float(self.value))

    def __repr__(self):
        # default, override if necessary.
        return repr(self.value)

    @cast
    def __eq__(self, other):
        return isinstance(other, Num) and self.value == other.value

    @cast
    def __ge__(self, other):
        return isinstance(other, Num) and self.value >= other.value

    @cast
    def __gt__(self, other):
        return isinstance(other, Num) and self.value > other.value

    @cast
    def __le__(self, other):
        return isinstance(other, Num) and self.value <= other.value

    @cast
    def __lt__(self, other):
        return isinstance(other, Num) and self.value < other.value

    @property
    def is_subtraction(self) -> bool:
        return self.value < 0

    @property
    def is_int(self) -> bool:
        return False

    @property
    def sign(self) -> int:
        return -1 if self.value < 0 else 1


def Const(value: Union[float, Fraction, int]) -> Num:
    """Wrapper to create a Num object from a value."""
    if isinstance(value, (int, Fraction)):
        return Rat(value)

    return Float(value)


class Float(Num, Expr):
    """A decimal number."""

    value: float

    def __new__(cls, value):
        assert isinstance(value, float)

        if value == float("inf"):
            return Infinity()
        if value == float("-inf"):
            return NegInfinity()
        if value == float("NaN"):
            return NaN()

        return super().__new__(cls)

    def __init__(self, value):
        self.value = value
        super().__post_init__()

    def latex(self):
        return repr(self)

    def abs(self) -> "Float":
        return Float(abs(self.value))

    def __neg__(self) -> "Float":
        return Float(-self.value)

    @property
    def is_int(self) -> bool:
        return int(self.value) == self.value


class Rat(Num, Expr):
    """A rational number."""

    value: Fraction

    def __init__(self, value: Union[Fraction, int, float], denom: int = 1):
        """I would check types but asserts are... expensive lol.

        if value is a float where int(value) != value, too bad you're getting int'd
        - this is also coincidentally sympy's behaviour

        if value is fraction, denom will be ignored.

        like just
        assert isinstance(value, Fraction) or int(value) == value, f"Got {value}, not allowed const"
        makes everything 5-7% slower
        """
        if not isinstance(value, Fraction):
            value = Fraction(int(value), int(denom))
        self.value = value
        super().__post_init__()

    def __repr__(self) -> str:
        return str(self.value)

    ### These are unnecessary because it would get simplified to this by Sum/Prod anyways (it's tested---remove these, all tests pass)
    # but I'm doing them for some sense of "efficiency" (not really, isn't actually an efficiency bottleneck at all.)
    @cast
    def __add__(self, other) -> "Expr":
        if isinstance(other, Rat):
            return Rat(self.value + other.value)
        return super().__add__(other)

    @cast
    def __radd__(self, other) -> "Expr":
        if isinstance(other, Rat):
            return Rat(other.value + self.value)
        return super().__radd__(other)

    @cast
    def __sub__(self, other) -> "Expr":
        if isinstance(other, Rat):
            return Rat(self.value - other.value)
        return super().__sub__(other)

    @cast
    def __rsub__(self, other) -> "Expr":
        if isinstance(other, Rat):
            return Rat(other.value - self.value)
        return super().__rsub__(other)

    @cast
    def __mul__(self, other) -> "Expr":
        if isinstance(other, Rat):
            return Rat(self.value * other.value)
        return super().__mul__(other)

    @cast
    def __rmul__(self, other) -> "Expr":
        if isinstance(other, Rat):
            return Rat(other.value * self.value)
        return super().__rmul__(other)

    @cast
    def __truediv__(self, other) -> "Expr":
        if isinstance(other, Rat) and other.value != 0:
            return Rat(self.value / other.value)
        return super().__truediv__(other)

    @cast
    def __rtruediv__(self, other) -> "Expr":
        if isinstance(other, Rat) and self.value != 0:
            return Rat(other.value / self.value)
        return super().__rtruediv__(other)

    def __neg__(self):
        return Rat(-self.value)

    @cast
    def __pow__(self, other) -> Expr:
        if isinstance(other, Rat):
            value = self.value**other.value
            if isinstance(value, Fraction):
                return Rat(value)
        return super().__pow__(other)

    @cast
    def __rpow__(self, other) -> Expr:
        if isinstance(other, Rat):
            return other.__pow__(self)
        return super().__rpow__(other)

    def __abs__(self) -> "Rat":
        return Rat(abs(self.value))

    ###

    @cast
    def __floordiv__(self, other) -> "Rat":
        if isinstance(other, Rat):
            return Rat(self.value // other.value)
        return NotImplemented

    def latex(self) -> str:
        from .latex import group

        if self.value.denominator == 1:
            return f"{self.value}"
        return "\\frac " + group(str(self.value.numerator)) + group(str(self.value.denominator))

    def reciprocal(self) -> "Rat":
        return Rat(self.value.denominator, self.value.numerator)

    @cast
    def __mod__(self, other) -> "Rat":
        if isinstance(other, Rat):
            return Rat(self.value % other.value)
        else:
            return NotImplemented

    @property
    def is_int(self) -> bool:
        return self.value.denominator == 1

    @property
    def numerator(self) -> "Rat":
        return Rat(self.value.numerator)

    @property
    def denominator(self) -> "Rat":
        return Rat(self.value.denominator)


@dataclass
class Pi(Num, Expr):
    value = 3.141592653589793

    def __repr__(self) -> str:
        return "pi"

    def latex(self) -> str:
        return "\\pi"


@dataclass
class E(Num, Expr):
    value = 2.718281828459045

    def __repr__(self) -> str:
        return "e"

    def latex(self) -> str:
        return "e"

    def __eq__(self, other) -> bool:
        return isinstance(other, E)


class Infinity(Num, Expr):
    value = float("inf")

    def __init__(self):
        pass

    def latex(self) -> str:
        return "\\infty"

    def __abs__(self) -> "Infinity":
        return self

    def reciprocal(self) -> "Rat":
        return Rat(0)

    def __neg__(self):
        return NegInfinity()


class NegInfinity(Num, Expr):
    value = float("-inf")

    def __init__(self):
        pass

    def latex(self) -> str:
        return "-\\infty"

    def __abs__(self) -> "Infinity":
        return inf

    def reciprocal(self) -> "Rat":
        return Rat(0)

    def __neg__(self):
        return inf


class NaN(Num, Expr):
    value = float("NaN")

    def __init__(self):
        pass

    def latex(self):
        return NotImplemented


pi = Pi()
e = E()
inf = Infinity()  # should division by zero be inf or be zero divisionerror or? // sympy makes it zoo
# not gonna export this yet bc it's experimental?


Accumulateable = Union[Rat, Infinity, NegInfinity, Float]
AccumulaTuple = (Rat, Infinity, NegInfinity, Float)  # because can't use Union in isinstance checks.


def accumulate(*consts: List[Accumulateable], type_: Literal["sum", "prod"] = "sum") -> Accumulateable:
    """Accumulate constants into a single constant.

    New rule is that we're just gonna make Float * Rat = Float

    If it's the identity, returns None.
    """

    if type_ == "sum":
        operation = sum
    elif type_ == "prod":
        operation = lambda terms: reduce(lambda x, y: x * y, terms, 1)

    if len(consts) == 1:
        return consts[0]
    else:
        return Const(operation(c.value for c in consts))


def _accumulate_power(b: Accumulateable, x: Accumulateable) -> Optional[Expr]:
    """If returns None, means it cannot be simplified."""
    if isinstance(b, (Infinity, NegInfinity, Float)) or isinstance(x, (Infinity, NegInfinity, Float)):
        return Const(b.value**x.value)

    if b == 0:
        if x > 0:
            return Rat(0)
        else:
            # Division by zero.
            return inf

    if isinstance(b, Rat) and isinstance(x, Rat):
        # If the answer is rational, return right away
        num = b.value.numerator ** abs(x.value)
        den = b.value.denominator ** abs(x.value)
        isint = lambda x: int(x.real) == x
        if isint(num) and isint(den):
            if x > 0:
                return Rat(int(num), int(den))
            else:
                return Rat(int(den), int(num))

        # Rewriting for consistency between same values.
        if b.value.numerator == 1:
            return Power(b.reciprocal(), -x)
        elif b.value.denominator != 1 and x < 0:
            return Power(b.reciprocal(), -x)

        if x.value.denominator % 2 == 0 and b < 0:
            # Cannot be simplified further.
            return
        elif b < 0:
            return -1 * Power(-b, x)

        ans = None
        if isint(num) and num != 1:
            ans = Prod([Rat(num), Power(b.denominator, -x, skip_checks=True)], skip_checks=True)
        elif isint(den) and den != 1:
            ans = Prod([Rat(1, den), Power(b.numerator, x, skip_checks=True)], skip_checks=True)
        if ans:
            if x > 0:
                return ans
            else:
                return 1 / ans


@dataclass
class Symbol(Expr):
    """A symbol. A variable."""

    name: str

    def __post_init__(self):
        super().__post_init__()
        assert len(self.name) > 0, "Symbol name cannot be empty"

    def __repr__(self) -> str:
        return self.name

    @cast
    def subs(self, subs: Dict[str, "Rat"]):
        return subs.get(self.name, self)

    def _evalf(self, subs):
        return subs.get(self.name, self)

    def diff(self, var) -> Rat:
        return Rat(1) if self == var else Rat(0)

    def __eq__(self, other):
        return isinstance(other, Symbol) and self.name == other.name

    def children(self) -> List["Expr"]:
        return []

    def symbols(self) -> List["Expr"]:
        return [self]

    def latex(self) -> str:
        return self.name


def _combine_like_terms_sum(terms: List[Expr]) -> List[Expr]:
    """accumulate all like terms of a sum"""

    consts = []
    non_constant_terms = []
    for i, term in enumerate(terms):
        if term is None:
            continue
        is_hit = False
        if isinstance(term, AccumulaTuple):
            consts.append(term)
            continue

        coeffs, non_const_factors1 = _deconstruct_prod(term)

        # check if any later terms are the same
        for j in range(i + 1, len(terms)):
            term2 = terms[j]
            if term2 is None:
                continue

            coeffs2, non_const_factors2 = _deconstruct_prod(term2)

            if non_const_factors1 == non_const_factors2:
                if not is_hit:
                    is_hit = True
                coeffs = coeffs + coeffs2  # use + instead of extend to not mutate the original list
                terms[j] = None

        # Yes you can replace the next few lines with Prod([new_coeff] + non_const_factors1)
        # but by skipping checks for combine like terms, we make it significantly faster.
        # Doing this speeds up the sum constructor by ~30% and everything by ~7%
        if not is_hit:
            non_constant_terms.append(term)
            continue

        coeff = accumulate(*coeffs)
        if coeff == 0:
            continue
        elif coeff == 1:
            non_constant_terms.append(Prod(non_const_factors1, skip_checks=True))
        else:
            non_constant_terms.append(Prod([coeff] + non_const_factors1, skip_checks=True))

    # accumulate all constants
    if consts:
        const = accumulate(*consts)
        if const != 0:
            non_constant_terms.append(const)

    return non_constant_terms


@dataclass
class Sum(Associative, Expr):
    """A sum expression."""

    _fields_already_casted = True

    def __new__(cls, terms: List[Expr], *, skip_checks: Union[bool, Literal["sort"]] = False) -> "Expr":
        """When a sum is initiated:
        - terms are converted to expr
        - flatten
        - accumulate like terms & constants
        - sort
        """
        if skip_checks:
            if len(terms) == 0:
                return Rat(0)
            if len(terms) == 1:
                return terms[0]
            return super().__new__(cls)

        terms = _cast(terms)
        terms = cls._flatten_terms(terms)
        final_terms = _combine_like_terms_sum(terms)

        if len(final_terms) == 0:
            return Rat(0)
        if len(final_terms) == 1:
            return final_terms[0]

        final_terms = cls._sort_terms(final_terms)

        instance = super().__new__(cls)
        instance.terms = final_terms
        return instance

    def __init__(self, terms: List[Expr], *, skip_checks: Union[bool, Literal["sort"]] = False):
        # Overrides the shit that does self.terms = terms because i've already set terms
        # in __new__.
        if skip_checks is True:
            self.terms = terms
        if skip_checks == "sort":
            self.terms = self._sort_terms(terms)
        super().__post_init__()

    @classmethod
    def _sort_terms(cls, terms):
        # Sums are typically written from largest complexity to smallest (whereas for products it's the opposite)
        terms = super()._sort_terms(terms)
        return list(reversed(terms))

    def __neg__(self) -> "Sum":
        return Sum([-t for t in self.terms])

    def _expand(self) -> Expr:
        assert self.expandable(), f"Cannot expand {self}"
        return Sum([t.expand() if t.expandable() else t for t in self.terms])

    @cast
    def subs(self, subs: Dict[str, "Rat"]):
        return Sum([t.subs(subs) for t in self.terms])

    def diff(self, var) -> "Sum":
        return Sum([diff(e, var) for e in self.terms])

    def __repr__(self) -> str:
        ongoing_str = ""
        for i, term in enumerate(self.terms):
            if i == 0:
                ongoing_str += f"{term}"
            elif term.is_subtraction:
                ongoing_str += f" - {-term}"
            else:
                ongoing_str += f" + {term}"

        return ongoing_str

    def latex(self) -> str:
        ongoing_str = ""
        for i, term in enumerate(self.terms):
            if i == 0:
                ongoing_str += term.latex()
            elif term.is_subtraction:
                ongoing_str += f" - {(-term).latex()}"
            else:
                ongoing_str += f" + {term.latex()}"

        return ongoing_str

    def factor(self) -> Union["Prod", "Sum"]:
        """
        This method currently does:
        - If there is a factor that is common to all terms, factor it out.
        - If there is a factor that is common to some terms, let's just ignore it.
        """
        # TODO: this feels like not the most efficient algo
        # TODO: doesn't factor ex. quadratics into 2 binomials. implement some form of multi-term polynomial factoring at some point
        # (needed for partial fractions)

        def _df(term: Expr) -> Tuple[Rat, Optional[List[Tuple[Expr, int, bool]]]]:
            """Deconstruct a term into its factors.

            Returns: Number, List[(factor, abs(exponent), sign(exponent))]
            """
            if isinstance(term, Prod):
                num, terms = _deconstruct_prod(term)
                return Prod(num, skip_checks=True), [_df(f)[1][0] for f in terms]
            if isinstance(term, Power) and isinstance(term.exponent, Rat):
                return Rat(1), [[term.base, abs(term.exponent), term.exponent.value > 0]]
            if isinstance(term, Rat):
                return term, [[term, Rat(1), True]]
            return Rat(1), [[term, Rat(1), True]]

        dfs = [_df(term) for term in self.terms]
        factors_per_term = [d[1] for d in dfs]
        coeffs = [d[0] for d in dfs]
        common_factors = factors_per_term[0]

        for this_terms_factors in factors_per_term[1:]:
            for i, cfactor in enumerate(common_factors):
                if cfactor is None:
                    continue
                is_in_at_least_1 = False
                for tfactor in this_terms_factors:
                    if cfactor[0].__repr__() == tfactor[0].__repr__() and cfactor[2] == tfactor[2]:
                        cfactor[1] = min(cfactor[1], tfactor[1])
                        is_in_at_least_1 = True
                        break
                if not is_in_at_least_1:
                    common_factors[i] = None

        # Factor coeffs
        common_coeff = abs(coeffs[0])
        for c in coeffs[1:]:
            x: Rat = c / common_coeff
            y: Rat = common_coeff / c
            if x.value.denominator == 1 or y.value.denominator == 1:
                common_coeff = min(abs(c), abs(common_coeff))
            else:
                common_coeff = None
                break
        is_negative = all(c < 0 for c in coeffs)
        if is_negative and common_coeff:
            common_coeff *= -1

        common_factors = [f for f in common_factors if f is not None]

        def _makeprod(terms: List[Tuple[Expr, int, bool]]):
            return Rat(1) if len(terms) == 0 else Prod([Power(t[0], t[1] if t[2] else -t[1]) for t in terms])

        common_expr = _makeprod(common_factors)
        if common_coeff:
            common_expr *= common_coeff

        # factor out the common factors
        new_terms = []
        for term in self.terms:
            new_terms.append(term / common_expr)

        return common_expr * Sum(new_terms)

    @property
    def is_subtraction(self):
        return all(t.is_subtraction for t in self.terms)


def _deconstruct_prod(expr: Expr) -> Tuple[List[Accumulateable], List[Expr]]:
    """turns an expression into a constant and a list of other terms.
    constant can be any accumulateable. <- the purpose fo this function is to accumulate like terms
        in a sum, so numbers like pi that are not accumulateable are ignored.

    ex: 3*x^2*y -> ([3], [x^2, y])
    ex: 2*3.3*x^2*y -> ([2, 3.3], [x^2, y])
    """

    if hasattr(expr, "_deconstruct_prod_cache"):
        return expr._deconstruct_prod_cache

    def _dp(expr: Expr):
        if isinstance(expr, Prod):
            non_const_factors = []
            const_factors = []
            for term in expr.terms:
                if isinstance(term, AccumulaTuple):
                    const_factors.append(term)
                else:
                    non_const_factors.append(term)
            coeff = const_factors if const_factors else [Rat(1)]
        else:
            non_const_factors = [expr]
            coeff = [Rat(1)]
        return (coeff, non_const_factors)

    expr._deconstruct_prod_cache = _dp(expr)
    return expr._deconstruct_prod_cache


def deconstruct_power(expr: Expr) -> Tuple[Expr, Expr]:
    # x^3 -> (x, 3). x -> (x, 1). 3 -> (3, 1)
    if hasattr(expr, "_deconstruct_power_cache"):
        return expr._deconstruct_power_cache

    def _dpo(expr):
        if isinstance(expr, Power):
            return expr.base, expr.exponent
        else:
            return expr, Rat(1)

    expr._deconstruct_power_cache = _dpo(expr)
    return expr._deconstruct_power_cache


isfractionorneg = lambda x: isinstance(x, Rat) and (x.value.denominator != 1 or x < 0)
islongsymbol = lambda x: isinstance(x, Symbol) and len(x.name) > 1 or x.__class__.__name__ == "Any_" and len(x.key) > 1


def _combine_like_terms(initial_terms: List[Expr]) -> List[Expr]:
    """accumulates all like terms of a product.

    Takesin a list of terms, returns a list of terms
    """
    consts = []
    non_constant_terms = []
    decon = {}

    def _add_term(term):
        if isinstance(term, AccumulaTuple):
            consts.append(term)
        else:
            non_constant_terms.append(term)

    for i, term in enumerate(initial_terms):
        is_hit = False
        if term is None:
            continue
        if not i in decon:
            decon[i] = deconstruct_power(term)
        base, expo = decon[i]

        # other terms with same base
        for j in range(i + 1, len(initial_terms)):
            if initial_terms[j] is None:
                continue
            if not j in decon:
                decon[j] = deconstruct_power(initial_terms[j])
            other_base, other_expo = decon[j]
            if other_base == base:
                is_hit = True
                expo += other_expo
                initial_terms[j] = None
                initial_terms[i] = None

        if not is_hit:
            _add_term(term)
            continue

        if expo == 0:
            continue
        if expo == 1:
            _add_term(base)
            continue

        _add_term(Power(base, expo))

    if consts:
        coeff = accumulate(*consts, type_="prod")
        if coeff == 0:
            return [coeff]
        if coeff != 1:
            non_constant_terms.append(coeff)
    return non_constant_terms


@dataclass
class Prod(Associative, Expr):
    """A product expression."""

    _numerator_denominator_cache = None
    _fields_already_casted = True

    def __new__(cls, terms: List[Expr], *, skip_checks: bool = False) -> "Expr":
        if skip_checks:
            if len(terms) == 0:
                return Rat(1)
            if len(terms) == 1:
                return terms[0]
            return super().__new__(cls)

        if any(isinstance(t, Piecewise) for t in terms):
            piecewise = [t for t in terms if isinstance(t, Piecewise)][0]
            other_terms = [t for t in terms if not isinstance(t, Piecewise)]
            return piecewise * Prod(other_terms)

        # We need to flatten BEFORE we accumulate like terms
        # ex: Prod(x, Prod(Power(x, -1), y))
        terms = _cast(terms)
        terms = cls._flatten_terms(terms)
        new_terms = _combine_like_terms(terms)

        if len(new_terms) == 0:
            return Rat(1)
        if len(new_terms) == 1:
            return new_terms[0]

        new_terms = cls._sort_terms(new_terms)

        if len(new_terms) == 2 and new_terms[0] == Rat(-1) and isinstance(new_terms[1], Sum):
            # do not let allow this to exist as a Prod because it causes a fucky repr
            # ex -1 * (x + y) will get displayed as -x + y the way that repr is currently done.
            # this is innaccurate and dangerous.
            return -new_terms[1]

        instance = super().__new__(cls)
        instance.terms = new_terms
        return instance

    def __init__(self, terms: List[Expr], *, skip_checks: bool = False):
        if skip_checks:
            self.terms = terms
        # terms are already set in __new__
        super().__post_init__()

    def __repr__(self) -> str:
        def _term_repr(term):
            if isinstance(term, Sum) or islongsymbol(term):
                return "(" + repr(term) + ")"
            return repr(term)

        # special case for subtraction:
        if self.is_subtraction:
            new_prod = self * -1
            if not isinstance(new_prod, Prod):
                return f"-{_term_repr(new_prod)}"
            if new_prod.is_subtraction:
                from .debug.utils import debug_repr

                raise ValueError(f"Cannot get repr of {debug_repr(self)}")
            return "-" + new_prod.__repr__()

        numerator, denominator = self.numerator_denominator
        if denominator != Rat(1):

            def _x(expr: Expr, b=True):
                """b: bracketize multiple terms (boolean)"""
                if not isinstance(expr, Prod):
                    return _term_repr(expr)
                return "(" + repr(expr) + ")" if b else repr(expr)

            return _x(numerator, b=False) + "/" + _x(denominator)

        return "*".join(map(_term_repr, self.terms))

    def latex(self) -> str:
        from .latex import bracketfy, group

        def _term_latex(term: Expr):
            if isinstance(term, Sum):
                return bracketfy(term)
            return term.latex()

        # special case for subtraction:
        if self.is_subtraction:
            new = self * -1
            if not isinstance(new, Prod):
                return "-" + _term_latex(new)
            return "-" + new.latex()

        numerator, denominator = self.numerator_denominator
        if denominator != Rat(1):
            # don't need brackets around num/denom bc the frac bar handles it.
            return "\\frac " + group(numerator) + group(denominator)

        return " \\cdot ".join(map(_term_latex, self.terms))

    @property
    def _numerator_denominator(self) -> Tuple[Expr, Expr]:
        denominator = []
        numerator = []
        for term in self.terms:
            # handle consts seperately
            if isinstance(term, Rat):
                if term.value.numerator != 1:
                    numerator.append(Rat(term.value.numerator))
                if term.value.denominator != 1:
                    denominator.append(Rat(term.value.denominator))
                continue

            b, x = deconstruct_power(term)
            if isinstance(x, Num) and x.value < 0:
                denominator.append(b if x == Rat(-1) else Power(b, -x))
            else:
                numerator.append(term)

        num_expr = Prod(numerator, skip_checks=True)
        denom_expr = Prod(denominator, skip_checks=True)
        return [num_expr, denom_expr]

    @property
    def numerator_denominator(self) -> Tuple[Expr, Expr]:
        if self._numerator_denominator_cache is None:
            self._numerator_denominator_cache = self._numerator_denominator
        return self._numerator_denominator_cache

    @property
    def is_subtraction(self):
        return isinstance(self.terms[0], Rat) and self.terms[0] < 0

    def _expandable(self) -> bool:
        # a product is expandable if it contains any sums in the numerator
        # OR if it contains sums in the denominator AND the denominator has another term other than the sum
        # (so, a singular sum in a numerator is expandable but a single sum in the denominator isn't.)
        if any(isinstance(t, Sum) or t.expandable() for t in self.terms):
            return True
        num, den = self.numerator_denominator
        if isinstance(den, Prod) and any(isinstance(t, Sum) for t in den.terms):
            return True
        return False

    def _expand(self):
        assert self.expandable(), f"Cannot expand {self}"
        # expand sub-expressions
        num, denom = self.numerator_denominator
        if denom.expandable():
            denom = denom.expand()

        if not isinstance(denom, Prod):
            denom_terms = [Power(denom, -1)]
        else:
            denom_terms = [Power(t.base, -t.exponent) if isinstance(t, Power) else Power(t, -1) for t in denom.terms]

        # now we assume denom is good and we move on with life as usual
        sums: List[Sum] = []
        other = []
        if not isinstance(num, Prod):
            num = num.expand() if num.expandable() else num
            if isinstance(num, Sum):
                sums.append(num)
            else:
                return Prod([num] + denom_terms)
        else:
            for t in num.terms:
                te = t.expand() if t.expandable() else t
                if isinstance(te, Sum):
                    sums.append(te)
                else:
                    other.append(te)

            if not sums:
                return Prod(sums + other + denom_terms)

        # for every combination of terms in the sums, multiply them and add
        # (using itertools)
        final_sum_terms = []
        for expanded_terms in itertools.product(*[s.terms for s in sums]):
            final_sum_terms.append(Prod(other + list(expanded_terms) + denom_terms))

        return Sum(final_sum_terms)

    @cast
    def subs(self, subs: Dict[str, "Rat"]):
        return Prod([t.subs(subs) for t in self.terms])

    def diff(self, var) -> Sum:
        return Sum([Prod([diff(e, var)] + [t for t in self.terms if t is not e]) for e in self.terms])


def _multiply_exponents(b: Expr, x1: Expr, x2: Expr) -> Expr:
    """(b^x1)^x2 -> b^(x1*x2)

    But if it's like sqrt(b^2), where the base 'b' has a radical expression
    and the exponent outside the radical is even (like 2 in this example),
    it should return abs(b). This is because raising a number with an even
    exponent inside a radical to another even exponent cancels out the radical
    and results in the absolute value of the base.
    """
    # TODO: this is not exhausitve, like when x1 and x2 aren't exactly reciprocals.
    # but i'm fine with this for now.
    if isinstance(x1, Rat) and x1 % 2 == 0:
        if isinstance(x2, Rat) and x1 == x2.reciprocal():
            return Abs(b)

    return b ** (x1 * x2)


def exp(x: Expr) -> Expr:
    return e**x


@dataclass
class Power(Expr):
    base: Expr
    exponent: Expr

    _fields_already_casted = True

    def __repr__(self) -> str:
        def _term_repr(term):
            if isinstance(term, Sum) or isinstance(term, Prod) or isfractionorneg(term) or islongsymbol(term):
                return "(" + repr(term) + ")"
            return repr(term)

        # represent negative powers as reciprocals
        if self.exponent == Rat(-1):
            return "1/" + _term_repr(self.base)
        if isinstance(self.exponent, (Rat, Float)) and self.exponent < 0:
            new_power = Power(self.base, -self.exponent)
            return "1/" + repr(new_power)

        # special case for sqrt
        if self.exponent == Rat(1, 2):
            return _repr(self.base, "sqrt")

        return f"{_term_repr(self.base)}^{_term_repr(self.exponent)}"

    def latex(self) -> str:
        from .latex import bracketfy, group

        def _base_latex(term: Expr):
            if isinstance(term, Sum) or isinstance(term, Prod) or isfractionorneg(term) or islongsymbol(term):
                return bracketfy(term)
            return term.latex()

        # special case for sqrt
        if self.exponent == Rat(1, 2):
            return "\\sqrt{" + self.base.latex() + "}"
        if self.exponent == Rat(-1, 2):
            return "{\\sqrt{" + self.base.latex() + "}" + "}^{-1}"

        return _base_latex(self.base) + "^" + group(self.exponent)

    def __new__(cls, base: Expr, exponent: Expr, *, skip_checks: bool = False) -> "Expr":
        if skip_checks:
            return super().__new__(cls)

        b = _cast(base)
        x = _cast(exponent)

        default_return = super().__new__(cls)
        default_return.base = b
        default_return.exponent = x

        if x == 0:
            # Ok this is up to debate, but since python does 0**0 = 1 I'm gonna do it too.
            # "0**0 represents the empty product (the number of sets of 0 elements that can be chosen from a set of 0 elements), which by definition is 1. This is also the same reason why anything else raised to the power of 0 is 1."
            return Rat(1)
        if x == 1:
            return b
        if b == 1:
            return Rat(1)
        if isinstance(b, AccumulaTuple) and isinstance(x, AccumulaTuple):
            ans = _accumulate_power(b, x)
            if ans is None:
                return default_return
            return ans
        if isinstance(b, Power):
            # Have to call the class here. In the case that x*b.exponent = 1, this will have to re-simplfiy
            # through calling the constructor.
            return _multiply_exponents(b.base, b.exponent, x)
        if isinstance(b, Prod) and not b.is_subtraction:
            # when you construct this new power entity you have to simplify it.
            # because what if the term raised to this exponent can be simplified?
            # ex: if you have (ab)^n where a = c^m

            # it gets fucky if we isolate a negative factor of the base, so we won't bother with that.
            return Prod([Power(term, x) for term in b.terms])
        if isinstance(x, log) and b == x.base:
            return x.inner
        if isinstance(x, Prod):
            for i, t in enumerate(x.terms):
                if isinstance(t, log) and t.base == b:
                    rest = Prod(x.terms[:i] + x.terms[i + 1 :], skip_checks=True)
                    return Power(
                        t.inner, rest
                    )  # Just in case this needs to be simplified, we will pass this through the constructor
                    # return Power(Power(b, t), rest)
                    # Power(b, t) will be simplified to t.inner
        if b.is_subtraction and isinstance(x, Rat) and x.value.denominator == 1:
            if x.value.numerator % 2 == 0:
                return Power(-b, x)
            else:
                return -Power(-b, x)
        # not fully exhaustive.
        if (
            isinstance(b, Num) or isinstance(b, Prod) and all(isinstance(t, Num) for t in b.terms)
        ) and not b.is_subtraction:
            if x == inf:
                return inf
            if x == -inf:
                return Rat(0)

        return default_return

    def __init__(self, base: Expr, exponent: Expr, skip_checks: bool = False):
        if skip_checks:
            self.base = base
            self.exponent = exponent
        self.__post_init__()

    def _power_expandable(self) -> bool:
        return (
            isinstance(self.exponent, Rat)
            and self.exponent.is_int
            and abs(self.exponent) != 1
            and isinstance(self.base, Sum)
        )

    def _expandable(self) -> bool:
        return self._power_expandable() or self.base.expandable() or self.exponent.expandable()

    def _expand(self) -> Expr:
        assert self.expandable(), f"Cannot expand {self}"
        b = self.base.expand() if self.base.expandable() else self.base
        x = self.exponent.expand() if self.exponent.expandable() else self.exponent
        new = Power(b, x)
        if not isinstance(new, Power) or not new._power_expandable():
            return new

        expanded = []
        n = abs(new.exponent.value.numerator)
        i = len(new.base.terms)
        permutations = generate_permutations(i, n)
        for permutation in permutations:
            new_term = [Power(t, p) for t, p in zip(new.base.terms, permutation)]
            coefficient = multinomial_coefficient(permutation, n)
            expanded.append(Prod([Rat(coefficient)] + new_term))
        return Sum(expanded) ** new.exponent.sign

    @cast
    def subs(self, subs: Dict[str, Expr]):
        return Power(self.base.subs(subs), self.exponent.subs(subs))

    def _evalf(self, subs: Dict[str, Expr]):
        return Power(self.base._evalf(subs), self.exponent._evalf(subs))

    def children(self) -> List["Expr"]:
        return [self.base, self.exponent]

    def diff(self, var) -> Expr:
        if not self.exponent.contains(var):
            return self.exponent * self.base ** (self.exponent - 1) * self.base.diff(var)
        if not self.base.contains(var):
            return log(self.base) * self * self.exponent.diff(var)

        # if both base and exponent contain var
        raise NotImplementedError("Power differentiation not implemented for both base and exponent containing var")
        # new_power = Power(e, self.exponent * log(self.base))
        # return new_power.diff(var)

    @property
    def is_subtraction(self) -> bool:
        # if base is negative and the -1 can be factored out, we would have factored it out in new.
        return False


@dataclass
class SingleFunc(Expr):
    inner: Expr

    @property
    @abstractmethod
    def _label(self) -> str:
        raise NotImplementedError("Label not implemented")

    def children(self) -> List["Expr"]:
        return [self.inner]

    def __repr__(self) -> str:
        return _repr(self.inner, self._label)

    def latex(self) -> str:
        from .latex import bracketfy

        return "\\text{" + self._label + "} " + bracketfy(self.inner)

    @cast
    def subs(self, subs: Dict[str, "Expr"]):
        inner = self.inner.subs(subs)
        return self.__class__(inner)

    def _expand(self) -> Expr:
        assert self.inner.expandable(), f"Cannot expand {self}"
        return self.__class__(self.inner.expand())


def _repr(inner: Expr, label: str) -> str:
    inner_repr = inner.__repr__()
    if inner_repr[0] == "(" and inner_repr[-1] == ")":
        return f"{label}{inner_repr}"
    return f"{label}({inner_repr})"


@dataclass
class log(Expr):
    inner: Expr
    base: Expr = e

    @property
    def _label(self):
        if self.base == e:
            return "ln"
        elif isinstance(self.base, Rat) and self.base.is_int:
            return "log" + self.base
        else:
            return f"log[base={self.base}]"

    def __repr__(self) -> str:
        return _repr(self.inner, self._label)

    @cast
    def subs(self, subs: Dict[str, "Expr"]):
        inner = self.inner.subs(subs)
        base = self.base.subs(subs)
        return log(inner, base)

    def _expand(self) -> Expr:
        assert self.inner.expandable(), f"Cannot expand {self}"
        return self.__class__(self.inner.expand())

    def children(self) -> List[Expr]:
        return [self.inner, self.base]

    def latex(self) -> str:
        # Have informally tested this; does the job.
        from .latex import bracketfy

        if self.base == e:
            return "\\ln " + bracketfy(self.inner)
        else:
            return "\\log_{" + self.base.latex() + "} " + bracketfy(self.inner)

    @cast
    def __new__(cls, inner: Expr, base: Expr = e, *, skip_checks: bool = False):
        if skip_checks:
            super().__new__(cls)

        if inner == 1:
            return Rat(0)
        if inner == base:
            return Rat(1)
        if isinstance(inner, Power) and inner.base == base:
            return inner.exponent
        if isinstance(inner, Float):
            if base == e:
                return Float(math.log(inner.value))
            if isinstance(base, Float):
                return Float(math.log(inner.value) / math.log(base.value))

        return super().__new__(cls)

    def __init__(self, inner: Expr, base: Expr = e, *, skip_checks: bool = False):
        self.inner = inner
        self.base = base
        self.__post_init__()

    def diff(self, var) -> Expr:
        return self.inner.diff(var) / self.inner

    def _evalf(self, subs: Dict[str, "Expr"]):
        inner = self.inner._evalf(subs)
        base = self.base._evalf(subs)
        return Const(math.log(inner.value)) if isinstance(inner, Num) and isinstance(base, Num) else log(inner, base)


@cast
def sqrt(x: Expr) -> Expr:
    return x ** Rat(1, 2)


class classproperty:
    """python 3.11 no longer supports
    @classmethod
    @property

    so we make our own :)
    """

    def __init__(self, func):
        self.func = func

    def __get__(self, instance, cls):
        return self.func(cls)


class staticproperty:
    """
    @staticmethod
    @property
    """

    def __init__(self, func):
        self.func = func

    def __get__(self, instance, cls):
        return self.func()


TrigStr = Literal["sin", "cos", "tan", "sec", "csc", "cot"]


class TrigFunction(SingleFunc, ABC):
    is_inverse: bool  # class property
    _fields_already_casted = True
    _odd = False
    _even = False

    _SPECIAL_KEYS = [
        "0",
        "1/6",
        "1/4",
        "1/3",
        "1/2",
        "2/3",
        "3/4",
        "5/6",
        "1",
        "7/6",
        "5/4",
        "4/3",
        "3/2",
        "5/3",
        "7/4",
        "11/6",
    ]
    _special_values_cache = None

    @abstractmethod
    @classproperty
    def func(cls) -> TrigStr:
        pass

    @staticproperty
    def _double_simplification_dict() -> Dict[str, Callable[[Expr], Expr]]:
        # Making this a property instead of a class attribute or a global var because it requires computation.
        return {
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

    @classproperty
    def reciprocal_class(cls) -> Type["TrigFunction"]:
        if cls.is_inverse:
            raise ValueError(f"Inverse trig function {cls.__name__} does not have a reciprocal")
        else:
            raise NotImplementedError(f"reciprocal_class not implemented for {cls.__name__}")

    @classproperty
    @abstractmethod
    def _special_values(cls) -> Dict[str, Expr]:
        pass

    @classproperty
    def special_values(cls) -> Dict[str, Expr]:
        if cls._special_values_cache is None:
            cls._special_values_cache = cls._special_values
        return cls._special_values_cache

    @classproperty
    @abstractmethod
    def _func(cls) -> Callable[[float], float]:
        """Function that acts on floats"""
        pass

    @property
    def _label(self) -> str:
        return f"{'a' if self.is_inverse else ''}{self.func}"

    @cast
    def __new__(cls, inner: Expr, *, skip_checks: bool = False) -> "Expr":
        if skip_checks:
            return super().__new__(cls)

        # 1. Check if inner is a special value
        if inner == 0:
            return cls.special_values["0"]

        if not cls.is_inverse and inner.has(Pi):
            pi_coeff = inner / pi
            if isinstance(pi_coeff, Rat):
                pi_coeff = pi_coeff % 2
                if str(pi_coeff.value) in cls._SPECIAL_KEYS:
                    return cls.special_values[str(pi_coeff.value)]

            if (pi_coeff / 2)._is_int:
                return cls.special_values["0"]

            # check if inner has ... a 2pi term in a sum
            if isinstance(inner, Sum):
                for t in inner.terms:
                    if t.has(Pi):
                        coeff = t / pi
                        if isinstance(coeff, Rat) and coeff % cls._period == 0:
                            instance = super().__new__(cls)
                            instance.inner = inner - t
                            return instance

        # Odd and even shit
        if cls._odd and inner.is_subtraction:
            return -cls(-inner)
        if cls._even and inner.is_subtraction:
            return cls(-inner)

        # 2. Check if inner is trigfunction
        # things like sin(cos(x)) cannot be more simplified.
        if isinstance(inner, TrigFunction) and inner.is_inverse != cls.is_inverse:
            # sin(asin(x)) -> x
            if inner.func == cls.func:
                return inner.inner

            # sin(acsc(x)) -> 1/x
            if (
                inner.is_inverse
                and inner.func == cls.reciprocal_class.func
                or not inner.is_inverse
                and inner.reciprocal_class.func == cls.func
            ):
                return 1 / inner.inner

            if not cls.is_inverse:
                if cls.func in ["sin", "cos", "tan"]:
                    callable_ = cls._double_simplification_dict[f"{cls.func} {inner._label}"]
                    return callable_(inner.inner)

                else:
                    callable_ = cls._double_simplification_dict[f"{cls.reciprocal_class.__name__} {inner._label}"]
                    return 1 / callable_(inner.inner)

            # not supporting stuff like asin(cos(x)) sorry.

        # 3. Evaluate floats immediately
        if isinstance(inner, Float):
            return Float(cls._func(inner.value))

        instance = super().__new__(cls)
        instance.inner = inner
        return instance

    def __init__(self, inner: Expr, *, skip_checks: bool = False):
        if skip_checks:
            self.inner = inner

        super().__post_init__()

    def _evalf(self, subs):
        inner = self.inner._evalf(subs)
        if isinstance(inner, Num):
            return Float(self._func(inner.value))
        return self.__class__(inner)

    def latex(self) -> str:
        from .latex import bracketfy

        inner = bracketfy(self.inner)
        if not self.is_inverse:
            return "\\" + self.func + " " + inner

        return "\\" + self.func + "^{-1} " + inner


class TrigFunctionNotInverse(TrigFunction, ABC):
    is_inverse = False
    _period = 2


class sin(TrigFunctionNotInverse):
    func = "sin"
    _func = math.sin
    _odd = True

    @classproperty
    def reciprocal_class(cls):
        return csc

    @classproperty
    def _special_values(cls):
        return {
            "0": Rat(0),
            "1/6": Rat(1, 2),
            "1/4": 1 / sqrt(2),
            "1/3": sqrt(3) / 2,
            "1/2": Rat(1),
            "2/3": sqrt(3) / 2,
            "3/4": 1 / sqrt(2),
            "5/6": Rat(1, 2),
            "1": Rat(0),
            "7/6": -Rat(1, 2),
            "5/4": -1 / sqrt(2),
            "4/3": -sqrt(3) / 2,
            "3/2": -Rat(1),
            "5/3": -sqrt(3) / 2,
            "7/4": -1 / sqrt(2),
            "11/6": -Rat(1, 2),
        }

    def diff(self, var) -> Expr:
        return cos(self.inner) * self.inner.diff(var)

    @cast
    def __new__(cls, inner: Expr) -> Expr:
        new = super().__new__(cls, inner)
        if not isinstance(new, sin):
            return new
        if isinstance(new.inner, Sum):
            for t in new.inner.terms:
                if t == pi or t == -pi:
                    return -sin(new.inner - t)

        # sin(n * pi) = 0
        if new.inner.has(Pi) and (new.inner / pi)._is_int:
            return Rat(0)

        return new


class cos(TrigFunctionNotInverse):
    func = "cos"
    _func = math.cos
    _even = True

    @classproperty
    def reciprocal_class(cls):
        return sec

    @classproperty
    def _special_values(cls):
        return {
            "0": Rat(1),
            "1/6": sqrt(3) / 2,
            "1/4": 1 / sqrt(2),
            "1/3": Rat(1, 2),
            "1/2": Rat(0),
            "2/3": -Rat(1, 2),
            "3/4": -1 / sqrt(2),
            "5/6": -sqrt(3) / 2,
            "1": Rat(-1),
            "7/6": -sqrt(3) / 2,
            "5/4": -1 / sqrt(2),
            "4/3": -Rat(1, 2),
            "3/2": -Rat(0),
            "5/3": Rat(1, 2),
            "7/4": 1 / sqrt(2),
            "11/6": sqrt(3) / 2,
        }

    def diff(self, var: Symbol) -> Expr:
        return -sin(self.inner) * self.inner.diff(var)

    @cast
    def __new__(cls, inner: Expr) -> "Expr":
        # bruh this is so complicated because doing cos(new, -inner) just sets inner as the original inner because of passing
        # the same args down to init.
        new = super().__new__(cls, inner)
        if not isinstance(new, cos):
            return new
        if isinstance(new.inner, Sum):
            for t in new.inner.terms:
                if t == pi or t == -pi:
                    return -cos(new.inner - t)
        return new


class tan(TrigFunctionNotInverse):
    func = "tan"
    _func = math.tan
    _period = 1
    _odd = True

    @classproperty
    def reciprocal_class(cls):
        return cot

    @classproperty
    def _special_values(cls):
        return {k: sin.special_values[k] / cos.special_values[k] for k in cls._SPECIAL_KEYS}

    def diff(self, var) -> Expr:
        return sec(self.inner) ** 2 * self.inner.diff(var)


class csc(TrigFunctionNotInverse):
    func = "csc"
    _func = lambda x: 1 / math.sin(x)
    reciprocal_class = sin
    _odd = True

    @classproperty
    def _special_values(cls):
        return {k: 1 / sin.special_values[k] for k in cls._SPECIAL_KEYS}

    def diff(self, var) -> Expr:
        return (1 / sin(self.inner)).diff(var)


class sec(TrigFunctionNotInverse):
    func = "sec"
    _func = lambda x: 1 / math.cos(x)
    reciprocal_class = cos
    _even = True

    @classproperty
    def _special_values(cls):
        return {k: 1 / cos.special_values[k] for k in cls._SPECIAL_KEYS}

    def diff(self, var) -> Expr:
        return sec(self.inner) * tan(self.inner) * self.inner.diff(var)


class cot(TrigFunctionNotInverse):
    reciprocal_class = tan
    func = "cot"
    _func = lambda x: 1 / math.tan(x)
    _period = 1
    _odd = True

    @classproperty
    def _special_values(cls):
        return {k: cos.special_values[k] / sin.special_values[k] for k in cls._SPECIAL_KEYS}

    def diff(self, var) -> Expr:
        return (1 / tan(self.inner)).diff(var)


class asin(TrigFunction):
    func = "sin"
    is_inverse = True
    _func = math.asin
    _odd = True

    def diff(self, var):
        return 1 / sqrt(1 - self.inner**2) * self.inner.diff(var)

    @classproperty
    def _special_values(cls):
        return {v: k for k, v in sin.special_values}


class acos(TrigFunction):
    func = "cos"
    is_inverse = True
    _func = math.acos

    def diff(self, var):
        return -1 / sqrt(1 - self.inner**2) * self.inner.diff(var)

    @classproperty
    def _special_values(cls):
        return {v: k for k, v in cos.special_values}


class atan(TrigFunction):
    func = "tan"
    is_inverse = True
    _func = math.atan
    _odd = True

    def __new__(cls, inner):
        # TODO: standardize special value for inverse trig functions
        if inner == 1:
            return pi / 4

        return super().__new__(cls, inner)

    def diff(self, var):
        return 1 / (1 + self.inner**2) * self.inner.diff(var)

    @classproperty
    def _special_values(cls):
        return {v: k for k, v in tan.special_values}


class Abs(SingleFunc):
    def __repr__(self) -> str:
        return "|" + repr(self.inner) + "|"

    @classproperty
    def _label(self) -> str:
        return "abs"

    def latex(self) -> str:
        from .latex import bracketfy

        return bracketfy(self.inner, bracket="||")

    @cast
    def __new__(cls, inner: Expr) -> Expr:
        if isinstance(inner, (E, Pi)):
            return inner
        if isinstance(inner, Num):
            return Const(abs(inner.value))
        if inner.is_subtraction:
            return Abs(-inner)
        if isinstance(inner, Prod):
            for t in inner.terms:
                if isinstance(inner, Num):
                    return Abs(inner / t) * Abs(t)

        if _is_strictly_positive(inner):
            return inner

        return super().__new__(cls)

    def diff(self, var) -> Expr:
        warnings.warn(f"Differentiation of {self} not implemented, so returning diff of {self.inner}.")
        return self.inner.diff(var)

    def _evalf(self, subs: Dict[str, Expr]) -> Expr:
        return Abs(self.inner._evalf(subs))


def _is_strictly_positive(expr: Expr) -> bool:
    if expr._strictly_positive:
        return True
    if isinstance(expr, (Prod, Sum)):
        return all(_is_strictly_positive(t) for t in expr.terms)
    if isinstance(expr, Power):
        # if base is positive then the power is always positive right
        return _is_strictly_positive(expr.base)
    if isinstance(expr, Num):
        return expr >= 0

    return False


def symbols(symbols: str) -> Union[Symbol, List[Symbol]]:
    """Creates symbols from a string of symbol names seperated by spaces."""
    symbols = [Symbol(name=s) for s in symbols.split(" ")]
    return symbols if len(symbols) > 1 else symbols[0]


@cast
def diff(expr: Expr, var: Optional[Symbol]) -> Expr:
    """Takes the derivative of expr relative to var. If expr has only one symbol in it, var doesn't need to be specified."""
    if not hasattr(expr, "diff"):
        raise NotImplementedError(f"Differentiation of {expr} not implemented")

    if var is None:
        symbols = expr.symbols()
        if len(symbols) != 1:
            return ValueError(f"Must provide variable of differentiation for {var}")
        var = symbols[0]

    return expr.diff(var)


def remove_const_factor(expr: Expr, include_factor=False) -> Expr:
    from .regex import get_anys

    if isinstance(expr, Prod):
        if not hasattr(expr, "_sans_const_cache"):
            sans_const = []
            const = []
            for t in expr.terms:
                if len(t.symbols()) > 0 or len(get_anys(t)) > 0:
                    sans_const.append(t)
                else:
                    const.append(t)

            expr._sans_const_cache = Prod(sans_const, skip_checks=True)
            expr._const_cache = Prod(const, skip_checks=True)
        if include_factor:
            return expr._sans_const_cache, expr._const_cache
        else:
            return expr._sans_const_cache

    if include_factor:
        return expr, Rat(1)
    else:
        return expr


def latex(expr: Expr) -> str:
    return expr.latex()


class Bound(NamedTuple):
    value: Expr
    inclusive: bool


@dataclass
class Piece:
    expr: Expr
    lower_bound: Bound
    upper_bound: Bound

    def __post_init__(self):
        if self.expr.has(Piecewise):
            raise ValueError("Piecewise functions cannot be nested.")

    def __repr__(self) -> str:
        return f"{self.lower_bound.value} <= x < {self.upper_bound.value}: {self.expr}"


class Piecewise(Expr):
    pieces: List[Piece]

    def __init__(self, *args: List[Tuple[Expr, Expr, Expr]], var: Symbol = None):
        pieces = []
        for arg in args:
            if isinstance(arg, Piece):
                pieces.append(arg)
                continue

            pieces.append(Piece(_cast(arg[0]), Bound(_cast(arg[1]), True), Bound(_cast(arg[2]), False)))
        self.pieces = pieces
        self.var = var

    def __repr__(self) -> str:
        return (
            "Piecewise("
            + ", ".join([f"{f.lower_bound.value} <= x < {f.upper_bound.value}: {f.expr}" for f in self.pieces])
            + ")"
        )
        # return "Piecewise(...)"

    def latex(self) -> str:
        return (
            "\\begin{cases} "
            + " \\\\ ".join([f"{f.lower_bound.value} \\leq x < {f.upper_bound.value}: {f.expr}" for f in self.pieces])
            + " \\end{cases}"
        )

    def _evalf(self, subs) -> "Piecewise":
        return Piecewise(*[Piece(p.expr._evalf(subs), p.lower_bound, p.upper_bound) for p in self.pieces])

    def children(self) -> List[Expr]:
        return [p.expr for p in self.pieces]

    def diff(self, var) -> "Piecewise":
        return Piecewise(*[Piece(p.expr.diff(var), p.lower_bound, p.upper_bound) for p in self.pieces])

    def subs(self, subs) -> "Piecewise":
        return Piecewise(*[Piece(p.expr.subs(subs), p.lower_bound, p.upper_bound) for p in self.pieces])

    def __add__(self, other: Expr) -> "Piecewise":
        return self._operate(other, fn=lambda x, y: x + y)

    def _operate(self, other: Expr, fn) -> "Piecewise":
        if not isinstance(other, Piecewise):
            return Piecewise(*[Piece(fn(p.expr, other), p.lower_bound, p.upper_bound) for p in self.pieces])

        if self.var != other.var:
            raise NotImplementedError
        if len(self.pieces) != len(other.pieces):
            raise NotImplementedError

        pieces = []
        for p1, p2 in zip(self.pieces, other.pieces):
            if p1.lower_bound != p2.lower_bound or p1.upper_bound != p2.upper_bound:
                raise NotImplementedError
            pieces.append(Piece(fn(p1.expr, p2.expr), p1.lower_bound, p1.upper_bound))

        return Piecewise(*pieces, var=self.var)

    def __radd__(self, other: Expr) -> "Piecewise":
        return self + other

    def __sub__(self, other: Expr) -> "Piecewise":
        return self._operate(other, fn=lambda x, y: x - y)

    def __rsub__(self, other: Expr) -> "Piecewise":
        return -self + other

    def __mul__(self, other: Expr) -> "Piecewise":
        return self._operate(other, fn=lambda x, y: x * y)

    def __rmul__(self, other: Expr) -> "Piecewise":
        return self * other

    def __truediv__(self, other: Expr) -> "Piecewise":
        return self._operate(other, fn=lambda x, y: x / y)

    def __rtruediv__(self, other: Expr) -> "Piecewise":
        return self**-1 * other

    def __neg__(self) -> "Piecewise":
        # neg each expr
        return Piecewise(*[Piece(-p.expr, p.lower_bound, p.upper_bound) for p in self.pieces])

    def __pow__(self, other: Expr) -> "Piecewise":
        return self._operate(other, fn=lambda x, y: x**y)

    def __rpow__(self, other: Expr) -> "Piecewise":
        return self._operate(other, fn=lambda x, y: y**x)

    def __eq__(self, other: Expr) -> bool:
        if not isinstance(other, Piecewise):
            return False
        if len(self.pieces) != len(other.pieces):
            return False
        for p1, p2 in zip(self.pieces, other.pieces):
            if p1.lower_bound != p2.lower_bound or p1.upper_bound != p2.upper_bound or p1.expr != p2.expr:
                return False
        return True

    def __ne__(self, other: Expr) -> bool:
        return not self == other
