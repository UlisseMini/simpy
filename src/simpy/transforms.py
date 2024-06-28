# structures for integration w transforms

from abc import ABC, abstractmethod
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, Dict, List, Literal, Optional, Tuple, Type, Union

import numpy as np

from .expr import (
    Expr,
    Num,
    Power,
    Prod,
    Rat,
    Sum,
    Symbol,
    TrigFunction,
    TrigFunctionNotInverse,
    asin,
    atan,
    cos,
    cot,
    csc,
    e,
    exp,
    log,
    remove_const_factor,
    sec,
    sin,
    sqrt,
    symbols,
    tan,
)
from .integral_table import check_integral_table
from .linalg import invert
from .polynomial import Polynomial, is_polynomial, polynomial_to_expr, rid_ending_zeros, to_const_polynomial
from .regex import count, general_contains, general_count, replace, replace_class, replace_factory
from .simplify import pythagorean_simplification
from .simplify.product_to_sum import product_to_sum_unit
from .utils import ExprFn, eq_with_var, random_id

Number_ = Union[Fraction, int]


@dataclass
class Node:
    """A node in the integration nodetree.

    expr: the expression to integrate
    var: the variable that THIS EXPR is integrated by
    transform: the transform that led to this node
    parent: the parent node. None for root node only.
    type: AND, OR, UNSET, SOLUTION, FAILURE
        - failure = can't proceed forward. not a single transform matches.
    solution: only for SOLUTION nodes (& their parents when we go backwards)
    is_filler: fillers are ones where the expr is not *really* the expression to integrate, and just a copy of the parents. the node exists to store info other than the expr.


    `_children` is private. do not modify it directly; only use the `add_child` or `add_children` methods. you can read from the
    `children` property.
    """

    expr: Expr
    var: Symbol
    transform: Optional["Transform"] = None
    parent: Optional["Node"] = None
    type: Literal["AND", "OR", "UNSET", "SOLUTION", "FAILURE"] = "UNSET"
    solution: Optional[Expr] = None
    is_filler: bool = False
    _children: Optional[List["Node"]] = None

    def __post_init__(self):
        if self._children is None:
            self._children = []
        else:
            raise ValueError

    @property
    def children(self) -> List["Node"]:
        return self._children

    def add_child(self, child: "Node") -> None:
        if not child.is_filler:
            parents = _parents(self)
            if any(eq_with_var((p.expr, p.var), (child.expr, child.var)) for p in parents):
                return
        self.children.append(child)

    def add_children(self, children: List["Node"]) -> None:
        for c in children:
            self.add_child(c)

    def __repr__(self):
        num_children = len(self.children) if self.children else 0
        return f"Node({self.expr.__repr__()}, {self.var}, transform {self.transform.__class__.__name__}, {num_children} children, {self.type})"

    @property
    def leaves(self) -> List["Node"]:
        # Returns the leaves of the tree (all nodes without children)
        if not self.children:
            return [self]

        return [leaf for child in self.children for leaf in child.leaves]

    @property
    def unfinished_leaves(self) -> List["Node"]:
        # Leaves to work on :)
        return [leaf for leaf in self.leaves if not leaf.is_finished]

    @property
    def root(self) -> "Node":
        if not self.parent:
            return self
        return self.parent.root

    @property
    def distance_from_root(self) -> int:
        if not self.parent:
            return 0
        return 1 + self.parent.distance_from_root

    @property
    def is_stale(self) -> bool:
        """Stale = unfinished node that is no longer needed"""
        return not self.is_finished and any(parent.is_finished for parent in _parents(self))

    @property
    def is_solved(self) -> bool:
        # Returns True if all leaves WITH AND NODES are solved and all OR nodes are solved
        # if limit is reached, return False
        if self.type == "SOLUTION":
            return True

        if not self.children:
            return False

        if self.type == "AND" or self.type == "UNSET":
            # UNSET should only have one child.
            return all(child.is_solved for child in self.children)

        if self.type == "OR":
            return any(child.is_solved for child in self.children)

    @property
    def is_failed(self) -> bool:
        # Is not solveable if one branch is not solveable and it has no "OR"s

        if self.type == "FAILURE":
            return True

        if not self.children:
            # if it has no children and it's not "FAILURE", it means this node is an unfinished leaf (or a solution).
            return False

        if self.type == "OR":
            return all(child.is_failed for child in self.children)

        return any(child.is_failed for child in self.children)

    @property
    def is_finished(self) -> bool:
        return self.is_solved or self.is_failed

    @property
    def unfinished_children(self) -> List["Node"]:
        if not self.children:
            return []
        return [child for child in self.children if not child.is_finished]

    @property
    def child(self) -> Optional["Node"]:
        # this is only here because when debugging im lazy and want to type node.child instead of node.children[0]
        if not self.children:
            return None
        return self.children[0]


class Transform(ABC):
    "An integral transform -- base class"
    # forward and backward modify the nodetree directly. check is a pure function

    def __init__(self):
        pass

    @abstractmethod
    def forward(self, node: Node) -> None:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def backward(self, node: Node) -> None:
        """Subclasses must call super().backward(...) before doing its own logic."""
        if not node.solution:
            raise ValueError("Node has no solution")

    @abstractmethod
    def check(self, node: Node) -> bool:
        # sanity check: if the nestings get strictly bigger for the past 5 levels, quit.
        # parents = _parents(node)
        # parents = [p for p in parents if p.transform.__class__ in HEURISTICS]
        # if len(parents) < 5:
        #     return True
        # nests = [nesting(p.expr) for p in parents[:5]]
        # breakpoint()
        # if all(nests[i+1] < nests[i] for i in range(4)):
        #     return False
        # return True

        return True


class USub(Transform, ABC):
    """Base class for u-substituion transforms."""

    _u: Expr = None

    def backward(self, node: Node) -> None:
        super().backward(node)
        node.parent.solution = replace(node.solution, node.var, self._u)


class SafeTransform(Transform, ABC):
    pass


class PullConstant(SafeTransform):
    """Pulls out a constant from the integral. Linearity."""

    _constant: Expr = None
    _non_constant_part: Expr = None

    def check(self, node: Node) -> bool:
        if super().check(node) is False:
            return False

        expr = node.expr
        var = node.var
        if isinstance(expr, Prod):
            # if there is a constant, pull it out
            # # or if there is a symbol that's not the variable, pull it out
            for i, term in enumerate(expr.terms):
                if var not in term.symbols():
                    self._constant = term
                    self._non_constant_part = Prod(expr.terms[:i] + expr.terms[i + 1 :])
                    return True

        return False

    def forward(self, node: Node):
        node.add_child(Node(self._non_constant_part, node.var, self, node))

    def backward(self, node: Node) -> None:
        super().backward(node)
        node.parent.solution = self._constant * node.solution


class PolynomialDivision(SafeTransform):
    """Divides a polynomial by another polynomial."""

    _numerator: Polynomial = None
    _denominator: Polynomial = None

    def check(self, node: Node) -> bool:
        if super().check(node) is False:
            return False

        # This is so messy we can honestly just do catching in the `to_polynomial`
        expr = node.expr
        # currently we don't support division of polynomials with multiple variables
        if expr.symbols() != [node.var]:
            return False
        if not isinstance(expr, Prod):
            return False

        ## Make sure numerator and denominator are both polynomials
        numerator, denominator = expr.numerator_denominator
        if denominator == 1:
            return False
        try:
            numerator_list = to_const_polynomial(numerator, node.var)
            denominator_list = to_const_polynomial(denominator, node.var)
        except AssertionError:
            return False

        # You can divide if they're same order
        if len(numerator_list) < len(denominator_list):
            return False

        self._numerator = numerator_list
        self._denominator = denominator_list
        return True

    def forward(self, node: Node):
        var = node.var
        quotient = [Rat(0)] * (len(self._numerator) - len(self._denominator) + 1)
        quotient = np.array(quotient)

        while self._numerator.size >= self._denominator.size:
            quotient_degree = len(self._numerator) - len(self._denominator)
            quotient_coeff = self._numerator[-1] / self._denominator[-1]
            quotient[quotient_degree] = quotient_coeff
            self._numerator -= np.concatenate(([Rat(0)] * quotient_degree, self._denominator * quotient_coeff))
            self._numerator = rid_ending_zeros(self._numerator)

        remainder = polynomial_to_expr(self._numerator, var) / polynomial_to_expr(self._denominator, var)
        quotient_expr = polynomial_to_expr(quotient, var)
        answer = quotient_expr + remainder
        node.add_child(Node(answer, var, self, node))

    def backward(self, node: Node) -> None:
        super().backward(node)
        node.parent.solution = node.solution


class Expand(SafeTransform):
    """Expands the expression"""

    def forward(self, node: Node):
        node.add_child(Node(node.expr.expand(), node.var, self, node))

    def check(self, node: Node) -> bool:
        if super().check(node) is False:
            return False

        # If the last heuristic was completing the square,
        # Please give it a break.
        t = _get_last_heuristic_transform(node)
        if isinstance(t, CompleteTheSquare):
            return False

        return node.expr.expandable()

    def backward(self, node: Node) -> None:
        super().backward(node)
        node.parent.solution = node.solution


class Additivity(SafeTransform):
    """Additivity of integrals"""

    def forward(self, node: Node):
        node.type = "AND"
        node.add_children([Node(e, node.var, self, node) for e in node.expr.terms])

    def check(self, node: Node) -> bool:
        if super().check(node) is False:
            return False

        return isinstance(node.expr, Sum)

    def backward(self, node: Node) -> None:
        super().backward(node)

        # For this to work, we must have a solution for each sibling.
        if not all(child.solution for child in node.parent.children):
            raise ValueError(f"Additivity backward for {node} failed")

        node.parent.solution = Sum([child.solution for child in node.parent.children])


def _get_last_heuristic_transform(node: Node, tup=(PullConstant, Additivity)):
    """Returns the last heuristic transform that was used on the node.
    tup: tuple of transform classes to exclude from the search.
    """
    if isinstance(node.transform, tup):
        # We'll let polynomial division go because it changes things sufficiently that
        # we actually sorta make progress towards the integral.
        # PullConstant and Additivity are like fake, they dont make any substantial changes.
        # Expand is also like, idk, if we do A and then expand we dont rlly wanna do A again.

        # Alternatively, we could just make sure that the last transform didnt have the same
        # key. (but no the lecture example has B tan then polydiv then C tan)

        # Idk this thing rn is a lil messy and there might be a better way to do it.

        # 05/13/2024: no more expand bc it's not even a "safe transform" anymore lwk.
        return _get_last_heuristic_transform(node.parent, tup)
    return node.transform


# Let's just add all the transforms we've used for now.
# and we will make this shit good and generalized later.
class TrigUSub2(USub):
    """
    u-sub of a trig function
    this is the weird u-sub where if u=sinx, dx != du/cosx but dx = du/sqrt(1-u^2)
    ex: integral of f(tanx) -> integral of f(u) / 1 + y^2, sub u = tanx
    -> dx = du/(1+x^2)
    """

    _key: str = None
    # {label: trigfn class, derivative of inverse trigfunction}
    _table: Dict[str, Tuple[ExprFn, ExprFn]] = {
        "sin": (sin, lambda var: 1 / sqrt(1 - var**2)),  # Asin(x).diff(x)
        "cos": (cos, lambda var: -1 / sqrt(1 - var**2)),
        "tan": (tan, lambda var: 1 / (1 + var**2)),
    }

    def forward(self, node: Node):
        intermediate = generate_intermediate_var()
        expr = node.expr
        # y = tanx
        cls, dy_dx = self._table[self._key]
        new_integrand = replace(expr, cls(node.var), intermediate) * dy_dx(intermediate)
        new_node = Node(new_integrand, intermediate, self, node)
        node.add_child(new_node)

        self._u = cls(node.var)

    def check(self, node: Node) -> bool:
        if super().check(node) is False:
            return False

        # Since B and C essentially undo each other, we want to make sure that the last
        # heuristic transform wasn't C.

        t = _get_last_heuristic_transform(node)
        if isinstance(t, InverseTrigUSub):
            return False

        for k, v in self._table.items():
            cls, dy_dx = v
            count_ = count(node.expr, cls(node.var))
            if count_ >= 1 and count_ == count(node.expr, node.var):
                self._key = k
                return True

        return False


class RewriteTrig(Transform):
    """Rewrites trig functions in terms of (sin, cos), (tan, sec), and (cot, csc)"""

    def forward(self, node: Node):
        expr = node.expr
        r1 = replace_class(
            expr,
            [tan, csc, cot, sec],
            [
                lambda x: sin(x) / cos(x),
                lambda x: 1 / sin(x),
                lambda x: cos(x) / sin(x),
                lambda x: 1 / cos(x),
            ],
        )
        r2 = replace_class(
            expr,
            [sin, cos, cot, sec],
            [
                lambda x: 1 / csc(x),
                lambda x: 1 / tan(x) / csc(x),
                lambda x: 1 / tan(x),
                lambda x: tan(x) * csc(x),
            ],
        )
        r3 = replace_class(
            expr,
            [sin, cos, tan, csc],
            [
                lambda x: 1 / cot(x) / sec(x),
                lambda x: 1 / sec(x),
                lambda x: 1 / cot(x),
                lambda x: cot(x) * sec(x),
            ],
        )

        new_exprs = [r1, r2, r3]
        node.add_children([Node(option, node.var, self, node) for option in new_exprs if option != expr])
        node.type = "OR"

    def check(self, node: Node) -> bool:
        if super().check(node) is False:
            return False

        # make sure that this node didn't get here by this transform
        # lots of time, rewriting trig would make it naturally expand.
        # without this including expand, csc^2 did not get solved depth-first.
        t = _get_last_heuristic_transform(node, (Additivity, PullConstant, Expand))
        if isinstance(t, RewriteTrig):
            return False

        expr = node.expr
        return expr.has(TrigFunctionNotInverse)

    def backward(self, node: Node) -> None:
        super().backward(node)
        node.parent.solution = node.solution


class InverseTrigUSub(USub):
    """Does the sub of u = asin(x), u = atan(x)"""

    _key = None

    # {label: class, search query, dy_dx, variable_change}
    _table: Dict[str, Tuple[ExprFn, ExprFn, ExprFn, ExprFn]] = {
        "sin": (sin, lambda symbol: 1 - symbol**2, lambda var: cos(var), asin),
        "tan": (tan, lambda symbol: 1 + symbol**2, lambda var: sec(var) ** 2, atan),
    }

    def forward(self, node: Node):
        intermediate = generate_intermediate_var()
        cls, q, dy_dx, var_change = self._table[self._key]
        dy_dx = dy_dx(intermediate)
        new_integrand = replace(node.expr, node.var, cls(intermediate)) * dy_dx
        new_node = Node(new_integrand, intermediate, self, node)
        node.add_child(new_node)
        self._u = var_change(node.var)

        # I feel like you already know that it's gonna be pythagorean-simplified so why not just tack
        # that on right now
        second_transform = Simplify()
        if second_transform.check(new_node):
            second_transform.forward(new_node)

    def check(self, node: Node) -> bool:
        if super().check(node) is False:
            return False

        t = _get_last_heuristic_transform(node)
        if isinstance(t, TrigUSub2):
            # If it just went through B, C is guaranteed to have a match.
            # going through C will just undo B.
            return False

        for k, v in self._table.items():
            query = v[1](node.var)
            if count(node.expr, query) > 0 or count(node.expr, (query * -1)) > 0:
                self._key = k
                return True

        return False


class PolynomialUSub(USub):
    """check that x^n-1 is a term and every other instance of x is x^n
    you're gonna replace u=x^n
    ex: x/sqrt(1-x^2)
    """

    # _u = x^n

    def check(self, node: Node) -> bool:
        if super().check(node) is False:
            return False

        if not isinstance(node.expr, Prod):
            return False

        rest: Expr = None
        n = None
        for i, term in enumerate(node.expr.terms):
            # yes you can assume it's an expanded simplified product. so no terms
            # are Prod or Sum.
            # so x^n-1 must be exactly a term with no fluff. :)
            if isinstance(term, Power) and term.base == node.var and not term.exponent.contains(node.var):
                n = term.exponent + 1
                rest = Prod(node.expr.terms[:i] + node.expr.terms[i + 1 :])
                break

            if term == node.var:
                n = 2
                rest = Prod(node.expr.terms[:i] + node.expr.terms[i + 1 :])
                break

        if n is None:
            return False
        if n == 0:
            # How are you gonna sub u = x^0 = 1, du = 0 dx
            return False

        self._u = Power(node.var, n)  # x^n
        count_ = count(node.expr, self._u)
        return count_ > 0 and count_ == count(rest, node.var)

    def forward(self, node: Node) -> None:
        intermediate = generate_intermediate_var()
        dx_dy = self._u.diff(node.var)
        new_integrand = replace(node.expr, self._u, intermediate) / dx_dy
        new_integrand = new_integrand
        node.add_child(Node(new_integrand, intermediate, self, node))


class LinearUSub(USub):
    """u-substitution for smtn like f(ax+b)
    u = ax+b
    du = a dx
    \int f(ax+b) dx = 1/a \int f(u) du
    """

    # _u writes u in terms of x
    _inverse_var_change: ExprFn = None  # this will write x in terms of u

    def check(self, node: Node) -> bool:
        if super().check(node) is False:
            return False

        if count(node.expr, node.var) < 1:  # just to cover our bases bc the later thing assume it appears >=1
            return False

        def _is_a_linear_sum_or_prod(
            expr: Expr,
        ) -> Optional[Tuple[Expr, Optional[ExprFn]]]:
            """Returns what u should be in terms of x. & None if it's not."""
            if isinstance(expr, Sum):
                if all(not c.contains(node.var) or not (c / node.var).contains(node.var) for c in expr.terms):
                    return expr, None
            if isinstance(expr, Prod):
                # Is a constant multiple of var
                if not (expr / node.var).contains(node.var):
                    return expr, None
                # or is constant multiple of var**const
                # ex: if we have x**2/36, we want to rewrite it as (x/6)**2 and sub u=x/6
                # if we have sth like (x+3)**2/36 we will just let the sum catch it and do 2 linearusubs in sequence :shrug:
                if expr.is_subtraction:
                    expr = -expr
                xyz = remove_const_factor(expr)
                is_const_multiple_of_power = (
                    isinstance(xyz, Power) and xyz.base == node.var and not xyz.exponent.contains(node.var)
                )
                if not is_const_multiple_of_power:
                    return None
                coeff = expr / xyz
                if coeff == Rat(1):
                    return None
                coeff_abs = abs(coeff) if isinstance(coeff, Num) else coeff
                inner_coeff = Power(coeff_abs, 1 / xyz.exponent)
                return inner_coeff * node.var, lambda u: u / inner_coeff
            return None

        def _check(e: Expr) -> bool:
            if not e.contains(node.var):
                return True
            result = _is_a_linear_sum_or_prod(e)
            if result is not None:
                u, u_inverse = result
                if self._u is not None:
                    return u == self._u
                self._u = u

                # If u_inverse exists, set it.
                # it must be the same as any prev u_inverse because the same u implies the same
                # u_inverse
                if u_inverse is not None:
                    self._inverse_var_change = u_inverse
                return True
            if not e.children():
                # This must mean that we contain the var and it has no children, which means that we are the var.
                # have to do this because all([]) = True. Covering my bases.
                return False
            else:
                return all(_check(child) for child in e.children())

        return _check(node.expr)

    def forward(self, node: Node) -> None:
        intermediate = generate_intermediate_var()
        du_dx = self._u.diff(node.var)

        # We have to account for the case where self._u doesn't appear directly
        # in the integrand.
        if self._inverse_var_change is not None:
            new = self._inverse_var_change(intermediate)
            new_integrand = replace(node.expr, node.var, new)
        else:
            new_integrand = replace(node.expr, self._u, intermediate)
        new_integrand /= du_dx
        new_integrand = new_integrand
        node.add_child(Node(new_integrand, intermediate, self, node))


class ProductToSum(Transform):
    """product to sum identities for sin & cos

    only works if the node.expr is a product of sin and cos
    doesn't check nested things.
    since when have we solved an integral that has e^{sum} where you need to apply pts on the sum anyways
    """

    def check(self, node: Node) -> bool:
        if super().check(node) is False:
            return False
        return True

    def forward(self, node: Node) -> None:
        new_integrand = product_to_sum_unit(node.expr)
        if new_integrand:
            node.add_child(Node(new_integrand, node.var, self, node))

    def backward(self, node: Node) -> None:
        super().backward(node)
        node.parent.solution = node.solution


def _parents(node: Node) -> List[Node]:
    """Returns the node and all its parents as we ascend the tree"""
    if node.parent is None:
        return [node]
    return [node] + _parents(node.parent)


class ByParts(Transform):
    """Integration by parts"""

    _stuff: List[Tuple[Expr, Expr, Expr, Expr]] = None

    @staticmethod
    def _integrate_dv_check(dv, var) -> Optional[Expr]:
        from .integration import Integration

        return Integration.integrate_without_heuristics(dv, var)

    @staticmethod
    def _get_all_byparts_parents(node: Node) -> List[Node]:
        return [p for p in _parents(node) if isinstance(p.transform, ByParts)]

    @staticmethod
    def _get_last_byparts_parent(node: Node) -> Optional[Node]:
        # no heuristics have been applied yet
        for p in _parents(node):
            if isinstance(p.transform, ByParts):
                return p
            if not isinstance(p.transform, (PullConstant, Additivity)):
                return
        return

    @staticmethod
    def _parents_exprs_check(node: Node, du: Expr, v: Expr) -> bool:
        # if -u'v = node.expr, it means means you get 0 * integral(node.expr) = uv
        # which is invalid
        integrand2 = du * v * -1
        factor = integrand2 / node.expr
        if factor == 1:
            return False

        # check for more layers above -- if any of the integrands is the same, factor = 1 and it's a nogo.
        # honestly this is a bit sad; why don't we just always check that a node doesn't appear in its parents?
        byparts_parents = ByParts._get_all_byparts_parents(node)
        old_byparts_integrands = [node.expr for node in byparts_parents]
        if integrand2 in old_byparts_integrands:
            return False
        return True

    def __init__(self):
        self._stuff = []

    def check(self, node: Node) -> bool:
        if super().check(node) is False:
            return False

        if not isinstance(node.expr, Prod):
            if (
                isinstance(node.expr, log) or isinstance(node.expr, TrigFunction) and node.expr.is_inverse
            ) and node.expr.inner == node.var:  # Special case for Log, ArcSin, etc.
                # TODO universalize it more?? how to make it more universal without making inf loops?
                dv = 1
                v = node.var
                u = node.expr
                du = u.diff(node.var)
                self._stuff.append((u, du, v, dv))
                return True
            return False
        if not len(node.expr.terms) == 2:
            return False

        def _check(u: Expr, dv: Expr) -> bool:
            if is_polynomial(dv, node.var):
                return False
            du = u.diff(node.var)
            v = self._integrate_dv_check(dv, node.var)
            if v is None:
                return False

            if not self._parents_exprs_check(node, du, v):
                return False

            self._stuff.append((u, du, v, dv))
            return True

        a, b = node.expr.terms
        return _check(a, b) or _check(b, a)

    @staticmethod
    def _get_first_factor(parent_byparts: Node, node: Node) -> Expr:
        if parent_byparts.children[1] == node:
            return Rat(1)
        elif isinstance(node.transform, PullConstant) and parent_byparts.children[1].child == node:
            return node.transform._constant
        raise ValueError

    def forward(self, node: Node) -> None:
        # This is tricky bc you have 2 layers of children here.
        for u, du, v, dv in self._stuff:
            child1 = u * v
            integrand2 = du * v * -1

            tr = ByParts()
            tr._stuff = [(u, du, v, dv)]

            ### special case: expr is the same as current new
            # and you can jump directly to the solution wheeee
            factor = integrand2 / node.expr
            if not factor.contains(node.var):
                solution = child1 / (1 - factor)
                node.add_child(
                    Node(
                        node.expr,
                        node.var,
                        tr,
                        node,
                        type="SOLUTION",
                        solution=solution,
                        is_filler=True,
                    )
                )
                node.type = "OR"
                return
            ###

            ### special case: when parent is same as you 2 layers above
            # this isnt the most elegant but it works lol
            parent_byparts = self._get_last_byparts_parent(node)
            if parent_byparts:
                second_factor = integrand2 / parent_byparts.expr
                if not second_factor.contains(node.var):
                    first_factor = self._get_first_factor(parent_byparts, node)
                    factor = first_factor * second_factor
                    other_uv = parent_byparts.child
                    other_uv.solution /= 1 - factor  # mutating is sus
                    solution = child1 / (1 - factor)  # going back will mul it w second factor by default.
                    node.add_child(
                        Node(
                            node.expr,
                            node.var,
                            tr,
                            node,
                            type="SOLUTION",
                            solution=solution,
                            is_filler=True,
                        )
                    )
                    node.type = "OR"
                    assert parent_byparts.is_solved
                    return
            ###

            funky_node = Node(node.expr, node.var, tr, node, type="AND", is_filler=True)
            funky_node.add_children(
                [
                    Node(
                        node.expr,
                        node.var,
                        Additivity(),
                        funky_node,
                        type="SOLUTION",
                        solution=child1,
                        is_filler=True,
                    ),
                    Node(integrand2, node.var, Additivity(), funky_node),
                ]
            )
            node.add_child(funky_node)

    def backward(self, node: Node) -> None:
        super().backward(node)
        node.parent.solution = node.solution


class PartialFractions(Transform):
    """Integration using partial fractions"""

    _new_integrand: Expr = None

    def check(self, node: Node) -> bool:
        if super().check(node) is False:
            return False

        # First, make sure that this is a fraction
        if not isinstance(node.expr, Prod):
            return False
        num, denom = node.expr.numerator_denominator
        if denom == Rat(1):
            return False

        # and that both numerator and denominator are polynomials
        try:
            numerator_list = to_const_polynomial(num, node.var)
            denominator_list = to_const_polynomial(denom, node.var)
        except AssertionError:
            return False

        # numerator has to be a smaller order than the denom
        if len(numerator_list) >= len(denominator_list):
            return False

        # The denominator has to be a product
        if not isinstance(denom, (Prod, Sum)):
            return False
        if isinstance(denom, Sum):
            new = denom.factor()
            if not isinstance(new, Prod):
                return False
            denom = new

        # ok im stupid so im gonna only do the case for 2 factors for now
        # shouldnt be hard to generalize
        if len(denom.terms) != 2:
            return False

        d1, d2 = denom.terms

        # Make sure that it's not the case that one of the denominator factors is just a constant.
        if not (d1.contains(node.var) and d2.contains(node.var)):
            return False

        d1_list = to_const_polynomial(d1, node.var)
        d2_list = to_const_polynomial(d2, node.var)

        matrix = np.array([d2_list, d1_list]).T
        inv = invert(matrix)
        if inv is None:
            return False
        if numerator_list.size == 1:
            numerator_list = np.array([numerator_list[0], 0])
        ans = inv @ numerator_list
        self._new_integrand = ans[0] / d1 + ans[1] / d2
        return True

    def forward(self, node: Node) -> None:
        node.add_child(Node(self._new_integrand, node.var, self, node))

    def backward(self, node: Node) -> None:
        super().backward(node)
        node.parent.solution = node.solution


class GenericUSub(USub):
    """Generic u-substitution"""

    def check(self, node: Node) -> bool:
        if super().check(node) is False:
            return False

        if not isinstance(node.expr, Prod):
            return False
        for i, term in enumerate(node.expr.terms):
            integral = check_integral_table(term, node.var)
            if integral is None:
                continue
            integral = remove_const_factor(integral)
            # assume term appears only once in integrand
            # because node.expr is simplified
            rest = Prod(node.expr.terms[:i] + node.expr.terms[i + 1 :], skip_checks=True)
            if count(rest, integral) == count(rest, node.var) / count(integral, node.var):
                self._u = integral
                return True

        return False

    def forward(self, node: Node) -> None:
        intermediate = generate_intermediate_var()
        du_dx = self._u.diff(node.var)
        new_integrand = replace((node.expr / du_dx), self._u, intermediate)
        node.add_child(Node(new_integrand, intermediate, self, node))


class ExponentialUSub(USub):
    def check(self, node: Node) -> bool:
        if super().check(node) is False:
            return False

        if not isinstance(node.expr, Prod):
            return False

        for i, term in enumerate(node.expr.terms):
            if term == exp(node.var):
                # now make sure all other occurances of x are in the exponent of an exp
                # like it can be exp(2x) or smtn where it can be rewritten as exp(x)^2

                rest = Prod(node.expr.terms[:i] + node.expr.terms[i + 1 :], skip_checks=True)
                if count(rest, node.var) == general_count(
                    rest,
                    lambda x: isinstance(x, Power) and x.base == e and not (x.exponent / node.var).contains(node.var),
                ):
                    self._u = exp(node.var)
                    self._rest = rest
                    return True

        return False

    def forward(self, node: Node) -> None:
        intermediate = generate_intermediate_var()
        x_in_terms_of_u = log(intermediate)
        new_integrand = replace(self._rest, node.var, x_in_terms_of_u)
        node.add_child(Node(new_integrand, intermediate, self, node))


class CompleteTheSquare(Transform):
    """Integration via completing the square"""

    def check(self, node: Node) -> bool:
        if super().check(node) is False:
            return False

        # Completing the square is only useful if you can do InverseTrigUSub after it.
        # So we only look for these two cases:
        # 1 / quadratic
        # 1 / sqrt(quadratic)
        def condition(expr: Expr) -> bool:
            # 1 / xyz should be epxressed as a power so i omit prod check
            if not isinstance(expr, Power):
                return False
            if expr.exponent != Fraction(-1, 2) and expr.exponent != -1:
                return False
            try:
                poly = to_const_polynomial(expr.base, node.var)
            except AssertionError:
                return False

            # hmm completing the square could work on non-quadratics in some cases no?
            # but I'll just limit it to quadratics for now
            return poly.size == 3 and poly[1] != 0
            # poly[1] = 0 implies that there's no bx term
            # which means that there's no square to complete.
            # the result will just be the same as the original.

        return general_contains(node.expr, condition)

    def forward(self, node: Node) -> None:
        # replace all quadratics with their completed-the-square form
        def condition(expr: Expr) -> bool:
            try:
                poly = to_const_polynomial(expr, node.var)
            except AssertionError:
                return False
            return poly.size == 3

        def perform(expr: Expr) -> Expr:
            poly = to_const_polynomial(expr, node.var)
            norm_factor = poly[-1]
            normalized = poly / norm_factor
            const = normalized[1] / 2
            diff = normalized[0] - const**2
            new_expr = (((node.var + const) / sqrt(diff)) ** 2 + 1) * diff * norm_factor

            return new_expr

        new_expr = replace_factory(condition, perform)(node.expr)
        node.add_child(Node(new_expr, node.var, self, node))

    def backward(self, node: Node) -> None:
        super().backward(node)
        node.parent.solution = node.solution


class RewritePythagorean(Transform):
    """
    sin(x)^(2n+1) -> sin(x) (1-cos^2(x))^n

    ykw idk if this transform has ever helped anyone when it's not performed at the outer level of nesting.
    """

    @staticmethod
    def condition(expr: Expr) -> bool:
        return (
            isinstance(expr, Power)
            and isinstance(expr.base, (sin, cos))
            and isinstance(expr.exponent, Rat)
            and expr.exponent > 1
            and expr.exponent.value.denominator == 1
            and expr.exponent % 2 == 1
        )

    @staticmethod
    def perform(expr: Power) -> bool:
        """assumes RewritePythagorean.condition(expr) == True"""
        b, x = expr.base, expr.exponent
        n = (x - 1) / 2
        d = {"sin": cos, "cos": sin}
        return (1 - d[b.func](b.inner) ** 2) ** n * b.__class__(b.inner)

    def check(self, node: Node) -> bool:
        if super().check(node) is False:
            return False

        return True

    def forward(self, node: Node) -> bool:
        is_hit = {"is_hit": False}

        def _replace_factory(c, p, e):
            # chill replace factory that only checks outer layer of nesting.
            def _replace(e) -> Expr:
                if c(e):
                    is_hit["is_hit"] = True
                    return p(e)
                if isinstance(e, Prod):
                    return Prod([_replace(term) for term in e.terms])
                else:
                    return e

            return _replace(e)

        new_expr = _replace_factory(self.condition, self.perform, node.expr)
        if not is_hit["is_hit"]:
            return
        node.add_child(Node(new_expr, node.var, self, node))

    def backward(self, node: Node) -> None:
        super().backward(node)
        node.parent.solution = node.solution


class Simplify(Transform):
    """Simplifies using pythagorean trigonometric identities if possible"""

    def check(self, node: Node) -> bool:
        if super().check(node) is False:
            return False

        # t = _get_last_heuristic_transform(node)
        # # The point of these 2 transforms is to rewrite the expr in a not necessarily 'simpler' form.
        # if isinstance(t, (RewritePythagorean, RewriteTrig, Simplify)):
        #     return False
        ans, success = pythagorean_simplification(node.expr, verbose=True)
        if success:
            self._ans = ans
        return success

    def forward(self, node: Node) -> None:
        node.add_child(Node(self._ans, node.var, self, node))

    def backward(self, node: Node) -> None:
        super().backward(node)
        node.parent.solution = node.solution


# Leave RewriteTrig, InverseTrigUSub near the end bc they are deprioritized
# and more fucky
# Do not include Simplify because it only is needed after InverseTrigUSub, and we just manually
# trigger it.
# Not including CompoundAngle because no integral has actually been solved with it (backwards is never called)
# Not including SinUSub because it's a strict subset of GenericUSub
HEURISTICS: List[Type[Transform]] = [
    PolynomialUSub,
    ProductToSum,
    TrigUSub2,
    ByParts,
    RewriteTrig,
    RewritePythagorean,
    InverseTrigUSub,
    CompleteTheSquare,
    ExponentialUSub,
    GenericUSub,
]
SAFE_TRANSFORMS: List[Type[Transform]] = [
    Additivity,
    PullConstant,
    PartialFractions,
    PolynomialDivision,
    Expand,  # expanding a fraction is not safe bc it destroys partialfractions. but if you put it after polynomial division & partial fractions, it doesn't cause any issues. more robust solution is to refactor & put expanding a fraction seperately as a heuristic transform, but idt this is necessary right now.
    LinearUSub,
]


def generate_intermediate_var() -> Symbol:
    return symbols(f"u_{random_id(10)}")
