# structures for integration w transforms

from typing import Any, Optional

from main import *

# feels like var should be global of some sorts for each integration call. being passed down each by each feels sad.
# idk im overthinking it.


@dataclass
class Node:
    expr: Expr
    var: Symbol  # variable that THIS EXPR is integrated by.
    transform: Optional["Transform"] = None  # the transform that led to this node
    parent: Optional["Node"] = None  # None for root node only
    children: Optional[List["Node"]] = (
        None  # smtn smtn setting it to [] by default causes errors
    )
    type: Literal["AND", "OR", "UNSET", "SOLUTION", "FAILURE"] = "UNSET"
    solution: Optional[Expr] = (
        None  # only for SOLUTION nodes (& their parents when we go backwards)
    )
    # failure = can't proceed forward.

    def __post_init__(self):
        self.expr = self.expr.simplify()

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
    def is_solved(self) -> bool:
        # Returns True if all leaves WITH AND NODES are solved and all OR nodes are solved
        # if limit is reached, return False
        if self.type == "SOLUTION":
            return True

        if not self.children:
            return False

        if self.type == "AND" or self.type == "UNSET":
            # UNSET should only have one child.
            return all([child.is_solved for child in self.children])

        if self.type == "OR":
            return any([child.is_solved for child in self.children])

    @property
    def is_failed(self) -> bool:
        # TODO
        # Is not solveable if one branch is not solveable and it has no "OR"s

        if self.type == "FAILURE":
            return True

        if not self.children:
            # if it has no children and it's not "FAILURE", it means this node is an unfinished leaf (or a solution).
            return False

        if self.type == "OR":
            return all([child.is_failed for child in self.children])

        return any([child.is_failed for child in self.children])

    @property
    def is_finished(self) -> bool:
        return self.is_solved or self.is_failed

    @property
    def unsolved_children(self) -> List["Node"]:
        if not self.children:
            return []
        return [child for child in self.children if not child.is_solved]

    # @property
    # def grouped_unsolved_leaves(self) -> list:
    #     # returns lists of like
    #     # based on AND/OR
    #     # like i want to group based on "groups you need to solve in order to solve the problem"


class Transform(ABC):
    "An integral transform -- base class"
    # forward and backward modify the nodetree directly

    def __init__(self):
        pass

    @abstractmethod
    def forward(self, node: Node) -> None:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def backward(self, node: Node) -> None:
        if not node.solution:
            raise ValueError("Node has no solution")

    @abstractmethod
    def check(self, node: Node) -> bool:
        raise NotImplementedError("Not implemented")


class PullConstant(Transform):
    _constant: Expr = None
    _non_constant_part: Expr = None

    def check(self, node: Node) -> bool:
        expr = node.expr
        var = node.var
        if isinstance(expr, Prod):
            # if there is a constant, pull it out
            if isinstance(expr.terms[0], Const):
                self._constant = expr.terms[0]
                self._non_constant_part = Prod(expr.terms[1:]).simplify()
                return True

            # or if there is a symbol that's not the variable, pull it out
            for i, term in enumerate(expr.terms):
                is_nonvar_symbol = isinstance(term, Symbol) and term != var
                is_nonvar_power = (
                    isinstance(term, Power)
                    and isinstance(term.base, Symbol)
                    and term.base != var
                )
                if is_nonvar_symbol or is_nonvar_power:
                    self._constant = term
                    self._non_constant_part = Prod(
                        expr.terms[:i] + expr.terms[i + 1 :]
                    ).simplify()
                    return True

        return False

    def forward(self, node: Node):
        node.children = [Node(self._non_constant_part, node.var, self, node)]

    def backward(self, node: Node) -> None:
        super().backward(node)
        node.parent.solution = (self._constant * node.solution).simplify()


class PolynomialDivision(Transform):
    _numerator: Polynomial = None
    _denominator: Polynomial = None

    def check(self, node: Node) -> bool:
        # This is so messy we can honestly just do catching in the `to_polynomial`
        expr = node.expr
        # currently we don't support division of polynomials with multiple variables
        if expr.symbols() != [node.var]:
            return False
        if not isinstance(expr, Prod):
            return False

        ## Don't contain any SingleFunc with inner containing var
        def _contains_singlefunc_w_inner(expr: Expr) -> bool:
            if isinstance(expr, SingleFunc) and expr.inner.contains(node.var):
                return True

            return any([_contains_singlefunc_w_inner(e) for e in expr.children()])

        if _contains_singlefunc_w_inner(expr):
            return False

        ## Make sure each factor is a polynomial
        for factor in expr.terms:

            def _is_polynomial(expression: Expr):
                if isinstance(expression, Power):
                    if not (
                        isinstance(expression.exponent, Const)
                        and expression.exponent.value.denominator == 1
                    ):
                        return False
                    return True
                if isinstance(expression, Const) or isinstance(expression, Symbol):
                    return True

                if isinstance(expression, Sum):
                    return all(
                        [_is_polynomial(term, node.var) for term in expression.terms]
                    )

                raise NotImplementedError(f"Not implemented: {expression}")

            if not _is_polynomial(factor):
                return False

        ## Make sure numerator and denominator are good
        numerator = 1
        denominator = 1
        for factor in expr.terms:
            b, x = deconstruct_power(factor)
            if x.value > 0:
                numerator *= factor
            else:
                denominator *= Power(b, -x).simplify()

        numerator = numerator.simplify()
        denominator = denominator.simplify()

        try:
            numerator_list = to_polynomial(numerator, node.var)
            denominator_list = to_polynomial(denominator, node.var)
        except AssertionError:
            return False

        if len(numerator_list) < len(denominator_list):
            return False

        self._numerator = numerator_list
        self._denominator = denominator_list
        return True

    def forward(self, node: Node):
        var = node.var
        quotient = np.zeros(len(self._numerator) - len(self._denominator) + 1)

        while self._numerator.size >= self._denominator.size:
            quotient_degree = len(self._numerator) - len(self._denominator)
            quotient_coeff = self._numerator[-1] / self._denominator[-1]
            quotient[quotient_degree] = quotient_coeff
            self._numerator -= np.concatenate(
                ([0] * quotient_degree, self._denominator * quotient_coeff)
            )
            self._numerator = rid_ending_zeros(self._numerator)

        remainder = polynomial_to_expr(self._numerator, var) / polynomial_to_expr(
            self._denominator, var
        )
        quotient_expr = polynomial_to_expr(quotient, var)
        answer = (quotient_expr + remainder).simplify()
        node.children = [Node(answer, var, self, node)]

    def backward(self, node: Node) -> None:
        super().backward(node)
        node.parent.solution = node.solution


class Expand(Transform):
    def forward(self, node: Node):
        node.children = [Node(node.expr.expand(), node.var, self, node)]

    def check(self, node: Node) -> bool:
        return node.expr.expandable()

    def backward(self, node: Node) -> None:
        super().backward(node)
        node.parent.solution = node.solution


class Additivity(Transform):
    def forward(self, node: Node):
        node.type = "AND"
        node.children = [Node(e, node.var, self, node) for e in node.expr.terms]

    def check(self, node: Node) -> bool:
        return isinstance(node.expr, Sum)

    def backward(self, node: Node) -> None:
        super().backward(node)

        # For this to work, we must have a solution for each sibling.
        if not all([child.solution for child in node.parent.children]):
            return ValueError(f"Additivity backward for {node} failed")

        node.parent.solution = Sum(
            [child.solution for child in node.parent.children]
        ).simplify()


# Let's just add all the transforms we've used for now.
# and we will make this shit good and generalized later.
class B_Tan(Transform):
    _variable_change = None

    def forward(self, node: Node):
        intermediate = generate_intermediate_var()
        expr = node.expr
        # y = tanx
        new_integrand = replace(expr, Tan(node.var), intermediate) / (
            1 + intermediate**2
        )
        new_node = Node(new_integrand, intermediate, self, node)
        node.children = [new_node]

        self._variable_change = Tan(node.var)

    def check(self, node: Node) -> bool:
        expr = node.expr
        return contains(expr, Tan) and count(expr, Tan(node.var)) == count(
            expr, node.var
        )  # ugh everything is so sus

    def backward(self, node: Node) -> None:
        super().backward(node)
        node.parent.solution = replace(
            node.expr, node.var, self._variable_change
        ).simplify()


class A(Transform):
    def forward(self, node: Node):
        expr = node.expr
        r1 = replace_class(
            expr,
            [Tan, Csc, Cot, Sec],
            [
                lambda x: Sin(x) / Cos(x),
                lambda x: 1 / Sin(x),
                lambda x: Cos(x) / Sin(x),
                lambda x: 1 / Cos(x),
            ],
        ).simplify()
        r2 = replace_class(
            expr,
            [Sin, Cos, Cot, Sec],
            [
                lambda x: 1 / Csc(x),
                lambda x: 1 / Tan(x) / Csc(x),
                lambda x: 1 / Tan(x),
                lambda x: Tan(x) * Csc(x),
            ],
        ).simplify()
        r3 = replace_class(
            expr,
            [Sin, Cos, Tan, Csc],
            [
                lambda x: 1 / Cot(x) / Sec(x),
                lambda x: 1 / Sec(x),
                lambda x: 1 / Cot(x),
                lambda x: Cot(x) * Sec(x),
            ],
        ).simplify()

        stuff = [r1, r2, r3]
        for thing in stuff:
            if thing.__repr__() == expr.__repr__():
                stuff.remove(thing)
        node.children = [Node(option, node.var, self, node) for option in stuff]
        node.type = "OR"

    def check(self, node: Node) -> bool:
        # make sure that this node didn't get here by this transform
        if isinstance(node.transform, A):
            return False

        expr = node.expr
        return contains(expr, TrigFunction)

    def backward(self, node: Node) -> None:
        super().backward(node)
        node.parent.solution = node.solution


class C_Sin(Transform):
    _variable_change = None

    def forward(self, node: Node):
        intermediate_var = generate_intermediate_var()
        self._variable_change = ArcSin(node.var)
        # intermediate = sin^-1 x
        new_thing = replace(node.expr, node.var, Sin(intermediate_var)) * Cos(
            intermediate_var
        )
        new_thing = new_thing.simplify()

        # then that's a node and u store the transform and u take the integral of that.
        node.children = [Node(new_thing, intermediate_var, self, node)]

    def check(self, node: Node) -> bool:
        s = f"(1 + (-1 * {node.var.name}^2))"
        return s in node.expr.__repr__()  # ugh unclean

    def backward(self, node: Node) -> None:
        super().backward(node)
        node.parent.solution = replace(
            node.solution, node.var, self._variable_change
        ).simplify()


class C_Tan(Transform):
    _variable_change = None

    def forward(self, node: Node):
        intermediate = generate_intermediate_var()
        dy_dx = Sec(intermediate) ** 2
        new_thing = (replace(node.expr, node.var, Tan(intermediate)) * dy_dx).simplify()
        node.children = [Node(new_thing, intermediate, self, node)]

        self._variable_change = ArcTan(node.var)

    def check(self, node: Node) -> bool:
        s2 = f"1 + {node.var.name}^2"
        return s2 in node.expr.__repr__()

    def backward(self, node: Node) -> None:
        super().backward(node)
        node.parent.solution = replace(
            node.solution, node.var, self._variable_change
        ).simplify()


HEURISTICS = [B_Tan, A, C_Sin, C_Tan]
SAFE_TRANSFORMS = [Additivity, PullConstant, Expand, PolynomialDivision]


def _check_if_solvable(node: Node):
    expr = node.expr
    var = node.var
    answer = None
    if isinstance(expr, Power):
        if expr.base == var and isinstance(expr.exponent, Const):
            n = expr.exponent
            answer = (1 / (n + 1)) * Power(var, n + 1) if n != -1 else Log(expr.base)
        elif isinstance(expr.base, Symbol) and expr.base != var:
            answer = expr * var

    elif isinstance(expr, Symbol):
        answer = Fraction(1 / 2) * Power(var, 2) if expr == var else expr * var
    elif isinstance(expr, Const):
        answer = expr * var

    if answer is None:
        return

    # node.children = [Node(answer, var=node.var, parent=node, type="SOLUTION")]
    node.type = "SOLUTION"
    node.solution = answer


def _cycle(node: Node):
    # 1. APPLY ALL SAFE TRANSFORMS
    _integrate_safely(node)

    # now we have a tree with all the safe transforms applied
    # 2. LOOK IN TABLE
    for leaf in node.unfinished_leaves:
        _check_if_solvable(leaf)

    if len(node.unfinished_leaves) == 0:
        return "SOLVED"

    # 3. APPLY HEURISTICS
    next_node = node.unfinished_leaves[0]  # random lol
    _integrate_heuristically(next_node)

    next_next_node = _get_next_node_post_heuristic(next_node)
    return next_next_node


def _get_next_node_post_heuristic(node: Node) -> Node:

    if len(node.unfinished_leaves) == 0:
        if node.is_failed:
            # we want to go back and solve the parent

            parent = node.parent
            while len(parent.children) == 1 or parent.type == "AND":
                if parent.parent is None:
                    # we've reached root.
                    # this means... we can't solve this integral.
                    return None
                parent = parent.parent
            # now parent is the lowest OR node with multiple children
            return _get_next_node_post_heuristic(parent)
        else:
            raise NotImplementedError("TODO _get_next_node for success nodes")

    if len(node.unfinished_leaves) == 1:
        return node.unfinished_leaves[0]

    if len(node.unfinished_leaves) > 1:
        return _nesting_node(node)


# a recursive function.
def _nesting_node(node: Node) -> Node:
    if len(node.unsolved_children) == 1:
        return _nesting_node(node.unsolved_children[0])

    if len(node.unsolved_children) == 0:
        return node  # base case ???
        raise ValueError("nesting_node on a solved node?")

    is_2nd_lowest_parent = all(
        [not child.unsolved_children for child in node.unsolved_children]
    )
    fn = min if node.type == "OR" else max
    if is_2nd_lowest_parent:
        return _get_node_with_best_nesting(node.unsolved_children, fn)

    childrens_best_nodes = [_nesting_node(c) for c in node.unsolved_children]
    return _get_node_with_best_nesting(childrens_best_nodes, fn)


def _get_node_with_best_nesting(
    nodes: List[Node], fn: Callable[[List[Node]], Node]
) -> Node:
    results = [nesting(node.expr, node.var) for node in nodes]
    best_value = fn(results)
    return nodes[results.index(best_value)]


class Integration:
    """
    Keeps track of integration work as we go
    """

    @staticmethod
    def integrate(integrand: Expr, var: Symbol):

        root = Node(integrand, var)
        curr_node = root
        while True:
            answer = _cycle(curr_node)

            if root.is_finished:
                break

            if answer == "SOLVED":
                # just do any other thing in root
                curr_node = _get_next_node_post_heuristic(root)
            else:
                curr_node = answer

        if root.is_failed:
            raise NotImplementedError(f"Failed to integrate {integrand} wrt {var}")

        # now we have a solved tree or a failed tree
        # we can go back and get the answer
        solved_leaves = [leaf for leaf in root.leaves if leaf.is_solved]
        for leaf in solved_leaves:
            # GO backwards on each leaf until it errors out, then go backwards on the next leaf.
            l = leaf
            while True:
                try:
                    l.transform.backward(l)
                    l = l.parent
                except ValueError:
                    break

        if root.solution is None:
            raise ValueError("something went wrong while going backwards...")

        return root.solution


def _integrate_safely(node: Node):
    for transform in SAFE_TRANSFORMS:
        tr = transform()
        if tr.check(node):
            tr.forward(node)
            for child in node.children:
                _integrate_safely(child)


def _integrate_heuristically(node: Node):
    for transform in HEURISTICS:
        tr = transform()
        if tr.check(node):
            tr.forward(node)

    if not node.children:
        node.type = "FAILURE"
        return

    if len(node.children) > 1:
        node.type = "OR"


CCOUNT = 0

if __name__ == "__main__":

    F = Fraction
    x, y = symbols("x y")
    expression = -5 * x**4 / (1 - x**2) ** F(5, 2)
    print(expression)
    integral = Integration.integrate(expression, x)  # TODO auto simplify
    print(integral)
    breakpoint()
