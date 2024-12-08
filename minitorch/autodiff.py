from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    return (
        f(*vals[:arg], vals[arg] + epsilon, *vals[arg + 1 :])
        - f(*vals[:arg], vals[arg] - epsilon, *vals[arg + 1 :])
    ) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulates the derivative of this variable."""
        pass

    @property
    def unique_id(self) -> int:
        """Returns the unique id of this variable."""
        return id(self)

    def is_leaf(self) -> bool:
        """Checks if this variable is a leaf in the computation graph."""
        return True

    def is_constant(self) -> bool:
        """Checks if this variable is a constant."""
        return True

    @property
    def parents(self) -> Iterable["Variable"]:
        """Returns the parent variables in the computation graph."""
        return []

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Applies the chain rule to compute gradients."""
        return []


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    visited = set()
    order = []

    def dfs(var: Variable) -> None:
        if var.is_constant() or var.unique_id in visited:
            return
        visited.add(var.unique_id)
        for parent in var.parents:
            dfs(parent)
        order.append(var)

    dfs(variable)
    return reversed(order)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
    ----
        variable: The right-most variable
        deriv: Its derivative that we want to propagate backward to the leaves.

    Returns:
    -------
        No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.

    """
    derivatives = {variable.unique_id: deriv}

    for var in topological_sort(variable):
        if var.unique_id in derivatives:
            if var.is_leaf():
                var.accumulate_derivative(derivatives[var.unique_id])
            else:
                for parent, grad in var.chain_rule(derivatives[var.unique_id]):
                    if parent.unique_id in derivatives:
                        derivatives[parent.unique_id] += grad
                    else:
                        derivatives[parent.unique_id] = grad
                del derivatives[var.unique_id]

    derivatives.clear()


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Returns the saved values as tensors."""
        return self.saved_values
