"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Callable, Iterable

# ## Task 0.1

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(a: float, b: float) -> float:
    """Multiplies two numbers

    Args:
    ----
        a: A float
        b: A float
    ----

    Returns:
    -------
        The product of `a` and `b`
        (a * b)
    -------

    """
    return a * b


def id(a: float) -> float:
    """Returns the identity of a number

    Args:
    ----
        a: A float
    ----

    Returns:
    -------
        The identity of `a`
        (a)
    -------


    """
    return a


def add(a: float, b: float) -> float:
    """Adds two numbers

    Args:
    ----
        a: A float
        b: A float
    ----

    Returns:
    -------
        The sum of `a` and `b`
        (a + b)
    -------

    """
    return a + b


def neg(a: float) -> float:
    """Negates a number

    Args:
    ----
        a: A float
    ----

    Returns:
    -------
        The negation of `a`
        (-a)
    -------

    """
    return -a


def lt(a: float, b: float) -> float:
    """Returns 1.0 if a is less than b, 0.0 otherwise

    Args:
    ----
        a: A float
        b: A float
    ----

    Returns:
    -------
        1.0 if a < b else 0.0
    -------

    """
    return 1.0 if a < b else 0.0


def eq(a: float, b: float) -> float:
    """Returns 1.0 if a is equal to b, 0.0 otherwise

    Args:
    ----
        a: A float
        b: A float
    ----

    Returns:
    -------
        1.0 if a == b, 0.0 otherwise
        (1.0 if a == b else 0.0)
    -------

    """
    return 1.0 if a == b else 0.0


def max(a: float, b: float) -> float:
    """Returns the maximum of two numbers

    Args:
    ----
        a: A float
        b: A float
    ----

    Returns:
    -------
        The maximum of `a` and `b`
        (a if a > b else b)
    -------

    """
    return a if a > b else b


def is_close(a: float, b: float) -> float:
    """Returns 1.0 if a is close to b, 0.0 otherwise

    Args:
    ----
        a: A float
        b: A float
    ----

    Returns:
    -------
        1.0 if a is close to b, 0.0 otherwise
        (1.0 if abs(a - b) < 1e-2 else 0.0)
    -------

    """
    return 1.0 if abs(a - b) < 1e-2 else 0.0


def sigmoid(a: float) -> float:
    """Calculates the sigmoid function for a given number

    Args:
    ----
        a: A float
    ----

    Returns:
    -------
        The sigmoid function value for `a`
        (1.0 / (1.0 + math.exp(-a)) if a >= 0 else math.exp(a) / (1.0 + math.exp(a)))
    -------

    """
    if a >= 0:
        return 1.0 / (1.0 + math.e ** (-a))
    else:
        return math.e ** (a) / (1.0 + math.e ** (a))


def relu(a: float) -> float:
    """Applies the ReLU activation function to a given number

    Args:
    ----
        a: A float
    ----

    Returns:
    -------
        The ReLU function value for `a`
        (a if a > 0 else 0)
    -------

    """
    return a if a > 0 else 0


EPS = 1e-6


def log(a: float) -> float:
    """Calculates the natural logarithm for a given number

    Args:
    ----
        a: A float
    ----

    Returns:
    -------
        The natural logarithm of `a`
        (math.log(a))
    -------

    """
    if a <= 0:
        return -float("inf")
    return math.log(a + EPS)


def exp(a: float) -> float:
    """Calculates the exponential function for a given number

    Args:
    ----
        a: A float
    ----

    Returns:
    -------
        The exponential function value for `a`
        (math.exp(a))
    -------

    """
    return math.exp(a)


def inv(a: float) -> float:
    """Calculates the reciprocal of a given number

    Args:
    ----
        a: A float
    ----

    Returns:
    -------
        The reciprocal of `a`
        (1.0 / a)
    -------

    """
    return 1.0 / a


def log_back(a: float, grad: float) -> float:
    """Computes the derivative of log times a second arg

    Args:
    ----
        a: A float
        grad: A float
    ----

    Returns:
    -------
        The derivative of log times a second arg
        (grad * (1.0 / a))
    -------

    """
    return grad * (1.0 / a)


def inv_back(a: float, grad: float) -> float:
    """Computes the derivative of reciprocal times a second arg

    Args:
    ----
        a: A float
        grad: A float
    ----

    Returns:
    -------
        The derivative of reciprocal times a second arg
        (grad * (-1.0 / a**2))
    -------

    """
    return grad * (-1.0 / a**2)


def relu_back(a: float, grad: float) -> float:
    """Computes the derivative of ReLU times a second arg

    Args:
    ----
        a: A float
        grad: A float
    ----

    Returns:
    -------
        The derivative of ReLU times a second arg
        (grad * (1.0 if a > 0 else 0))
    -------

    """
    return grad * (1.0 if a > 0 else 0)


def sigmoid_back(a: float, b: float) -> float:
    """Computes the derivative of sigmoid times a second arg

    Args:
    ----
        a: A float
        b: A float
    ----

    Returns:
    -------
        The derivative of sigmoid times a second arg
    -------

    """
    return mul(mul(sigmoid(a), add(1, neg(sigmoid(a)))), b)


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Higher-order function that returns a function to apply a given function to each element of an iterable

    Args:
    ----
        fn: A function that takes a float and returns a float
    ----

    Returns:
    -------
        A function that takes an iterable and returns a new iterable with `fn` applied to each element
    -------

    """

    def apply_fn(lst: Iterable[float]) -> Iterable[float]:
        return [fn(x) for x in lst]

    return apply_fn


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Higher-order function that returns a function to combine elements from two iterables using a given function

    Args:
    ----
        fn: A function that takes two floats and returns a float
    ----

    Returns:
    -------
        A function that takes two iterables and returns a new iterable with `fn` applied to pairs of elements from both iterables
    -------

    """

    def apply_fn(lst1: Iterable[float], lst2: Iterable[float]) -> Iterable[float]:
        return [fn(x, y) for x, y in zip(lst1, lst2)]

    return apply_fn


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    """Higher-order function that returns a function to reduce an iterable to a single value

    Args:
    ----
        fn: A function that combines two floats into one
        start: The initial value for the reduction
    ----

    Returns:
    -------
        A function that takes an iterable and returns a single float value
        This function computes the reduction as:
        fn(x_n, fn(x_{n-1}, ... fn(x_1, fn(x_0, start))...))
        where x_0, x_1, ..., x_n are the elements of the input iterable
    -------

    """

    def apply_reduce(lst: Iterable[float]) -> float:
        result = start
        for x in lst:
            result = fn(result, x)
        return result

    return apply_reduce


# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Use `map` and `neg` to negate each element in `ls`

    Args:
    ----
        ls: An iterable of floats
    ----

    Returns:
    -------
        A new iterable with each element negated
    -------

    """
    return map(neg)(ls)


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add the elements of `ls1` and `ls2` using `zipWith` and `add`

    Args:
    ----
        ls1: An iterable of floats
        ls2: An iterable of floats
    ----

    Returns:
    -------
        A new iterable with each element from both iterables added together
    -------

    """
    return zipWith(add)(ls1, ls2)


def sum(ls: Iterable[float]) -> float:
    """Sum up a list using `reduce` and `add`.

    Args:
    ----
        ls: An iterable of floats
    ----

    Returns:
    -------
        The sum of the elements in the list
    -------

    """
    return reduce(add, 0)(ls)


def prod(ls: Iterable[float]) -> float:
    """Product of a list using `reduce` and `mul`.

    Args:
    ----
        ls: An iterable of floats
    ----

    Returns:
    -------
        The product of the elements in the list
    -------

    """
    return reduce(mul, 1)(ls)
