from typing import Tuple
from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch_size, num_channels, input_height, input_width = input.shape
    kernel_height, kernel_width = kernel
    assert input_height % kernel_height == 0
    assert input_width % kernel_width == 0

    output_height = input_height // kernel_height
    output_width = input_width // kernel_width

    x = input.contiguous()

    tiled_tensor = x.view(
        batch_size,
        num_channels,
        output_height,
        kernel_height,
        output_width,
        kernel_width,
    )
    tiled_tensor = tiled_tensor.permute(0, 1, 2, 4, 3, 5)
    tiled_tensor = tiled_tensor.contiguous()
    tiled_tensor = tiled_tensor.view(
        batch_size,
        num_channels,
        output_height,
        output_width,
        kernel_height * kernel_width,
    )

    return tiled_tensor, output_height, output_width


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply 2D average pooling to input tensor.

    Args:
    ----
        input: Input tensor of shape (batch, channel, height, width)
        kernel: Tuple of (kernel_height, kernel_width) for pooling window

    Returns:
    -------
        Output tensor of shape (batch, channel, height/kernel_height, width/kernel_width)
        where each value is the average of the corresponding window in the input

    """
    tiled, new_height, new_width = tile(input, kernel)

    return tiled.mean(dim=4).view(tiled.shape[0], tiled.shape[1], new_height, new_width)


max_reduce = FastOps.reduce(operators.max, float("-inf"))


def argmax(x: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a 1-hot tensor.

    Args:
    ----
        x: Input tensor
        dim: Dimension to compute argmax over

    Returns:
    -------
        1-hot tensor with same shape as input, where 1s indicate maximum values

    """
    max_values = max_reduce(x, dim)
    result_mask = x == max_values
    return result_mask


class Max(Function):
    @staticmethod
    def forward(ctx: Context, x: Tensor, dim: Tensor) -> Tensor:
        """Take the maximum value along a dimension.

        Args:
        ----
            ctx: Context for backprop
            x: Input tensor
            dim: Dimension to reduce

        Returns:
        -------
            Tensor with max values along specified dimension

        """
        if isinstance(dim, Tensor):
            dim_index = int(dim._tensor._storage[0])
        else:
            dim_index = int(dim)

        ctx.save_for_backward(x, x._ensure_tensor(dim_index))
        return max_reduce(x, dim_index)

    @staticmethod
    def backward(ctx: Context, grad_out: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute gradient of max reduction.

        Args:
        ----
            ctx: Context from forward pass
            grad_out: Upstream gradient

        Returns:
        -------
            Tuple of gradients for input and dimension

        """
        x, dim = ctx.saved_values
        dim_value = int(dim.item())
        grad_input = (argmax(x, dim_value) * grad_out).sum(dim=dim_value)
        return grad_input, x._ensure_tensor(0.0)


def max(x: Tensor, dim: int) -> Tensor:
    """Apply max reduction along a dimension.

    Args:
    ----
        x: Input tensor
        dim: Dimension to reduce

    Returns:
    -------
        Tensor with maximum values along specified dimension

    """
    return Max.apply(x, x._ensure_tensor(dim))


def softmax(x: Tensor, dim: int) -> Tensor:
    """Apply softmax along a dimension.

    Args:
    ----
        x: input tensor
        dim: dimension to apply softmax

    """
    exponential_values = x.exp()
    exponential_sums = exponential_values.sum(dim)
    output_shape = list(exponential_values.shape)
    output_shape[dim] = 1
    return exponential_values / exponential_sums.contiguous().view(*output_shape)


def logsoftmax(x: Tensor, dim: int) -> Tensor:
    """Apply log softmax along a dimension.

    Args:
    ----
        x: input tensor
        dim: dimension to apply log softmax

    Returns:
    -------
        Tensor with log softmax applied

    """
    max_values = max(x, dim)
    shifted = x - max_values
    exp_values = shifted.exp()
    sum_exp = exp_values.sum(dim)
    return shifted - sum_exp.log().view(*max_values.shape)


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply 2D max pooling.

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    """
    batch_size, num_channels = input.shape[:2]
    tiled_tensor, output_height, output_width = tile(input, kernel)
    pooled_tensor = max(tiled_tensor, 4)
    return pooled_tensor.view(batch_size, num_channels, output_height, output_width)


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """Apply dropout with probability rate.

    Args:
    ----
        input: input tensor
        rate: dropout probability
        ignore: if True, don't apply dropout

    """
    if not ignore and rate > 0.0:
        dropout_mask = rand(input.shape) > rate
        return input * dropout_mask
    else:
        return input
