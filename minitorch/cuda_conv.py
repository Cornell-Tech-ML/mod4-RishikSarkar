from typing import Tuple

import numba
from numba import cuda
from typing import TypeVar, Any
import numba.cuda
from .autodiff import Context
from .tensor import Tensor
from .tensor_data import (
    Shape,
    Storage,
    Strides,
    UserShape,
    to_index,
    index_to_position,
    broadcast_index,
)
from .tensor_functions import Function
from numba.cuda import jit as _jit

FakeCUDAKernel = Any

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs: Any) -> Fn:
    """Device-specific JIT decorator for CUDA functions.

    Args:
    ----
        fn (Fn): Function to compile
        kwargs (Any): Additional arguments for the JIT compiler

    Returns:
    -------
        Fn: Compiled device function

    """
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn: Any, **kwargs: Any) -> FakeCUDAKernel:
    """JIT decorator for CUDA functions.

    Args:
    ----
        fn (Any): Function to compile
        kwargs (Any): Additional arguments for the JIT compiler

    Returns:
    -------
        FakeCUDAKernel: Compiled CUDA kernel

    """
    return _jit(**kwargs)(fn)  # type: ignore

to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

def _cuda_conv1d(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
    is_reverse: bool,
) -> None:
    """1D Convolution CUDA kernel implementation.

    Given input tensor of
       `batch, in_channels, width`
    and weight tensor
       `out_channels, in_channels, k_width`
    
    Computes padded output of
       `batch, out_channels, width`

    Requirements:
    * All data must be first moved to shared memory
    * Only read each cell in input and weight once
    * Only write to global memory once per kernel

    Args:
    ----
        out (Storage): storage for output tensor
        out_shape (Shape): shape for output tensor
        out_strides (Strides): strides for output tensor
        out_size (int): size of the output tensor
        a_storage (Storage): storage for input tensor
        a_shape (Shape): shape for input tensor
        a_strides (Strides): strides for input tensor
        b_storage (Storage): storage for weight tensor
        b_shape (Shape): shape for weight tensor
        b_strides (Strides): strides for weight tensor
        is_reverse (bool): anchor weight at left (False) or right (True)

    """
    BLOCK_DIM = 16
    CACHE_DIM = 32

    batch_size, output_channels, output_width = out_shape
    batch_size_in, input_channels, input_width = a_shape
    kernel_out_channels, kernel_in_channels, kernel_width = b_shape

    assert batch_size == batch_size_in and output_channels == kernel_out_channels
    assert input_channels == kernel_in_channels and output_width <= input_width

    current_width = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    tile_start_width = cuda.blockIdx.x * cuda.blockDim.x
    current_channel = cuda.blockIdx.z
    thread_x, thread_y = cuda.threadIdx.x, cuda.threadIdx.y

    kernel_cache = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    input_cache = cuda.shared.array((CACHE_DIM, CACHE_DIM), numba.float64)

    b_batch_stride, b_channel_stride, b_width_stride = b_strides
    a_batch_stride, a_channel_stride, a_width_stride = a_strides
    out_batch_stride, out_channel_stride, out_width_stride = out_strides

    kernel_step = -1 if is_reverse else 1

    for batch_idx in range(batch_size):
        accumulator = 0.0

        for channel_start in range(0, input_channels, BLOCK_DIM):
            channel_cache_idx = channel_start + thread_x

            for kernel_start in range(0, kernel_width, BLOCK_DIM):
                kernel_idx = thread_y + kernel_start

                if channel_cache_idx < input_channels and kernel_idx < kernel_width:
                    cache_idx = (current_channel * b_batch_stride + channel_cache_idx * b_channel_stride + kernel_idx * b_width_stride)
                    kernel_cache[thread_x, thread_y] = b_storage[cache_idx]
                else:
                    kernel_cache[thread_x, thread_y] = 0.0

                numba.cuda.syncthreads()

                for width_offset in range(0, CACHE_DIM, BLOCK_DIM):
                    if is_reverse:
                        pos = (tile_start_width - kernel_start - BLOCK_DIM + 1 + width_offset + thread_y)
                    else:
                        pos = tile_start_width + kernel_start + width_offset + thread_y

                    if channel_cache_idx < input_channels and 0 <= pos < input_width:
                        input_cache_idx = (batch_idx * a_batch_stride + channel_cache_idx * a_channel_stride + pos * a_width_stride)
                        input_cache[thread_x, width_offset + thread_y] = a_storage[input_cache_idx]
                    else:
                        input_cache[thread_x, width_offset + thread_y] = 0.0

                numba.cuda.syncthreads()

                if thread_y == 0 and current_width < output_width:
                    for channel_idx_inner in range(channel_start, min(input_channels, channel_start + BLOCK_DIM)):
                        for kernel_idx_inner in range(kernel_start, min(kernel_width, kernel_start + BLOCK_DIM)):
                            pos = current_width + kernel_idx_inner * kernel_step

                            if is_reverse:
                                min_bound = tile_start_width - kernel_start - BLOCK_DIM + 1
                            else:
                                min_bound = tile_start_width + kernel_start

                            max_bound = min_bound + CACHE_DIM

                            if min_bound <= pos < max_bound and 0 <= pos < input_width:
                                accumulator += (kernel_cache[channel_idx_inner - channel_start, kernel_idx_inner - kernel_start] * input_cache[channel_idx_inner - channel_start, abs(pos - min_bound)])
                numba.cuda.syncthreads()

        if thread_y == 0 and current_width < output_width:
            output_idx = batch_idx * out_batch_stride + current_channel * out_channel_stride + current_width * out_width_stride
            out[output_idx] = accumulator

def _cuda_conv2d(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
    is_reverse: bool,
) -> None:
    """2D Convolution CUDA kernel implementation.

    Given input tensor of
       `batch, in_channels, height, width`
    and weight tensor
       `out_channels, in_channels, k_height, k_width`
    
    Computes padded output of
       `batch, out_channels, height, width`

    Requirements:
    * All data must be first moved to shared memory
    * Only read each cell in input and weight once
    * Only write to global memory once per kernel

    Args:
    ----
        out (Storage): storage for output tensor
        out_shape (Shape): shape for output tensor
        out_strides (Strides): strides for output tensor
        out_size (int): size of the output tensor
        a_storage (Storage): storage for input tensor
        a_shape (Shape): shape for input tensor
        a_strides (Strides): strides for input tensor
        b_storage (Storage): storage for weight tensor
        b_shape (Shape): shape for weight tensor
        b_strides (Strides): strides for weight tensor
        is_reverse (bool): anchor weight at top-left (False) or bottom-right (True)
        
    """
    BLOCK_DIM = 16
    CACHE_DIM = 32

    batch_size, output_channels, output_height, output_width = out_shape
    batch_size_in, input_channels, input_height, input_width = a_shape
    kernel_out_channels, kernel_in_channels, kernel_height, kernel_width = b_shape

    assert batch_size == batch_size_in and input_channels == kernel_in_channels and output_channels == kernel_out_channels, "Shape mismatch between input, output, and weight tensors."
    assert output_width <= input_width and output_height <= input_height, "Output dims exceed input dims."

    current_width = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    current_height = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    tile_start_width = cuda.blockIdx.x * cuda.blockDim.x
    tile_start_height = cuda.blockIdx.y * cuda.blockDim.y
    current_out_channel = cuda.blockIdx.z
    thread_x = cuda.threadIdx.x
    thread_y = cuda.threadIdx.y

    kernel_cache = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    input_cache = cuda.shared.array((CACHE_DIM, CACHE_DIM), numba.float64)

    b_batch_stride, b_channel_stride, b_height_stride, b_width_stride = b_strides
    a_batch_stride, a_channel_stride, a_height_stride, a_width_stride = a_strides
    out_batch_stride, out_channel_stride, out_height_stride, out_width_stride = out_strides

    kid = -1 if is_reverse else 1

    for batch_i in range(batch_size):
        out_pos = batch_i * out_batch_stride + current_out_channel * out_channel_stride + current_height * out_height_stride + current_width * out_width_stride
        tmp = 0.0

        for in_channel_i in range(input_channels):
            for kh_start in range(0, kernel_height, BLOCK_DIM):
                for kw_start in range(0, kernel_width, BLOCK_DIM):
                    kw_now = kw_start + thread_x
                    kh_now = kh_start + thread_y

                    if kh_now < kernel_height and kw_now < kernel_width:
                        weight_cache_pos = current_out_channel * b_batch_stride + in_channel_i * b_channel_stride + kh_now * b_height_stride + kw_now * b_width_stride
                        kernel_cache[(thread_x, thread_y)] = b_storage[weight_cache_pos]
                    else:
                        kernel_cache[(thread_x, thread_y)] = 0.0
                    numba.cuda.syncthreads()

                    for w_cache_bias in range(0, CACHE_DIM, BLOCK_DIM):
                        for h_cache_bias in range(0, CACHE_DIM, BLOCK_DIM):
                            if is_reverse:
                                w_cache_pos = tile_start_width - kw_start - BLOCK_DIM + 1 + w_cache_bias + thread_x
                                h_cache_pos = tile_start_height - kh_start - BLOCK_DIM + 1 + h_cache_bias + thread_y
                            else:
                                w_cache_pos = tile_start_width + kw_start + w_cache_bias + thread_x
                                h_cache_pos = tile_start_height + kh_start + h_cache_bias + thread_y

                            if 0 <= w_cache_pos < input_width and 0 <= h_cache_pos < input_height:
                                input_cache_pos = batch_i * a_batch_stride + in_channel_i * a_channel_stride + h_cache_pos * a_height_stride + w_cache_pos * a_width_stride
                                input_cache[(w_cache_bias + thread_x, h_cache_bias + thread_y)] = a_storage[input_cache_pos]
                            else:
                                input_cache[(w_cache_bias + thread_x, h_cache_bias + thread_y)] = 0.0
                            numba.cuda.syncthreads()

                    if current_height < output_height and current_width < output_width:
                        for khi in range(kh_start, min(kernel_height, kh_start + BLOCK_DIM)):
                            h_now = current_height + khi * kid
                            height_cache_min = tile_start_height - kh_start - BLOCK_DIM + 1 if is_reverse else tile_start_height + kh_start
                            height_cache_max = height_cache_min + CACHE_DIM

                            if not (0 <= h_now < input_height and height_cache_min <= h_now < height_cache_max):
                                continue

                            for kwi in range(kw_start, min(kernel_width, kw_start + BLOCK_DIM)):
                                w_now = current_width + kwi * kid
                                width_cache_min = tile_start_width - kw_start - BLOCK_DIM + 1 if is_reverse else tile_start_width + kw_start
                                width_cache_max = width_cache_min + CACHE_DIM

                                if not (0 <= w_now < input_width and width_cache_min <= w_now < width_cache_max):
                                    continue

                                tmp += kernel_cache[(kwi - kw_start, khi - kh_start)] * input_cache[(abs(w_now - width_cache_min), abs(h_now - height_cache_min))]

                    numba.cuda.syncthreads()

        if current_height < output_height and current_width < output_width:
            out[out_pos] = tmp

cuda_conv2d = cuda.jit()(_cuda_conv2d)
cuda_conv1d = cuda.jit()(_cuda_conv1d)

class Conv1dFun(Function):
    @staticmethod
    def forward_inner(
        output_shape: UserShape, input: Tensor, weight: Tensor, reversed: bool = False
    ) -> Tensor:
        """Helper function to compute forward pass for both 1D and 2D convolutions.

        Args:
        ----
            output_shape (UserShape): desired shape of output tensor
            input (Tensor): input tensor
            weight (Tensor): weight/kernel tensor
            reversed (bool): whether to compute reverse convolution

        Returns:
        -------
            Tensor: output tensor of convolution operation

        """
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2, "Input and weight channels mismatch."

        output = input.zeros(output_shape)

        THREADS_PER_BLOCK = 16
        blockspergrid = ((w + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK, 1, out_channels)
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        cuda_conv1d[blockspergrid, threadsperblock](*output.tuple(), output.size, *input.tuple(), *weight.tuple(), reversed)
        
        return output

    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Compute a 1D Convolution.

        Args:
        ----
            ctx : Context
            input : batch x in_channel x width
            weight : out_channel x in_channel x k_width

        Returns:
        -------
            batch x out_channel x width

        """
        ctx.save_for_backward(input, weight)

        output = Conv1dFun.forward_inner((input.shape[0], weight.shape[0], input.shape[2]), input, weight, reversed=False)
        
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the backward pass of a 1D Convolution.
        
        Args:
        ----
            ctx : Context
            grad_output : gradient of the output

        Returns:
        -------
            Tuple[Tensor, Tensor]: gradients for input and weight
            
        """
        input, weight = ctx.saved_values

        new_input = input.permute(1, 0, 2)
        new_grad_output = grad_output.permute(1, 0, 2)
        grad_weight = Conv1dFun.forward_inner((weight.shape[1], weight.shape[0], weight.shape[2]), new_input, new_grad_output, reversed=False)
        grad_weight = grad_weight.permute(1, 0, 2)

        new_weight = weight.permute(1, 0, 2)
        grad_input = Conv1dFun.forward_inner(input.shape, grad_output, new_weight, reversed=True)

        return grad_input, grad_weight

class Conv2dFun(Function):
    @staticmethod
    def forward_inner(
        output_shape: UserShape, input: Tensor, weight: Tensor, reversed: bool = False
    ) -> Tensor:
        """Compute the inner forward pass of a 2D Convolution.

        Args:
        ----
            output_shape (UserShape): desired shape of output tensor
            input (Tensor): batch x in_channel x height x width
            weight (Tensor): out_channel x in_channel x k_height x k_width
            reversed (bool): whether to compute reverse convolution

        Returns:
        -------
            Tensor: batch x out_channel x height x width

        """
        batch, in_channels, h, w = input.shape
        out_channels, in_channels2, kh, kw = weight.shape
        assert in_channels == in_channels2, "Input and weight channels do not match."

        output = input.zeros(output_shape)
        THREADS_PER_BLOCK = 16

        blockspergrid = ((w + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK, (h + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK, out_channels)
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        cuda_conv2d[blockspergrid, threadsperblock](*output.tuple(), output.size, *input.tuple(), *weight.tuple(), reversed)
        
        return output

    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Compute a 2D Convolution.

        Args:
        ----
            ctx : Context
            input : batch x in_channel x height x width
            weight : out_channel x in_channel x k_height x k_width

        Returns:
        -------
            batch x out_channel x height x width

        """
        ctx.save_for_backward(input, weight)
        output_shape = (input.shape[0], weight.shape[0], input.shape[2], input.shape[3])
        output = Conv2dFun.forward_inner(output_shape, input, weight, reversed=False)
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the backward pass of a 2D Convolution.
        
        Args:
        ----
            ctx : Context
            grad_output : gradient of the output

        Returns:
        -------
            Tuple[Tensor, Tensor]: gradients for input and weight
            
        """
        input, weight = ctx.saved_values

        new_input = input.permute(1, 0, 2, 3)
        new_grad_output = grad_output.permute(1, 0, 2, 3)
        grad_weight = Conv2dFun.forward_inner((weight.shape[1], weight.shape[0], weight.shape[2], weight.shape[3]), new_input, new_grad_output, reversed=False)
        grad_weight = grad_weight.permute(1, 0, 2, 3)

        new_weight = weight.permute(1, 0, 2, 3)
        grad_input = Conv2dFun.forward_inner(input.shape, grad_output, new_weight, reversed=True)

        return grad_input, grad_weight

conv1d = Conv1dFun.apply
conv2d = Conv2dFun.apply