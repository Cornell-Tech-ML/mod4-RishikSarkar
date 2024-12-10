import pytest
from hypothesis import given, settings

import minitorch
from minitorch import Tensor

from .tensor_strategies import tensors

import numba
import numpy
import random
import time

try:
    import numba.cuda

    CUDA_AVAILABLE = numba.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False


@pytest.mark.task4_1
def test_conv1d_simple() -> None:
    t = minitorch.tensor([0, 1, 2, 3]).view(1, 1, 4)
    t.requires_grad_(True)
    t2 = minitorch.tensor([[1, 2, 3]]).view(1, 1, 3)
    out = minitorch.Conv1dFun.apply(t, t2)

    assert out[0, 0, 0] == 0 * 1 + 1 * 2 + 2 * 3
    assert out[0, 0, 1] == 1 * 1 + 2 * 2 + 3 * 3
    assert out[0, 0, 2] == 2 * 1 + 3 * 2
    assert out[0, 0, 3] == 3 * 1


@pytest.mark.task4_1
@given(tensors(shape=(1, 1, 6)), tensors(shape=(1, 1, 4)))
def test_conv1d(input: Tensor, weight: Tensor) -> None:
    print(input, weight)
    minitorch.grad_check(minitorch.Conv1dFun.apply, input, weight)


@pytest.mark.task4_1
@given(tensors(shape=(2, 2, 6)), tensors(shape=(3, 2, 2)))
@settings(max_examples=50)
def test_conv1d_channel(input: Tensor, weight: Tensor) -> None:
    minitorch.grad_check(minitorch.Conv1dFun.apply, input, weight)


@pytest.mark.task4_2
@given(tensors(shape=(1, 1, 6, 6)), tensors(shape=(1, 1, 2, 4)))
def test_conv(input: Tensor, weight: Tensor) -> None:
    minitorch.grad_check(minitorch.Conv2dFun.apply, input, weight)


@pytest.mark.task4_2
@given(tensors(shape=(2, 1, 6, 6)), tensors(shape=(1, 1, 2, 4)))
@settings(max_examples=10)
def test_conv_batch(input: Tensor, weight: Tensor) -> None:
    minitorch.grad_check(minitorch.Conv2dFun.apply, input, weight)


@pytest.mark.task4_2
@given(tensors(shape=(2, 2, 6, 6)), tensors(shape=(3, 2, 2, 4)))
@settings(max_examples=10)
def test_conv_channel(input: Tensor, weight: Tensor) -> None:
    minitorch.grad_check(minitorch.Conv2dFun.apply, input, weight)


@pytest.mark.task4_2
def test_conv2() -> None:
    t = minitorch.tensor([[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]).view(
        1, 1, 4, 4
    )
    t.requires_grad_(True)

    t2 = minitorch.tensor([[1, 1], [1, 1]]).view(1, 1, 2, 2)
    t2.requires_grad_(True)
    out = minitorch.Conv2dFun.apply(t, t2)
    out.sum().backward()

    minitorch.grad_check(minitorch.Conv2dFun.apply, t, t2)


# CUDA Tests
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
@pytest.mark.task4_4b
def test_conv1d_cuda() -> None:
    """Test 1D convolution implementation matches between CPU and CUDA"""
    SAMPLE_CASES = [
        ((1, 1, 8), (1, 1, 3)),  # Basic case
        ((2, 2, 10), (2, 2, 4)),  # Multi-channel
        ((4, 3, 12), (3, 3, 5)),  # Larger channels
        ((8, 4, 16), (4, 4, 6)),  # More complex
        ((16, 8, 20), (8, 8, 7)),  # Large case
    ]

    for input_shape, weight_shape in SAMPLE_CASES:
        print(
            f"\nTesting 1D Conv - Input shape: {input_shape}, Weight shape: {weight_shape}"
        )

        input_data = numpy.array(
            [random.random() * 2 - 1 for _ in range(numpy.prod(input_shape))]
        )

        weight_data = numpy.array(
            [random.random() * 2 - 1 for _ in range(numpy.prod(weight_shape))]
        )

        x = Tensor.make(input_data, input_shape, backend=minitorch.SimpleBackend)
        w = Tensor.make(weight_data, weight_shape, backend=minitorch.SimpleBackend)

        cpu_start = time.perf_counter()
        cpu_output = minitorch.Conv1dFun.apply(x, w)
        cpu_end = time.perf_counter()
        cpu_time = cpu_end - cpu_start

        cuda_start = time.perf_counter()
        cuda_output = minitorch.cuda_conv.Conv1dFun.apply(x, w)
        cuda_end = time.perf_counter()
        cuda_time = cuda_end - cuda_start

        print("CPU Output Shape:", cpu_output.shape)
        print("CUDA Output Shape:", cuda_output.shape)

        print("\nExecution Times:")
        print(f"CPU Time: {cpu_time:.6f} seconds")
        print(f"CUDA Time: {cuda_time:.6f} seconds")
        print(f"Speedup: {cpu_time/cuda_time:.2f}x")

        print("\nFirst few output values:")
        print("CPU:", cpu_output._tensor._storage[:5])
        print("CUDA:", cuda_output._tensor._storage[:5])

        max_diff = numpy.max(
            numpy.abs(cpu_output._tensor._storage - cuda_output._tensor._storage)
        )
        print(f"Max difference between CPU and CUDA outputs: {max_diff:.6f}")

        numpy.testing.assert_allclose(
            cpu_output._tensor._storage,
            cuda_output._tensor._storage,
            rtol=1e-2,
            atol=1e-2,
        )

        minitorch.grad_check(minitorch.cuda_conv.Conv1dFun.apply, x, w)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
@pytest.mark.task4_4b
def test_conv2d_cuda() -> None:
    """Test 2D convolution implementation matches between CPU and CUDA"""
    SAMPLE_CASES = [
        ((1, 1, 8, 8), (1, 1, 3, 3)),  # Basic case
        ((2, 2, 12, 12), (2, 2, 4, 4)),  # Multi-channel
        ((4, 4, 16, 16), (4, 4, 5, 5)),  # Larger size
        ((8, 8, 20, 20), (8, 8, 4, 4)),  # Complex case
        ((16, 16, 24, 24), (16, 16, 3, 3)),  # Large case
    ]

    for input_shape, weight_shape in SAMPLE_CASES:
        print(
            f"\nTesting 2D Conv - Input shape: {input_shape}, Weight shape: {weight_shape}"
        )

        input_data = numpy.array(
            [random.random() * 2 - 1 for _ in range(numpy.prod(input_shape))]
        )

        weight_data = numpy.array(
            [random.random() * 2 - 1 for _ in range(numpy.prod(weight_shape))]
        )

        x = Tensor.make(input_data, input_shape, backend=minitorch.SimpleBackend)
        w = Tensor.make(weight_data, weight_shape, backend=minitorch.SimpleBackend)

        cpu_start = time.perf_counter()
        cpu_output = minitorch.Conv2dFun.apply(x, w)
        cpu_end = time.perf_counter()
        cpu_time = cpu_end - cpu_start

        cuda_start = time.perf_counter()
        cuda_output = minitorch.cuda_conv.Conv2dFun.apply(x, w)
        cuda_end = time.perf_counter()
        cuda_time = cuda_end - cuda_start

        print("CPU Output Shape:", cpu_output.shape)
        print("CUDA Output Shape:", cuda_output.shape)

        print("\nExecution Times:")
        print(f"CPU Time: {cpu_time:.6f} seconds")
        print(f"CUDA Time: {cuda_time:.6f} seconds")
        print(f"Speedup: {cpu_time/cuda_time:.2f}x")

        print("\nFirst few output values:")
        print("CPU:", cpu_output._tensor._storage[:5])
        print("CUDA:", cuda_output._tensor._storage[:5])

        max_diff = numpy.max(
            numpy.abs(cpu_output._tensor._storage - cuda_output._tensor._storage)
        )
        print(f"Max difference between CPU and CUDA outputs: {max_diff:.6f}")

        numpy.testing.assert_allclose(
            cpu_output._tensor._storage,
            cuda_output._tensor._storage,
            rtol=1e-2,
            atol=1e-2,
        )

        minitorch.grad_check(minitorch.cuda_conv.Conv2dFun.apply, x, w)
