# MiniTorch Module 4

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module4.html

This module requires `fast_ops.py`, `cuda_ops.py`, `scalar.py`, `tensor_functions.py`, `tensor_data.py`, `tensor_ops.py`, `operators.py`, `module.py`, and `autodiff.py` from Module 3.


Additionally you will need to install and download the MNist library.

(On Mac, this may require installing the `wget` command)

```
pip install python-mnist
mnist_get_data.sh
```


* Tests:

```
python run_tests.py
```

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py minitorch/tensor_ops.py minitorch/fast_ops.py minitorch/cuda_ops.py project/parallel_check.py tests/test_tensor_general.py

---

### Task 4.5 Training Logs

Training logs are located at:
- [logs/mnist.txt](https://github.com/Cornell-Tech-ML/mod4-RishikSarkar/blob/master/logs/mnist.txt)
  - MNIST digit classification trained for 50 epochs, reaching 16/16 validation accuracy
- [logs/sentiment.txt](https://github.com/Cornell-Tech-ML/mod4-RishikSarkar/blob/master/logs/sentiment.txt)
  - SST2 sentiment classification trained for 250 epochs, reaching maximum validation accuracy of 75% at epoch 34

---

### Task 4.4b Extra Credit: CUDA Convolution Tests

Implementation is located in `minitorch/cuda_conv.py` with test cases in `tests/test_conv.py` (marked with @pytest.mark.task4_4b).

Test execution log:
```
python -m pytest tests/ -m task4_4b -s
==================================================== test session starts =====================================================
platform win32 -- Python 3.12.5, pytest-8.3.2, pluggy-1.5.0
rootdir: D:\Cornell\Academic\Fall 2024\Machine Learning Engineering\mod4-RishikSarkar
configfile: pyproject.toml
plugins: hypothesis-6.54.0, env-1.1.4
collected 132 items / 130 deselected / 2 selected

tests\test_conv.py
Testing 1D Conv - Input shape: (1, 1, 8), Weight shape: (1, 1, 3)
CPU Output Shape: (1, 1, 8)
CUDA Output Shape: (1, 1, 8)

Execution Times:
CPU Time: 0.843430 seconds
CUDA Time: 0.004435 seconds
Speedup: 190.17x

First few output values:
CPU: [-0.74562755  0.4305539  -0.49021031  1.01092123 -0.78452119]
CUDA: [-0.74562755  0.4305539  -0.49021031  1.01092123 -0.78452119]
Max difference between CPU and CUDA outputs: 0.000000

Testing 1D Conv - Input shape: (2, 2, 10), Weight shape: (2, 2, 4)
CPU Output Shape: (2, 2, 10)
CUDA Output Shape: (2, 2, 10)

Execution Times:
CPU Time: 0.003948 seconds
CUDA Time: 0.004146 seconds
Speedup: 0.95x

First few output values:
CPU: [ 0.90282487  0.14521973 -0.24045535 -0.76904464 -0.03903013]
CUDA: [ 0.90282487  0.14521973 -0.24045535 -0.76904464 -0.03903013]
Max difference between CPU and CUDA outputs: 0.000000

Testing 1D Conv - Input shape: (4, 3, 12), Weight shape: (3, 3, 5)
CPU Output Shape: (4, 3, 12)
CUDA Output Shape: (4, 3, 12)

Execution Times:
CPU Time: 0.004978 seconds
CUDA Time: 0.003716 seconds
Speedup: 1.34x

First few output values:
CPU: [-1.42079624 -1.53023112  0.50683832 -0.23198709  0.58417077]
CUDA: [-1.42079624 -1.53023112  0.50683832 -0.23198709  0.58417077]
Max difference between CPU and CUDA outputs: 0.000000

Testing 1D Conv - Input shape: (8, 4, 16), Weight shape: (4, 4, 6)
CPU Output Shape: (8, 4, 16)
CUDA Output Shape: (8, 4, 16)

Execution Times:
CPU Time: 0.003823 seconds
CUDA Time: 0.003687 seconds
Speedup: 1.04x

First few output values:
CPU: [ 0.98676888  1.28592082 -1.6003684  -1.12137863  0.02492905]
CUDA: [ 0.98676888  1.28592082 -1.6003684  -1.12137863  0.02492905]
Max difference between CPU and CUDA outputs: 0.000000

Testing 1D Conv - Input shape: (16, 8, 20), Weight shape: (8, 8, 7)
CPU Output Shape: (16, 8, 20)
CUDA Output Shape: (16, 8, 20)

Execution Times:
CPU Time: 0.004140 seconds
CUDA Time: 0.003643 seconds
Speedup: 1.14x

First few output values:
CPU: [ 1.33019852  4.18452713 -4.13704725  1.58006494 -0.36365781]
CUDA: [ 1.33019852  4.18452713 -4.13704725  1.58006494 -0.36365781]
Max difference between CPU and CUDA outputs: 0.000000
.
Testing 2D Conv - Input shape: (1, 1, 8, 8), Weight shape: (1, 1, 3, 3)
CPU Output Shape: (1, 1, 8, 8)
CUDA Output Shape: (1, 1, 8, 8)

Execution Times:
CPU Time: 0.890656 seconds
CUDA Time: 0.004491 seconds
Speedup: 198.34x

First few output values:
CPU: [ 0.9332479  -1.01079754  0.24687377 -0.68245857 -1.02310615]
CUDA: [ 0.9332479  -1.01079754  0.24687377 -0.68245857 -1.02310615]
Max difference between CPU and CUDA outputs: 0.000000

Testing 2D Conv - Input shape: (2, 2, 12, 12), Weight shape: (2, 2, 4, 4)
CPU Output Shape: (2, 2, 12, 12)
CUDA Output Shape: (2, 2, 12, 12)

Execution Times:
CPU Time: 0.004568 seconds
CUDA Time: 0.004079 seconds
Speedup: 1.12x

First few output values:
CPU: [-1.26947372  0.48015992  2.19072562  0.85154579  1.51148415]
CUDA: [-1.26947372  0.48015992  2.19072562  0.85154579  1.51148415]
Max difference between CPU and CUDA outputs: 0.000000

Testing 2D Conv - Input shape: (4, 4, 16, 16), Weight shape: (4, 4, 5, 5)
CPU Output Shape: (4, 4, 16, 16)
CUDA Output Shape: (4, 4, 16, 16)

Execution Times:
CPU Time: 0.004552 seconds
CUDA Time: 0.004061 seconds
Speedup: 1.12x

First few output values:
CPU: [-0.17333401  1.09588291 -8.1085348  -0.7769954   4.04655414]
CUDA: [-0.17333401  1.09588291 -8.1085348  -0.7769954   4.04655414]
Max difference between CPU and CUDA outputs: 0.000000

Testing 2D Conv - Input shape: (8, 8, 20, 20), Weight shape: (8, 8, 4, 4)
CPU Output Shape: (8, 8, 20, 20)
CUDA Output Shape: (8, 8, 20, 20)

Execution Times:
CPU Time: 0.005100 seconds
CUDA Time: 0.005139 seconds
Speedup: 0.99x

First few output values:
CPU: [-0.73166464  3.83488577  7.8223757  -4.74269633 -1.24189183]
CUDA: [-0.73166464  3.83488577  7.8223757  -4.74269633 -1.24189183]
Max difference between CPU and CUDA outputs: 0.000000

Testing 2D Conv - Input shape: (16, 16, 24, 24), Weight shape: (16, 16, 3, 3)
CPU Output Shape: (16, 16, 24, 24)
CUDA Output Shape: (16, 16, 24, 24)

Execution Times:
CPU Time: 0.014391 seconds
CUDA Time: 0.012748 seconds
Speedup: 1.13x

First few output values:
CPU: [ 1.0168063  -6.29142437  1.32154618 -1.65246272 -6.93053836]
CUDA: [ 1.0168063  -6.29142437  1.32154618 -1.65246272 -6.93053836]
Max difference between CPU and CUDA outputs: 0.000000
.

====================================================== warnings summary ======================================================
tests/test_conv.py::test_conv1d_practice
tests/test_conv.py::test_conv2d_practice
  D:\Cornell\Academic\Fall 2024\Machine Learning Engineering\mod4-RishikSarkar\.venv\Lib\site-packages\numba\cuda\dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_conv.py::test_conv1d_practice
tests/test_conv.py::test_conv2d_practice
  D:\Cornell\Academic\Fall 2024\Machine Learning Engineering\mod4-RishikSarkar\.venv\Lib\site-packages\numba\cuda\cudadrv\devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
    warn(NumbaPerformanceWarning(msg))

tests/test_conv.py::test_conv1d_practice
tests/test_conv.py::test_conv2d_practice
  D:\Cornell\Academic\Fall 2024\Machine Learning Engineering\mod4-RishikSarkar\.venv\Lib\site-packages\numba\cuda\dispatcher.py:536: NumbaPerformanceWarning: Grid size 2 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_conv.py::test_conv1d_practice
  D:\Cornell\Academic\Fall 2024\Machine Learning Engineering\mod4-RishikSarkar\.venv\Lib\site-packages\numba\cuda\dispatcher.py:536: NumbaPerformanceWarning: Grid size 3 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_conv.py::test_conv1d_practice
tests/test_conv.py::test_conv2d_practice
  D:\Cornell\Academic\Fall 2024\Machine Learning Engineering\mod4-RishikSarkar\.venv\Lib\site-packages\numba\cuda\dispatcher.py:536: NumbaPerformanceWarning: Grid size 4 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_conv.py::test_conv1d_practice
  D:\Cornell\Academic\Fall 2024\Machine Learning Engineering\mod4-RishikSarkar\.venv\Lib\site-packages\numba\cuda\dispatcher.py:536: NumbaPerformanceWarning: Grid size 16 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_conv.py::test_conv2d_practice
  D:\Cornell\Academic\Fall 2024\Machine Learning Engineering\mod4-RishikSarkar\.venv\Lib\site-packages\numba\cuda\dispatcher.py:536: NumbaPerformanceWarning: Grid size 32 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_conv.py::test_conv2d_practice
  D:\Cornell\Academic\Fall 2024\Machine Learning Engineering\mod4-RishikSarkar\.venv\Lib\site-packages\numba\cuda\dispatcher.py:536: NumbaPerformanceWarning: Grid size 64 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
====================================== 2 passed, 130 deselected, 12 warnings in 20.06s =======================================
```