## Portfolio Optimisation on GPU

Demonstration on how to perform financial computations i.e. portfolio optimisation using both CPU (with `pandas` and `NumPy`) and GPU (with `cuDF` and `CuPy`). The goal is to compare the performance of CPU and GPU for these computations.

---

RAPIDS is a suite of GPU-accelerated data science and AI libraries from Nvidia. It is built on NVIDIA CUDA-X AI™ and includes libraries that integrate with popular data science software.

Part of the problem with pandas is that it can be slow with large datasets. That is where `cuDF-pandas comes` in. `cuDF-pandas` accelerates pandas with zero code changes and brings great speed improvements.

`cuDF-pandas` is available as an extension that requires no code changes at all. To use it, just add the following code before you import pandas.

```sh
%load_ext cudf.pandas
```
