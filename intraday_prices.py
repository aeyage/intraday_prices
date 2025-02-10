import time
import numpy as np
import pandas as pd
import cudf
import cupy as cp


# Helper function to load our price data from a CSV file using pandas
def get_prices_as_pandas(prices_file):
    d = pd.read_csv(prices_file)
    d.set_index("date_time", inplace=True)
    d.index = pd.to_datetime(d.index)

    return d.bfill().ffill()

# Helper function to load the price data using cuDF
def get_prices_as_cudf(prices_file):
    c = cudf.read_csv(prices_file)
    c.set_index("date_time", inplace=True)
    c.index = cudf.to_datetime(c.index)

    return c.bfill().ffill()


# Compute optimal asset weights using pandas on the CPU
print(f"=== Pandas (CPU) Computation ===")

start_cpu = time.time()

df_pd = get_prices_as_pandas("intraday_prices.csv")
n_assets = len(df_pd.columns)

df_returns_cpu = df_pd.pct_change().dropna()
mean_returns_cpu = df_returns_cpu.mean()
cov_matrix_cpu = df_returns_cpu.cov()

inv_cov_cpu = np.linalg.inv(cov_matrix_cpu.values)
ones_cpu = np.ones((n_assets, 1))
w_cpu = inv_cov_cpu.dot(ones_cpu)
w_cpu = w_cpu / (ones_cpu.T.dot(w_cpu))

end_cpu = time.time()
cpu_elapsed = end_cpu - start_cpu

print(f"CPU elapsed time: {cpu_elapsed} seconds")
print(f"Optimal weights (first 5):\n{w_cpu[:5].flatten()}")

# Perform the same computations using cuDF and cuPY on the GPU
print(f"=== cuDF (GPU) Computation ===")

start_gpu = time.time()

df_cudf = get_prices_as_cudf("intraday_prices.csv")
n_assets = len(df_cudf.columns)

df_returns_gpu = df_cudf.pct_change().dropna()
mean_returns_gpu = df_returns_gpu.mean()
cov_matrix_gpu = df_returns_gpu.cov()

inv_cov_gpu = cp.linalg.inv(cov_matrix_gpu.values)
ones_gpu = cp.ones((n_assets, 1))
w_gpu = cp.matmul(inv_cov_gpu, ones_gpu)
w_gpu = w_gpu / (cp.matmul(ones_gpu.T, w_gpu))

end_gpu = time.time()
gpu_elapsed = end_gpu - start_gpu

print(f"GPU elapsed time: {gpu_elapsed} seconds")
print(f"Optimal weights (first 5):\n{w_gpu[:5].get()}")


# Compare the computation times between CPU and GPU
speedup = cpu_elapsed / gpu_elapsed if gpu_elapsed > 0 else float('inf')
print(f"Speedup (CPU/GPU): ~{speedup:.2f}x")
