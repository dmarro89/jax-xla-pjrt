import jax
import jax.numpy as jnp
import time
from computations import intensive_computation

# Initialize a large random array
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (10000, 10000), dtype=jnp.float32)

# JIT-compile the function for XLA optimization
compiled_fn = jax.jit(intensive_computation)

print("Devices available:", jax.devices()) 
# Measure execution time on GPU
gpu_device = jax.devices("METAL")[0]
with jax.default_device(gpu_device):
    start_time = time.time()
    result_gpu = compiled_fn(x).block_until_ready()
    print("Execution time on GPU:", time.time() - start_time, "seconds")
