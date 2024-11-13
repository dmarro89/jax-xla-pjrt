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
# Measure execution time on CPU
cpu_device = jax.devices("cpu")[0]
with jax.default_device(cpu_device):
    start_time = time.time()
    result_cpu = compiled_fn(x).block_until_ready()
    print("Execution time on CPU:", time.time() - start_time, "seconds")
