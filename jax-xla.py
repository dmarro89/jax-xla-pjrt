import jax
import jax.numpy as jnp
import time
from computations import intensive_computation

# Initialize a large random array
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (10000, 10000), dtype=jnp.float32)

# Measure execution time without JIT (no XLA)
start_time = time.time()
result_no_jit = intensive_computation(x).block_until_ready()  # Ensure synchronous execution
print("Execution time without XLA (no JIT):", time.time() - start_time, "seconds")

# JIT-compile the function to enable XLA optimization
compiled_intensive_computation = jax.jit(intensive_computation)

# Measure execution time with JIT (using XLA)
start_time = time.time()
result_jit = compiled_intensive_computation(x).block_until_ready()  # Ensure synchronous execution
print("Execution time with XLA (JIT):", time.time() - start_time, "seconds")
