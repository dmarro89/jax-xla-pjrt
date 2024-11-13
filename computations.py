import jax.numpy as jnp

def intensive_computation(x):
    for _ in range(10):  # Loop to increase the computational load
        x = jnp.sin(x) + jnp.cos(x) + jnp.tan(x)
        x = jnp.log1p(x ** 2)
    return jnp.sum(x)
