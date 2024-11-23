import jax
from jax_plugins import my_plugin
import jax.numpy as jnp

my_plugin.initialize()
jax.config.update("jax_platforms", "pjrt_c_api_cpu_plugin")

devices = jax.devices()
print("Devices available:")
for device in devices:
    print(f"Device ID: {device.id}")
    print(f"Device Kind: {device.device_kind}")
    print(f"Platform: {device.platform}")
    print(f"Host ID: {device.host_id}")
    print(f"Process Index: {device.process_index}")
    print("-" * 50)

# Define the function f(x) = x^2 + 1
def f(x):
    return x**2 + 1

# Test values
inputs = jnp.array([0.1, 1, 3, 4, 5], dtype=jnp.float32)
expected_outputs = jnp.array([1.01, 2, 10, 17, 26], dtype=jnp.float32)

# Execute the function on each input and compare it with the expected results
print("Testing f(x) = x^2 + 1 with PJRT plugin:")
for i, x in enumerate(inputs):
    result = f(x)
    expected = expected_outputs[i]
    print(f"f({x}) = {result} (expected: {expected})")
    assert jnp.isclose(result, expected, atol=0.001), f"Test failed for input {x}: got {result}, expected {expected}"

print("All tests passed successfully.")
