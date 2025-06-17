import jax
print(jax.devices())

import jax; import jaxlib
print("jax:", jax.__version__)
print("jaxlib:", jaxlib.__version__)

import jax.numpy as jnp
import time

x = jnp.ones((10000, 10000))
start = time.time()
y = jnp.dot(x, x)
y.block_until_ready()
print("Elapsed:", time.time() - start,"s")