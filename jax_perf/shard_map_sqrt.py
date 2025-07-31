import jax.numpy as jnp
import jax
from jax.sharding import PartitionSpec as P, NamedSharding

mesh = jax.make_mesh((jax.device_count(), ), ('axis', ))


x = jax.device_put(jnp.ones((4096, 4096)), NamedSharding(mesh, P('axis')))
y = jax.device_put(jnp.ones((4096, 4096)), NamedSharding(mesh, P('axis'))) / 100


def f(x, y):
  # return jax.lax.rsqrt(x + y)
  return jnp.sqrt(x + y)

  
f_shard_mapped = jax.shard_map(
  f, 
  mesh=mesh,
  in_specs=(P('axis'), P('axis')),
  out_specs=P('axis'))

print(f_shard_mapped(x, y))