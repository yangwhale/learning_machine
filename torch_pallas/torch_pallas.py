import functools

import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp
import numpy as np
import torchax
import torch
from torchax import interop
torchax.enable_globally()


def torch_pallas_call(kernel, *args, **kwargs):
  kernel_as_jax = interop.jax_view(kernel)
  orig_pallas_callable = pl.pallas_call(
      kernel_as_jax,
      *args,
      **kwargs,
  )
  return interop.torch_view(orig_pallas_callable)


# https://docs.jax.dev/en/latest/pallas/quickstart.html
# easiest hello world
def add_vectors_kernel(x_ref, y_ref, o_ref):
  x, y = x_ref[...], y_ref[...]
  o_ref[...] = torch.add(x, y)


  
def add_vectors(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
  return torch_pallas_call(
      add_vectors_kernel,
      out_shape=jax.ShapeDtypeStruct(x.shape, interop.jax_view(x.dtype)),
      interpret=True
  )(x, y)

print('add vector result', add_vectors(torch.randn(8, device='jax'), torch.randn(8, device='jax')))


# =====  matmul example ===
def matmul_kernel(x_ref, y_ref, z_ref, *, activation):
  z_ref[...] = activation(torch.matmul(x_ref[...], y_ref[...]))

def matmul(x: torch.Tensor, y: torch.Tensor, *, activation):
  return torch_pallas_call(
    functools.partial(matmul_kernel, activation=activation),
    out_shape=jax.ShapeDtypeStruct((x.shape[0], y.shape[1]), interop.jax_view(x.dtype)),
    grid=(2, 2),
    in_specs=[
        pl.BlockSpec((x.shape[0] // 2, x.shape[1]), lambda i, j: (i, 0)),
        pl.BlockSpec((y.shape[0], y.shape[1] // 2), lambda i, j: (0, j))
    ],
    out_specs=pl.BlockSpec(
        (x.shape[0] // 2, y.shape[1] // 2), lambda i, j: (i, j)
    ),
    interpret=True,
  )(x, y)

a = torch.randn((1024, 1024), device='jax')
b = torch.randn((1024, 1024), device='jax')


z = matmul(a, b, activation=torch.nn.functional.relu)
print('matmul result: ', z)
