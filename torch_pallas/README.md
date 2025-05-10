Writing Pallas Kernels using torch ops
======================================

## Running the example:

(NOTE: can also run on CPU using pallas interpret mode, convenient for development)

```bash
pip install torch
pip install jax
pip install git+https://github.com/pytorch/xla#subdirectory=torchax
```

then,

```bash
python torch_pallas.py
```


## What is Pallas

https://docs.jax.dev/en/latest/pallas/index.html

Pallas is a *domain-specific language* (DSL) for writting custom kernels
for TPU and GPU. It is mostly similar to OpenAI's triton for writting GPU kernels.
It is implemented on top of Jax and has a tight integration. 

## Pallas in Pytorch

If one wishes to run Pallas kernels on Pytorch TPU, one can accomplish so 
via the [torch_xla Pallas support](https://docs.pytorch.org/xla/master/features/pallas.html)
However, the math within the kernel need to be expressed via Jax's operators 
(either `jax.numpy.*` or `jax.lax.*`)

Therefore, one usability concern for Pallas for PyTorch users is that now the user need
to learn semantics of Jax ops to write Pallas kernels.


## Writing Pallas with torch op.

This example shows how to enable a user to write Pallas op using torch ops.

For example, the toy example in https://docs.jax.dev/en/latest/pallas/quickstart.html
becomes:

```python
def add_vectors_kernel(x_ref, y_ref, o_ref):
  x, y = x_ref[...], y_ref[...]
  o_ref[...] = torch.add(x, y)  # <-- here I use torch.add instead of + to show case it's really a torch op

  
def add_vectors(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
  return torch_pallas_call(
      add_vectors_kernel,
      out_shape=jax.ShapeDtypeStruct(x.shape, interop.jax_view(x.dtype)),
      interpret=True
  )(x, y)

print('add vector result', add_vectors(torch.randn(8, device='jax'), torch.randn(8, device='jax')))
```

## How does it works

First let's give a overview how Pallas works normally.
On a logical level, Pallas works by defining the math over a grid, the math 
itself is written in jax ops. 
Jax's tracer (jax.make_jaxpr etc) will trace the Pallas kernel implementation and
get a jaxpr with the math part. Then, there is a compiler  that compiles that into
Mosaic IR (some MLIR IR that TPU backend understand), then this IR is send over to TPU
to execute.

### One way to implement

One way to implement this feature is to mimic what Jax-pallas has done, and try to 
replicate with torch equivalents.

For example, jaxpr is roughly equivalent to `torch.fx`; `jax.make_jaxpr` would be
`make_fx` (or `torch.export` if we want to use dynamo to trace), after the graph
is produced, then, there need a compiler similar to the above that maps that `fx` graph
into Mosaic.

### Another way to implement

Another way to implement is to use [`torchax`](https://github.com/pytorch/xla/blob/master/torchax/docs/how_it_works.md)
This framework simply defined what is every torch's operator (ATen op) represented
as Jax ops. Interestingly, defining this one way mapping of torch->jax for operators, 
induces an isomorphism (bidirectional) between Jax functions and torch functions.

Therefore, the `torch_pallas_call` can just be defined as:

```python
def torch_pallas_call(kernel, *args, **kwargs):
  kernel_as_jax = interop.jax_view(kernel)
  orig_pallas_callable = pl.pallas_call(
      kernel_as_jax,
      *args,
      **kwargs,
  )
  return interop.torch_view(orig_pallas_callable)
```