import functools
import time
import jax.numpy as jnp
import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu


def matmul_kernel(x_ref, y_ref, z_ref, scratch):
    print(f'prog ids {pl.program_id(0)}, {pl.program_id(1)}, {pl.program_id(2)}')
    @pl.when(pl.program_id(2) == 0)
    def _():
        scratch[...] = jnp.zeros_like(scratch)

    out = jax.lax.dot(x_ref[...], y_ref[...], preferred_element_type=jnp.float32.dtype)
    scratch[...] += out.astype(jnp.bfloat16.dtype)

    @pl.when(pl.program_id(2) == pl.num_programs(2) - 1)
    def _():
        z_ref[...] = scratch[...]

@functools.partial(jax.jit, static_argnames=('width',))
def matmul(x, y, width = 4):
    assert x.shape[1] == y.shape[0]

    A, B = x.shape
    B, C = y.shape

    # v6e smem = 32M
    # (8, 128) is the register size
    #

    gx, gy, gz = (A // 512, B // 2048, C // 2048)

    kernel = pl.pallas_call(
        matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((x.shape[0], y.shape[1]), x.dtype),
        grid=(gx, gy, gz),
        out_specs = pl.BlockSpec(
            (A // gx, C // gz), lambda i, j, k: (i, j),
        ),
        in_specs=(
            pl.BlockSpec(
               (A//gx, B//gy), lambda i, j, k: (i, k),
            ),
            pl.BlockSpec(
               (B//gy, C//gz), lambda i, j, k: (k, j),
            )
        ),
        scratch_shapes=(pltpu.MemorySpace.VMEM(shape=(A//gx, C//gz), dtype=jnp.bfloat16.dtype), ),
        compiler_params=pltpu.TPUCompilerParams(
            # arbitrary means sequential
            dimension_semantics=('parallel', 'parallel', 'arbitrary')
        )
    )

    return kernel(x, y)


def main(size=4096):
    key = jax.random.key(0)
    x = jax.random.normal(key, (size, size), dtype=jnp.bfloat16.dtype)
    key = jax.random.key(1)
    y = jax.random.normal(key, (size, size), dtype=jnp.bfloat16.dtype)

    jax.block_until_ready((x, y))

    total = 10000
    for i in range(3):
        start = time.perf_counter()
        z = matmul(x, y, width=64)
        z.block_until_ready()
        end = time.perf_counter()
        print(f'{i} time is {end - start :4f}')
        total = min(total, end - start)
    throughput = 2 * size ** 3 / total
    theoretical = 918 * (1 << 40) # TFlops
    print(f'Flops / s: {2 * size ** 3 / total / (1 << 40) :4f} TFLOPs')
    print(f'MFU: {throughput / theoretical :4f}')


if __name__ == '__main__':
    import fire
    fire.Fire(main)
