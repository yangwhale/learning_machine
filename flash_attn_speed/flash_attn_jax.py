import time
import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention as jax_flash_attention, SegmentIds, BlockSizes


def main():
    block_sizes = BlockSizes(
        block_q=4096,
        block_k_major=1024,
        block_k=1024,
        block_b=1,
        block_q_major_dkv=128,
        block_k_major_dkv=128,
        block_k_dkv=128,
        block_q_dkv=128,
        block_k_major_dq=128,
        block_k_dq=128,
        block_q_dq=128,
    )

    key = jax.random.PRNGKey(123)
    batch = 1
    num_head = 5
    seq_length = 16384
    head_dim = 128

    q = jax.random.normal(key, (batch, num_head, seq_length, head_dim), dtype=jax.dtypes.bfloat16)
    key, _ = jax.random.split(key)
    k = jax.random.normal(key, (batch, num_head, seq_length, head_dim), dtype=jax.dtypes.bfloat16)
    key, _ = jax.random.split(key)
    v = jax.random.normal(key, (batch, num_head, seq_length, head_dim), dtype=jax.dtypes.bfloat16)
    jax.block_until_ready((q, k, v))

    traced = jax_flash_attention.trace(q, k, v)
    lowered = traced.lower()
    compiled = lowered.compile()
    costs = compiled.cost_analysis()

    for i in range(20):
        start = time.perf_counter()
        o = jax_flash_attention(q, k, v, block_sizes=block_sizes)
        jax.block_until_ready((o))
        end = time.perf_counter()
        tflops = (costs["flops"]) / (end - start) * 1e-12
        print('time', end - start, "tflops", tflops )

if __name__ == '__main__':
    main()
