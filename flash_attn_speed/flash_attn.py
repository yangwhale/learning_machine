import time
import torch.nn.functional as F
import torch
import torch_xla
import torch_xla.core.xla_model as xm
from torch_xla.core import xla_builder as xb
from torch_xla.experimental.custom_kernel import flash_attention as tpu_flash_attention
from torch_xla.experimental.custom_kernel import FlashAttention as TpuFlashAttention
from torch_xla.experimental.assume_pure import assume_pure
from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention as jax_flash_attention, SegmentIds, BlockSizes

from flax.core.frozen_dict import FrozenDict



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


    batch = 1
    num_head = 5
    seqlen = 16384
    head_dim = 128

    q = torch.randn((batch, num_head, seqlen, head_dim), dtype=torch.bfloat16, device=torch_xla.device())
    k = torch.randn((batch, num_head, seqlen, head_dim), dtype=torch.bfloat16, device=torch_xla.device())
    v = torch.randn((batch, num_head, seqlen, head_dim), dtype=torch.bfloat16, device=torch_xla.device())

    torch_xla.sync()
    xm.wait_device_ops()


    for i in range(20):
        start = time.perf_counter()
        x = xb.call_jax(jax_flash_attention, (q, k, v), {'block_sizes': block_sizes})
        torch_xla.sync()
        xm.wait_device_ops()
        end = time.perf_counter()
        print(f"round {i} flash_attn time: {end - start}")


def main2():
    default_block_sizes = {
        "block_q": 4096,
        "block_k_major": 1024,
        "block_k": 512,
        "block_b": 2,
        "block_q_major_dkv": 2048,
        "block_k_major_dkv": 512,
        "block_q_dkv": 2048,
        "block_k_dkv": 512,
        "block_q_dq": 2048,
        "block_k_dq": 256,
        "block_k_major_dq": 512,
    }

    TpuFlashAttention.DEFAULT_BLOCK_SIZES = default_block_sizes

    batch = 1
    num_head = 5
    seqlen = 16384
    head_dim = 128

    q = torch.randn((batch, num_head, seqlen, head_dim), dtype=torch.bfloat16, device=torch_xla.device())
    k = torch.randn((batch, num_head, seqlen, head_dim), dtype=torch.bfloat16, device=torch_xla.device())
    v = torch.randn((batch, num_head, seqlen, head_dim), dtype=torch.bfloat16, device=torch_xla.device())

    x = tpu_flash_attention(q, k, v)
    torch_xla.sync()
    xm.wait_device_ops()
    res = []
    start = time.perf_counter()
    for i in range(20):
        x = tpu_flash_attention(q, k, v)
        res.append(x)
        torch_xla.sync()
        #print(f"round {i} flash_attn time: {end - start}")
    xm.wait_device_ops()
    end = time.perf_counter()

    print('avg of 20 round', (end - start)/20)

if __name__ == '__main__':
    main()
