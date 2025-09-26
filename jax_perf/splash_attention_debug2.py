import time
from typing import Any, Optional, Tuple, Union
import math
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask
from jax.experimental.pallas.ops.tpu import flash_attention
from jax.tree_util import register_pytree_node

from flax import linen as nn


def create_kernel_blocks(
    query: jax.Array,
    key: jax.Array,
    q_block_size: Optional[int] = None,
    kv_block_size: Optional[int] = None,
    q_block_repeats: Optional[int] = None,
    kv_block_repeats: Optional[int] = None,
    fuse_if_possible: bool = True,
    ensure_block_sizes: bool = True,
) -> splash_attention_kernel.BlockSizes:
    seq_factor = 1 # self.sequence_block_factor

    if q_block_size is None or kv_block_size is None:
        dtype_block_factor = 1 # bfloat16 is 1
        block_size = int(512 * float(dtype_block_factor) / float(seq_factor))

        if q_block_size is None:
            q_block_size = block_size
        if kv_block_size is None:
            kv_block_size = block_size

    if q_block_repeats is None:
        q_block_repeats = 1
    if kv_block_repeats is None:
        kv_block_repeats = 1

    q_seq = query.shape[1] if query.ndim == 3 else query.shape[2]
    kv_seq = key.shape[1] if key.ndim == 3 else key.shape[2]

    q_seq = q_seq // seq_factor
    kv_seq = kv_seq // seq_factor

    layout = (
        splash_attention_kernel.QKVLayout.SEQ_MINOR
        if seq_factor == 1
        else splash_attention_kernel.QKVLayout.HEAD_DIM_MINOR
    )

    # FwD
    block_q: int = q_block_size
    block_kv: int = kv_block_size * kv_block_repeats
    block_kv_compute: int = kv_block_size

    # BwD - with respect to kv
    block_q_dkv: int = q_block_size
    block_kv_dkv: int = kv_block_size * kv_block_repeats
    block_kv_dkv_compute: int = kv_block_size

    # BwD - with respect to q
    block_q_dq: Optional[int] = None if fuse_if_possible else q_block_size
    block_kv_dq: Optional[int] = None if fuse_if_possible else kv_block_size

    if ensure_block_sizes:
        # Ensure that the block sizes do not exceed the sequence lengths
        block_q = min(block_q, q_seq)
        block_kv = min(block_kv, kv_seq)
        block_kv_compute = min(block_kv_compute, kv_seq)
        block_q_dkv = min(block_q_dkv, q_seq)
        block_kv_dkv = min(block_kv_dkv, kv_seq)
        block_kv_dkv_compute = min(block_kv_dkv_compute, kv_seq)
        if not fuse_if_possible:
            block_q_dq = min(block_q_dq, q_seq)  # type: ignore
            block_kv_dq = min(block_kv_dq, kv_seq)  # type: ignore

    block_sizes = splash_attention_kernel.BlockSizes(
        block_q=block_q,
        block_kv=block_kv,
        block_kv_compute=block_kv_compute,
        block_q_dkv=block_q_dkv,
        block_kv_dkv=block_kv_dkv,
        block_kv_dkv_compute=block_kv_dkv_compute,
        block_q_dq=block_q_dq,
        block_kv_dq=block_kv_dq,
        use_fused_bwd_kernel=fuse_if_possible,
        q_layout=layout,
        k_layout=layout,
        v_layout=layout,
    )
    #print('Splash block size are:', block_sizes)
    return block_sizes


class TimeMask(splash_attention_mask._ComputableMask):
  """Lazy causal mask, prevents the model from attending to future tokens.

  Attributes:
    offset: Offset of q start wrt kv. A positive offset shifts the bottom
      triangle upward, a negative one shifts it downward. A negative offset
      makes the first 'offset' rows of the attention matrix all 0s which leads
      to undefined softmax.
  """

  offset: int

  def __init__(
      self,
      shape: tuple[int, int],
      timestamp: jax.Array,
      offset: int = 0,
      shard_count: int = 1,
  ):
    self.timestamp = timestamp
    self.offset = offset

    def causal_mask_function(q_ids, kv_ids):
      # When evaluating the mask in _process_mask we typically work with numpy
      # array views.
      # Avoid the addition when possible to avoid instantiating an actual array.
      print('q_ids', q_ids, q_ids.shape)
      print('kv_ids', kv_ids, kv_ids.shape)
      # Got shape q_id is [512, 1] as numpy array
      # got shape kv_ids is [1, 512] as numpy array
      if isinstance(q_ids, np.ndarray):
        q_rank = self.timestamp[q_ids]
        kv_rank = self.timestamp[kv_ids]
        return q_rank >= kv_rank
      else:
        # Got shape q_id is [512, 512] as JitTracer
        # got shape kv_ids is [512, 512] as JitTracer
        q_rank = jnp.array(self.id_to_rank)[q_ids]
        kv_rank = jnp.array(self.id_to_rank)[kv_ids]

      return q_ids // 3 >= kv_ids //3
    #   if self.offset == 0:
    #     return q_ids >= kv_ids
    #   else:
    #     return q_ids + self.offset >= kv_ids

    mask_function = causal_mask_function

    super().__init__(
        shape=shape,
        mask_function=mask_function,
        shard_count=shard_count,
    )

  def __eq__(self, other: object):
    if not isinstance(other, type(self)):
      return NotImplemented

    return (
        self.shape == other.shape
        and self.offset == other.offset
        and np.array_equal(self.q_sequence, other.q_sequence)
    )

  def __hash__(self):
    return hash((
        type(self),
        self.shape,
        self.offset,
        self.q_sequence.tobytes() if self.q_sequence is not None else None,
    ))

def splash_single_device(query, key, value, timestamp):
    seq = timestamp.shape[-1]
    mha_mask = TimeMask((seq, seq), timestamp)

    block_sizes = create_kernel_blocks(query, key)

    kernel = splash_attention_kernel.make_splash_mha(
        mask=splash_attention_mask.MultiHeadMask([mha_mask]),
        head_shards=1,
        q_seq_shards=1,
        block_sizes=block_sizes,
        attn_logits_soft_cap=None,
    )

    scale_factor = 1 / math.sqrt(query.shape[-1])

    output: jax.Array = jax.vmap(kernel)(  # type: ignore
        query * scale_factor,  # splash attention does not support scaling, so we scale it here
        key,
        value,
        segment_ids=None,
    )
    return output


def main():
    num_devices = jax.device_count()
    mesh = jax.make_mesh(
        (1, num_devices,1,1),
        ("batch", "heads", "seq", "depth"),
    )
    head_sharding = jax.sharding.PartitionSpec(
        "batch", "heads", "seq", None
    )

    batch_size = 4
    seq = 8192
    heads = 16
    head_dim = 128

    q = jax.random.normal(
        jax.random.PRNGKey(0),
        (batch_size, heads, seq, head_dim),
    )
    k = jax.random.normal(
        jax.random.PRNGKey(1),
        (batch_size, heads, seq, head_dim),
    )
    v = jax.random.normal(
        jax.random.PRNGKey(2),
        (batch_size, heads, seq, head_dim),
    )

    timestamp = jax.random.normal(
        jax.random.PRNGKey(3),
        (1, seq)
    )

    @jax.jit
    def flash(q, k, v):
      block_sizes = flash_attention.BlockSizes(
        block_b=min(2, q.shape[0]),
        block_q=min(512, q.shape[2]),
        block_k_major=min(512, k.shape[2]),
        block_k=min(512, k.shape[2]),
        block_q_major_dkv=min(512, q.shape[2]),
        block_k_major_dkv=min(512, k.shape[2]),
        block_k_dkv=min(512, k.shape[2]),
        block_q_dkv=min(512, q.shape[2]),
        block_k_major_dq=min(512, k.shape[2]),
        block_k_dq=min(256, k.shape[2]),
        block_q_dq=min(1024, q.shape[2]),
      )

      wrap_flash_attention = shard_map(
          flash_attention.flash_attention,
          mesh=mesh,
          in_specs=(
              head_sharding,
              head_sharding,
              head_sharding,
          ),
          out_specs=head_sharding,
          check_rep=False,
      )
      return wrap_flash_attention(q, k, v)

    for i in range(3):
      jax.block_until_ready((q, k, v))
      start = time.perf_counter()
      static_res = flash(q, k, v)
      jax.block_until_ready(static_res)
      end = time.perf_counter()
      print('flash', i, end - start)


    # Splash
    replicated = jax.sharding.PartitionSpec()
    wrap_splash = jax.jit(shard_map(
        splash_single_device,
        mesh=mesh,
        in_specs=(
            head_sharding,
            head_sharding,
            head_sharding,
            replicated
        ),
        out_specs=head_sharding,
        check_rep=False,
    ))

    print('====')
    for i in range(3):
      jax.block_until_ready((q, k, v))
      start = time.perf_counter()
      static_res = wrap_splash(q, k, v, timestamp)
      jax.block_until_ready(static_res)
      end = time.perf_counter()
      print('splash', i, end - start)


if __name__ == "__main__":
    main()
