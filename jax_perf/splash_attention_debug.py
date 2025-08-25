#FROM: https://gist.github.com/zmelumian972/7afb64ce4f3d6a070cb61c31059575d6 
# Run on v6e-8:
# dynamic 0 3.4201311960059684
# dynamic 1 0.0592643000127282
# dynamic 2 0.05915972898947075
# static 0 3.303027266985737
# static 1 0.026993509993189946
# static 2 0.026713399012805894
# static causal 0 0.5970732840069104
# static causal 1 0.013313870003912598
# static causal 2 0.0133067199785728
# flash 0 0.39458162599476054
# flash 1 0.007796349993441254
# flash 2 0.0077992400038056076


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




class SplashAttention(nn.Module):
    """Sparse flash attention implementation for JAX/Flax models."""
    mesh: jax.sharding.Mesh
    window_size: Optional[int] = None
    
    def setup(self) -> None:
        super().setup()
        if self.mesh is None:
            raise NotImplementedError("Splash attention requires a mesh to be specified for sharding.")

    def create_mask(
        self, q_seq: int, kv_seq: int, dynamic_mask: Optional[jax.Array], static_mask: Optional[np.ndarray] = None
    ) -> Union[splash_attention_mask.Mask, jax.Array]:
        """
        Creates an attention mask for the splash attention mechanism.
        This method generates a mask that defines which tokens can attend to which others.
        """
        if dynamic_mask is not None:
            if self.window_size is not None:
                raise NotImplementedError("Dynamic masks are not supported with windowed attention.")

            mask = dynamic_mask
        else:
            if isinstance(static_mask, splash_attention_mask.Mask):
                return static_mask 
            mask = splash_attention_mask.FullMask((q_seq, kv_seq))
            if self.window_size is not None:
                mask &= splash_attention_mask.LocalMask(
                    shape=(q_seq, q_seq),
                    window_size=(self.window_size, self.window_size),
                    offset=0,
                )
            if static_mask is not None:
                np_mask = splash_attention_mask.NumpyMask(static_mask)
                mask &= np_mask

        return mask

    @nn.compact
    def __call__(
        self,
        query: jax.Array,
        key: jax.Array,
        value: jax.Array,
        q_segment_ids: Optional[jax.Array] = None,
        kv_segment_ids: Optional[jax.Array] = None,
        dynamic_mask: Optional[jax.Array] = None,
        static_mask: Optional[np.ndarray] = None,
        block_sizes: Optional[Any] = None,
    ):
        if q_segment_ids is None and kv_segment_ids is not None:
            q_segment_ids = jax.numpy.ones((query.shape[0], query.shape[2]), dtype=jnp.float32)

        vmap_inner = self.splash_attention_helper
        vmap_inner = partial(vmap_inner, static_mask=static_mask)
        def static_helper(*args, **kwargs):
            # this here is to ensure self is not overrided by shard_map accidently
            return jax.vmap(vmap_inner)(*args, **kwargs)

        wrapped_splash_attention = shard_map(
            partial(
                static_helper,
                block_sizes=block_sizes,
            ),
            mesh=self.mesh,  # type: ignore
            in_specs=(
                self._qkvo_partition_spec,
                self._qkvo_partition_spec,
                self._qkvo_partition_spec,
                self._segment_ids_spec,
                self._segment_ids_spec,
                jax.sharding.PartitionSpec(self._batch_partitions, None, None),
            ),
            out_specs=self._qkvo_partition_spec,
            check_rep=False,
        )

        return wrapped_splash_attention(query, key, value, q_segment_ids, kv_segment_ids, dynamic_mask)

    def splash_attention_helper(
        self,
        query: jax.Array,
        key: jax.Array,
        value: jax.Array,
        q_segment_ids: Optional[jax.Array],
        kv_segment_ids: Optional[jax.Array],
        dynamic_mask: Optional[jax.Array],
        static_mask: Optional[np.ndarray],
        block_sizes: Optional[splash_attention_kernel.BlockSizes],  # type: ignore
    ) -> jax.Array:
        """Helper function to call the splash attention kernel."""
        if q_segment_ids is not None and kv_segment_ids is not None:
            segment_ids = splash_attention_kernel.SegmentIds(
                q=q_segment_ids.astype(jnp.float32), kv=kv_segment_ids.astype(jnp.float32)
            )
        elif q_segment_ids is None and kv_segment_ids is None:
            segment_ids = None
        else:
            raise ValueError("Both q_segment_ids and kv_segment_ids must be provided or both must be None.")

        q_seq = query.shape[1]
        kv_seq = key.shape[1]
        heads = query.shape[0]

        mask = self.create_mask(
            q_seq=q_seq,
            kv_seq=kv_seq,
            dynamic_mask=dynamic_mask,
            static_mask=static_mask,
        )

        if isinstance(mask, jax.Array):
            mha_mask = jax.numpy.expand_dims(mask, axis=0).repeat(heads, axis=0)
        else:
            mha_mask = splash_attention_mask.MultiHeadMask([mask for _ in range(heads)])

        if block_sizes is None:
            block_sizes: splash_attention_kernel.BlockSizes = self.create_kernel_blocks(query, key)

        kernel = splash_attention_kernel.make_splash_mha(
            mask=mha_mask,
            head_shards=1,
            q_seq_shards=1,
            block_sizes=block_sizes,
            attn_logits_soft_cap=None,
        )

        scale_factor = self._calculate_scale_factor(query)

        output: jax.Array = kernel(  # type: ignore
            query * scale_factor,  # splash attention does not support scaling, so we scale it here
            key,
            value,
            segment_ids=segment_ids,
        )
        return output

    def create_kernel_blocks(
        self,
        query: jax.Array,
        key: jax.Array,
        q_block_size: Optional[int] = None,
        kv_block_size: Optional[int] = None,
        q_block_repeats: Optional[int] = None,
        kv_block_repeats: Optional[int] = None,
        fuse_if_possible: bool = True,
        ensure_block_sizes: bool = True,
    ) -> splash_attention_kernel.BlockSizes:
        seq_factor = self.sequence_block_factor

        if q_block_size is None or kv_block_size is None:
            dtype_block_factor = self.dtype_block_factor
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

    def _calculate_scale_factor(self, query: jax.Array) -> float:
        """
        Calculates the scale factor for the attention mechanism based on the query tensor.

        Args:
            query (jax.Array): Query tensor used to compute the scale factor.

        Returns:
            float: The scale factor for the attention mechanism.
        """
        return 1 / math.sqrt(query.shape[-1])
    
    @property
    def is_tpu_attention(self) -> bool:
        return True

    @property
    def dtype_block_factor(self) -> int:
        """
        Returns the block factor for the data type used in the attention mechanism.

        """
        # Memory will not really allocate due to JIT
        return jnp.ones(1, dtype=self.dtype).itemsize // 2
    
    @property
    def _batch_partitions(self) -> Tuple[str, ...]:
        """
        Returns the partition spec for batch dimensions.

        This is used to define how the batch dimensions are sharded across devices.
        The default implementation returns a partition spec that assumes
        all dimensions are sharded across the mesh.

        Subclasses can override this method to provide a custom partition spec.
        """
        return ("batch",)

    @property
    def _sequence_partitions(self) -> Tuple[str, ...]:
        """
        Returns the partition spec for sequence dimensions.

        This is used to define how the sequence dimensions are sharded across devices.
        The default implementation returns a partition spec that assumes
        all dimensions are sharded across the mesh.

        Subclasses can override this method to provide a custom partition spec.
        """
        return ("seq",)

    @property
    def _qkvo_partition_spec(self) -> jax.sharding.PartitionSpec:
        """
        Returns the partition spec for query, key, value, and output tensors.

        This is used to define how the tensors are sharded across devices.
        The default implementation returns a partition spec that assumes
        all dimensions are sharded across the mesh.

        Subclasses can override this method to provide a custom partition spec.
        """
        return jax.sharding.PartitionSpec(
            self._batch_partitions,
            "heads",
            self._sequence_partitions,
            None,
        )

    @property
    def _segment_ids_spec(self) -> jax.sharding.PartitionSpec:
        """
        Returns the partition spec for segment IDs.

        This is used to define how the segment IDs are sharded across devices.
        The default implementation returns a partition spec that assumes
        all dimensions are sharded across the mesh.

        Subclasses can override this method to provide a custom partition spec.
        """
        return jax.sharding.PartitionSpec(self._batch_partitions, self._sequence_partitions)

    @property
    def sequence_block_factor(self) -> int:
        """
        Returns the sequence block factor based on the mesh shape.

        This is used to determine how the sequence dimensions are sharded across devices.
        The default implementation assumes that the sequence dimensions are sharded evenly across the mesh.
        """
        if self.mesh is None:
            return 1
        return self.mesh.shape["seq"]
    
    @property
    def dtype(self):
        return jnp.bfloat16

class CausalMask(splash_attention_mask._ComputableMask):
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
      id_to_rank: jax.Array,
      offset: int = 0,
      shard_count: int = 1,
  ):
    self.offset = offset
    self.id_to_rank = id_to_rank

    def causal_mask_function(q_ids, kv_ids):
      # When evaluating the mask in _process_mask we typically work with numpy
      # array views.
      # Avoid the addition when possible to avoid instantiating an actual array.
      #print('q_ids', q_ids, q_ids.shape)
      #print('kv_ids', kv_ids, kv_ids.shape)
      # Got shape q_id is [512, 1] as numpy array
      # got shape kv_ids is [1, 512] as numpy array
    #   if isinstance(q_ids, np.ndarray):
    #     q_rank = self.id_to_rank[q_ids]
    #     kv_rank = self.id_to_rank[kv_ids]
    #   else:
    #     # Got shape q_id is [512, 512] as JitTracer
    #     # got shape kv_ids is [512, 512] as JitTracer
    #     q_rank = jnp.array(self.id_to_rank)[q_ids]
    #     kv_rank = jnp.array(self.id_to_rank)[kv_ids]
    
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

    
def _flatten_mask(mask):
    breakpoint()
    return (mask.shape, mask.id_to_rank), None

def _unflatten_mask(aux, data):
    breakpoint()
    shape, id_to_rank = data
    return CausalMask(shape, id_to_rank)

register_pytree_node(
    CausalMask, _flatten_mask, _unflatten_mask)

def main():
    mesh = jax.make_mesh(
        (1,1,1,1),
        ("batch", "heads", "seq", "depth"),
    )
    
    splash = SplashAttention(
        mesh,
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
    
    timestep = jax.random.normal(
        jax.random.PRNGKey(3),
        (1, seq)
    )
    
    def create_timestep_mask(timestep: jax.Array) -> jax.Array:
        q_timesteps = jnp.expand_dims(timestep, axis=1)
        kv_timesteps = jnp.expand_dims(timestep, axis=2)
        timesteps_map = q_timesteps >= kv_timesteps
        timesteps_map = nn.with_logical_constraint(
            timesteps_map, ("activation_batch", "activation_norm_length", None)
        )
        return timesteps_map
    

    dynamic_mask = create_timestep_mask(timestep)
    dynamic_mask = jnp.broadcast_to(
        dynamic_mask,
        (batch_size, seq, seq)
    )
    static_mask = np.array(jax.device_get(dynamic_mask[0]))

    splash_apply = jax.jit(splash.apply)

    id_to_rank = np.arange(0, seq) // 3 # (every 3 element is one rank)
    causal_mask = CausalMask((seq, seq), id_to_rank)

    for i in range(3):
      jax.block_until_ready((q, k, v))
      start = time.perf_counter()
      dynamic_res = splash_apply({}, q, k, v, dynamic_mask=dynamic_mask)
      #dynamic_res = dynamic(q, k, v)
      jax.block_until_ready(dynamic_res)
      end = time.perf_counter()
      print('dynamic', i, end - start)
    static_res = splash.apply({}, q, k, v, static_mask=static_mask)

    @jax.jit
    def static_apply(q, k, v):
      # hardcode static_mask
      return splash.apply({}, q, k, v, static_mask=static_mask)

    splash_apply_static = jax.jit(splash.apply, static_argnames=['static_mask'])
    for i in range(3):
      jax.block_until_ready((q, k, v))
      start = time.perf_counter()
      static_res = static_apply(q, k, v)
      jax.block_until_ready(static_res)
      end = time.perf_counter()
      print('static', i, end - start)

    @jax.jit
    def static_apply_causal(q, k, v):
      # hardcode static_mask
      return splash.apply({}, q, k, v, static_mask=causal_mask)

    splash_apply_static = jax.jit(splash.apply, static_argnames=['static_mask'])
    for i in range(3):
      jax.block_until_ready((q, k, v))
      start = time.perf_counter()
      static_res = static_apply_causal(q, k, v)
      jax.block_until_ready(static_res)
      end = time.perf_counter()
      print('static causal', i, end - start)

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
      return flash_attention.flash_attention(
          q, k, v, causal=True, block_sizes=block_sizes)

      wrap_flash_attention = shard_map(
          wrap_flash_attention,
          mesh=mesh,
          in_specs=(
            splash._qkvo_partition_spec,
            splash._qkvo_partition_spec,
            splash._qkvo_partition_spec),
          out_specs=splash._qkvo_partition_spec,
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


if __name__ == "__main__":
    main()