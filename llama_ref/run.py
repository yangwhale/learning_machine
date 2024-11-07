import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, NamedSharding
import torch
import model
import model_with_scan
import train
import torch_xla2
from torch_xla2.ops import mappings
import custom_mesh
from jax.sharding import Mesh


sharding_map_original = {
  "freqs_cis" : (), #  torch.complex64 (2048, 64)
  "tok_embeddings.weight" : ('tp', 'fsdp'), #  torch.float32 (vocab_size, 4096)
  "layers.*.attention.wo.weight" : ('fsdp', 'tp'), #  torch.int8 (4096, 4096)
  "layers.*.attention.wq.weight" : ('tp', 'fsdp'), #  torch.int8 (4096, 4096)
  "layers.*.attention.wk.weight" : ('tp', 'fsdp'), #  torch.int8 (4096, 4096)
  "layers.*.attention.wv.weight" : ('tp', 'fsdp'), #  torch.int8 (4096, 4096)
  "layers.*.feed_forward.w1.weight" : ('tp', 'fsdp'), #  torch.float32 (11008, 4096)
  "layers.*.feed_forward.w2.weight" : ('fsdp', 'tp'), #  torch.float32 (4096, 11008)
  "layers.*.feed_forward.w3.weight": ('tp', 'fsdp'), #  torch.float32 (11008, 4096)
  "layers.*.attention_norm.weight" : ('fsdp', ), #  torch.float32 (4096,)
  "layers.*.ffn_norm.weight" : ('fsdp', ), #  torch.float32 (4096,)
  "norm.weight" : ('fsdp', ), #  torch.float32 (4096,)
  "output.weight" : ('tp', 'fsdp'), #  torch.float32 (vocab_size, 4096)
}

sharding_map_scan = {
  "freqs_cis" : (), #  torch.complex64 (2048, 64)
  "tok_embeddings.weight" : ('tp', 'fsdp'), #  torch.float32 (vocab_size, 4096)
  "layers.params.attention___wo___weight" : (None, 'fsdp', 'tp'), #  torch.int8 (n, 4096, 4096)
  "layers.params.attention___wq___weight" : (None, 'tp', 'fsdp'), #  torch.int8 (n, 4096, 4096)
  "layers.params.attention___wk___weight" : (None, 'tp', 'fsdp'), #  torch.int8 (n, 4096, 4096)
  "layers.params.attention___wv___weight" : (None, 'tp', 'fsdp'), #  torch.int8 (n, 4096, 4096)
  "layers.params.feed_forward___w1___weight" : (None, 'tp', 'fsdp'), #  torch.float32 (n, 11008, 4096)
  "layers.params.feed_forward___w2___weight" : (None, 'fsdp', 'tp'), #  torch.float32 (n, 4096, 11008)
  "layers.params.feed_forward___w3___weight": (None, 'tp', 'fsdp'), #  torch.float32 (n, 11008, 4096)
  "layers.params.attention_norm___weight" : (None, 'fsdp', ), #  torch.float32 (n, 4096,)
  "layers.params.ffn_norm___weight" : (None, 'fsdp', ), #  torch.float32 (n, 4096,)
  "norm.weight" : ('fsdp', ), #  torch.float32 (4096,)
  "output.weight" : ('tp', 'fsdp'), #  torch.float32 (vocab_size, 4096)
}


def _process_sharding_name(name):
    """Replace integers in param name with *.

  Presumably all layers should have the same sharding.
  """

    def is_integer(t):
        try:
            int(t)
            return True
        # pylint: disable-next=all
        except:  # noqa: E722
            return False

    tokens = name.split(".")
    for i, t in enumerate(tokens):
        if is_integer(t):
            tokens[i] = "*"
    return ".".join(tokens)



def create_sharded_weights(model, mesh, sharding_map):
    res = {}
    for name, weight_meta in model.state_dict().items():
        sharding_spec = sharding_map.get(_process_sharding_name(name))
        if sharding_spec is None:
            print('Skipping weight:', name)
            continue
        sharding = NamedSharding(mesh, P(*sharding_spec))
        with jax.default_device(jax.devices('cpu')[0]):
            weight_torch = torch.randn(
              weight_meta.shape,
              dtype=weight_meta.dtype)
            weight_jax = torch_xla2.default_env().to_xla(weight_torch).jax()
        #print(name, weight.shape, weight.dtype)
        res[name] = jax.make_array_from_callback(
          weight_jax.shape, sharding, lambda a: weight_jax[a]
        )
    return res

def sharded_device_put(tensor, sharding):
    num_global_devices = jax.device_count()
    num_local_devices = jax.local_device_count()
    if isinstance(tensor, tuple):
        return tuple(sharded_device_put(t, sharding) for t in tensor)

    if num_global_devices == num_local_devices:
        return jax.device_put(tensor, sharding)

    shape = tensor.shape
    x_split = [jax.device_put(tensor[i], device) for device, i in sharding.addressable_devices_indices_map(shape).items()]
    return jax.make_array_from_single_device_arrays(shape, sharding, x_split)



def main(
  model_type: str='8B',
  lr: float=0.001,
  tp: int=4,
  seqlen: int = 2048,
  use_scan: bool = True,
  use_custom_mesh: bool = False,
  use_custom_offload: bool = True,
):
    torch.manual_seed(0)
    torch.set_default_dtype(torch.bfloat16)
    print(jax.local_devices())
    fsdp_size = len(jax.devices()) // tp

    if use_custom_mesh:
        assert len(jax.devices()) == 512
        dev_array = custom_mesh.create_custom_64x4_device_mesh(
          (64, 4), (2, 1), jax.devices()
        )
        mesh = Mesh(dev_array, ('fsdp', 'tp'))
    else:
        mesh = jax.make_mesh((fsdp_size, tp), ('fsdp', 'tp'))

    if use_custom_offload:
      policy = jax.checkpoint_policies.save_and_offload_only_these_names(
          names_which_can_be_saved=[],
          names_which_can_be_offloaded=[
              "decoder_layer_input",
              "query_proj",
              "key_proj",
              "value_proj",
              "out_proj",
          ],
          offload_src="device",
          offload_dst="pinned_host",
      )
    else:
      policy=jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims

    args = model.ModelArgs(
      **model.transformer_configs[model_type]
    )
    #args.n_layers = 2

    with torch.device('meta'):
        if use_scan:
            sharding_map = sharding_map_scan
            llama = model_with_scan.Transformer(args)
        else:
            sharding_map = sharding_map_original
            llama = model.Transformer(args)

    sharded_weights = create_sharded_weights(llama, mesh, sharding_map)
    with torch.device('cpu'):
        freqs_cis = model.precompute_freqs_cis(
            args.dim // args.n_heads,
            args.max_seq_len * 2,
            args.rope_theta,
            args.use_scaled_rope,
        ).numpy()
    sharding = NamedSharding(mesh, P()) # replicated

    env = torch_xla2.default_env()
    freqs_cis = env.j2t_iso(jax.device_put(freqs_cis, sharding))

    env.config.use_tpu_flash_attention = True
    env.config.shmap_flash_attention = True
    env._mesh = mesh


    train.train_loop(mesh, llama, sharded_weights, None, freqs_cis, lr, seqlen, policy)


if __name__ == '__main__':
    import fire
    fire.Fire(main)
