import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, NamedSharding
import torch
import model
import train
import torch_xla2
from torch_xla2.ops import mappings


sharding_map = {
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


def get_sharding(name):
  name = _process_sharding_name(name)
  shard_axis = sharding_map[name]
  return shard_axis



def create_sharded_weights(model, mesh):
  res = {}
  for name, weight_meta in model.state_dict().items():
    sharding = NamedSharding(mesh, P(*get_sharding(name)))
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
):
  torch.manual_seed(0)
  torch.set_default_dtype(torch.bfloat16)
  print(jax.local_devices())
  fsdp_size = len(jax.devices()) // tp
  
  mesh = jax.make_mesh((fsdp_size, tp), ('fsdp', 'tp'))

  args = model.ModelArgs(
    **model.transformer_configs[model_type]
  )
  #args.n_layers = 2

  with torch.device('meta'):
    llama = model.Transformer(args)
  sharded_weights = create_sharded_weights(llama, mesh)
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


  train.train_loop(mesh, llama, sharded_weights, None, freqs_cis, lr, seqlen)


if __name__ == '__main__':
  import fire
  fire.Fire(main)