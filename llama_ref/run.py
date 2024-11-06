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
      weight = jnp.ones(
        weight_meta.shape, 
        dtype=mappings.t2j_dtype(weight_meta.dtype))
    print(name, weight.shape, weight.dtype)
    res[name] = jax.device_put(weight, sharding)
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
  


def main():
  torch.manual_seed(0)
  print(jax.local_devices())
  TP = 4
  fsdp_size = len(jax.devices()) // TP
  
  mesh = jax.make_mesh((fsdp_size, TP), ('fsdp', 'tp'))

  args = model.ModelArgs()
  #args.n_layers = 2
  args.vocab_size = 32000

  torch.set_default_device('meta')
  llama = model.Transformer(args)
  sharded_weights = create_sharded_weights(llama, mesh)

  freqs_cis = torch_xla2.default_env().j2t_iso(sharded_weights['freqs_cis'])
  del sharded_weights['freqs_cis']
  train.train_loop(mesh, llama, sharded_weights, None, freqs_cis)


if __name__ == '__main__':
  main()