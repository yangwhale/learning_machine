import jax
from jax.sharding import NamedSharding, PartitionSpec as P, Mesh
from jax.experimental.mesh_utils import create_device_mesh, create_hybrid_device_mesh
import numpy as np
from torch import nn
import torch
import time

import torchax as tx
import torchax.interop
import torchax.train
import optax


class RandomTensorDataset:
    def __init__(self, tensor_shape, element_count):
        self.tensor_shape = tensor_shape
        self.element_count = element_count

    def __iter__(self):
        for _ in range(self.element_count):
            yield torch.randn(self.tensor_shape)

def make_train_step(model_fn, 
                    loss_fn, optax_optimizer, 
                    remat_policy=None):
  """Make a function that do one train step given model and loss.

  model_fn: a function representing the model's forward:
      i.e. has signature Callable[weights, buffers, args] -> result. Where,
      weights is a pytree of trainable parameters
      buffers is a pytree of non-trainable parameters / constants
      args is the input data loaded from the data set
      result is the return value of the model
  loss_fn: a function to compute loss.
      i.e. it has signature of Callable[result, label] -> loss
      where, result is what model_fn returned
        loss is loaded from the dataloader.
  optax_optimizer: the optimizer from optax library. for example, optax.adam
  remat_policy: One of jax.ad_checkpoint.checkpoint_policies, specifies how
      to do gradient checkpointing. If None, then it means checkpoint everything.
  """
  env = torchax.default_env()
  def loss(weights, buffers, args, label): # inputs are XLATensor
    with env, jax.named_scope('compute_loss'):
      res = model_fn(weights, buffers, args)
      l = loss_fn(res, label)
      return l

  #loss = interop.gradient_checkpoint(loss, kwargs={'policy': remat_policy})
  grad_fn = tx.interop.jax_value_and_grad(loss)

  def step(weights, buffers, opt_state, args, label): #inputs are array
    with jax.named_scope('compute_gradient'):
        loss, gradient = grad_fn(weights, buffers, args, label)

    with jax.named_scope("optimizer_updates"):
        updates, opt_state = tx.interop.call_jax(
            optax_optimizer.update,
            gradient, opt_state, weights)
        weights = tx.interop.call_jax(optax.apply_updates, weights, updates)
    return loss, weights, opt_state

  # TODO: apply jax.jit so the user don't have to.
  return tx.interop.jax_jit(step, {'donate_argnums': (0, 2)})

def sharded_device_put(tensor, sharding):
    num_global_devices = jax.device_count()
    num_local_devices = jax.local_device_count()
    if isinstance(tensor, tuple):
        return tuple(sharded_device_put(t, sharding) for t in tensor)

    if num_global_devices == num_local_devices:
        return jax.device_put(tensor, sharding)

    shape = tensor.shape
    x_split = [jax.device_put(tensor[i], device) 
               for device, i in sharding.addressable_devices_indices_map(shape).items()]
    return jax.make_array_from_single_device_arrays(shape, sharding, x_split)

def main(
  model_axis=4,
  num_layers=48,
  profile_dir='/tmp/profile_dir'
):
  env = tx.enable_globally()
  env.default_device_or_sharding = jax.devices('cpu')[0]

  num_devices = len(jax.devices())
  mesh_shape = (num_devices // model_axis, model_axis)
  device_ids = np.array(range(num_devices))
  print(f"running SPMD with num_devices: {num_devices} mesh: {mesh_shape}", flush=True)

  batch_size = num_devices // model_axis

  #mesh = jax.make_mesh((batch_size, model_axis), ('data', 'model'))
  dev_array = create_device_mesh((batch_size, model_axis), allow_split_physical_axes=True)
  mesh = Mesh(dev_array, ('data', 'model'))

  dim_out = dim = 4096
  inner_dim = dim * 4
  out_channels = 128

  tokens_count = 2048
  steps_count = 10


  class FFN(torch.nn.Module):
    def __init__(self):
      super().__init__()
      self.layer1 = nn.Linear(dim, inner_dim, bias=False)
      self.layer2 = nn.Linear(inner_dim, dim_out, bias=False)
      self.dropout = nn.Dropout(0.0)

    def forward(self, x):
      x = self.layer1(x)
      x = self.dropout(x)
      x = self.layer2(x)
      return x



  class Model(torch.nn.Module):

    def __init__(self):
      super().__init__()
      self.m= torch.nn.ModuleList(
        [FFN() for _ in range(num_layers)]
      )
      self.output = nn.Linear(dim_out, out_channels, bias=False)

    def forward(self, x):
      for layer in self.m:
        x = layer(x)
      x = self.output(x)
      return x
    
  
  model = Model().to('jax')
  print('device is', model.m[0].layer1.weight.jax().device)

  data_sharding = NamedSharding(mesh, P('data'))

  sharded_weights = {}
  for name, weights in model.state_dict().items():
    print(name, weights.shape)
    if 'layer1' in name:
      sharding_spec = P('model', 'data')
    if 'layer2' in name:
      sharding_spec = P('data', 'model')
    if 'output' in name:
      sharding_spec = P('model', 'data')
    with jax.default_device('cpu'):
      w = torch.randn(weights.shape, dtype=weights.dtype, device='jax')
    sharded_weights[name] = w.apply_jax(sharded_device_put, NamedSharding(mesh, sharding_spec))

  mse_loss = nn.MSELoss(reduction='sum')


  def call_model(weights, buffer, args):
    args[0].shard_(data_sharding)
    res = torch.func.functional_call(model, weights, args)
    res.shard_(data_sharding)
    return res

  optimizer = optax.adamw(0.03)

  opt_state = tx.interop.call_jax(optimizer.init, sharded_weights)

  train_step = make_train_step(
      call_model, 
      mse_loss,
      optimizer,
  )

  #train_step = tx.interop.jax_jit(train_step, kwargs_for_jax_jit={'donate_argnums': (0, 2)})


  target = torch.zeros(batch_size, tokens_count, out_channels, device='jax')
  target.apply_jax_(sharded_device_put, data_sharding)
  dataloader = RandomTensorDataset(tensor_shape=(batch_size, tokens_count, dim), element_count=steps_count)

  start = time.time()

  jax.profiler.start_trace(profile_dir)
  for sample_index, sample in enumerate(dataloader):
      print("step {}/{}".format(sample_index, steps_count), flush=True)
      sample = sample.to('jax').apply_jax(sharded_device_put, data_sharding)
      sample.apply_jax(jax.block_until_ready) # wait data sharding to complete

      start = time.perf_counter()
      loss, sharded_weights, opt_state = train_step(sharded_weights, {}, opt_state, (sample, ), target)

      # wait until ready to take accurate time
      jax.block_until_ready(loss.jax())
      end = time.perf_counter()
      print('step {} used {}s'.format(sample_index, end - start))
  jax.profiler.stop_trace()

  print(f"sec / step is {(time.time() - start) / steps_count}")


if __name__ == '__main__':
  import fire
  fire.Fire(main)