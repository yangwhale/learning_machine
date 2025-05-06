import torch_xla.runtime as xr
import numpy as np
from torch import nn
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.core.xla_model as xm
import torch
import time
from torch.optim import AdamW
import torch_xla.distributed.parallel_loader as pl
from torch_xla.experimental.spmd_fully_sharded_data_parallel import SpmdFullyShardedDataParallel


def log_tensor_sharding(t: torch.Tensor, log_prefix: str):
    xm.mark_step()
    sharding = torch_xla._XLAC._get_xla_sharding_spec(t)
    shape = t.shape
    device = t.device.type
    print(f"{log_prefix} sharding: {sharding} shape: {shape} device: {device}", flush=True)


class RandomTensorDataset:
    def __init__(self, tensor_shape, element_count):
        self.tensor_shape = tensor_shape
        self.element_count = element_count

    def __iter__(self):
        for _ in range(self.element_count):
            yield torch.randn(self.tensor_shape)


def main(
  model_axis=2,
  fsdp_axis=32,
  num_layers=48,
  profile_dir='/tmp/profile_dir',
  use_fsdp_wrapper=False,
  per_device_batch_size=None
):

  if use_fsdp_wrapper:
    model_axis = 1
  xr.use_spmd()
  num_devices = xr.global_runtime_device_count()
  ddp_axis = num_devices // model_axis // fsdp_axis

  mesh_shape = (ddp_axis, fsdp_axis, model_axis)
  device_ids = np.array(range(num_devices))
  print(f"running SPMD with num_devices: {num_devices} mesh: {mesh_shape}", flush=True)

  if use_fsdp_wrapper:
    mesh = xs.Mesh(device_ids, (num_devices, ), ("fsdp", ))
  else:
    mesh = xs.Mesh(device_ids, mesh_shape, ("ddp", "fsdp", "model"))

  dim_out = dim = 4096
  inner_dim = dim * 4
  out_channels = 128
  if per_device_batch_size is None:
    batch_size = (ddp_axis * fsdp_axis)
  else:
    batch_size = int(num_devices * per_device_batch_size)
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

  model = Model().to(xm.xla_device())

  if use_fsdp_wrapper:
    model = SpmdFullyShardedDataParallel(model, mesh=mesh)
  else:
    for name, weights in model.state_dict().items():
      print(name, weights.shape)
      if 'layer1' in name:
        xs.mark_sharding(weights, mesh, ('model', 'fsdp'))
      if 'layer2' in name:
        xs.mark_sharding(weights, mesh, ('fsdp', 'model'))
      if 'output' in name:
        xs.mark_sharding(weights, mesh, ('model', 'fsdp'))

  mse_loss = nn.MSELoss(reduction='sum')
  optimizer = AdamW(params=model.parameters())
  target = torch.zeros(batch_size, tokens_count, out_channels).to(device=xm.xla_device())
  target = target.reshape(-1, fsdp_axis, tokens_count, out_channels)
  xs.mark_sharding(target, mesh, ("ddp", "fsdp", None, None))

  dataloader = RandomTensorDataset(tensor_shape=(batch_size, tokens_count, dim), element_count=steps_count)
  dataloader_wrapper = pl.MpDeviceLoader(
      dataloader,
      device=xm.xla_device(),
      input_sharding=xs.ShardingSpec(mesh, partition_spec=("fsdp", None, None), minibatch=False)
  )

  xm.mark_step()

  import torch_xla.debug.profiler as xp

  server = xp.start_server(9012)

  xp.trace_detached(
      'localhost:9012',
      logdir=profile_dir,
      duration_ms=300000)


  for sample_index, sample in enumerate(dataloader_wrapper):
      print("step {}/{}".format(sample_index, steps_count), flush=True)
      
      sample = sample.reshape(-1, fsdp_axis, tokens_count, dim)
      print('shape', sample.shape)
      xs.mark_sharding(sample, mesh, ("ddp", "fsdp", None, None))
      xm.wait_device_ops()
      torch_xla.sync(wait=True)
      start = time.perf_counter()
      with xp.Trace('forward'):
        output = model(sample)
        xs.mark_sharding(output, mesh, ("ddp", "fsdp", None, None))
        loss = mse_loss(output, target)

      with xp.Trace('backward'):
        loss.backward()

      with xp.Trace('optimizer'):
        optimizer.step()

      torch_xla.sync(wait=True)
      xm.wait_device_ops()
      end = time.perf_counter()
      print('iteration {} and time {}'.format(sample_index, end - start))

      # log_tensor_sharding(model.m[0].layer1.weight, 'layer1')
      # log_tensor_sharding(model.m[0].layer1.weight.grad, 'layer1 grad')

      # for name, x in optimizer.state_dict()['state'].items():
      #   for name2, y in x.items():
      #     if isinstance(y, torch.Tensor) and 'xla' in str(y.device):
      #       log_tensor_sharding(y, 'optimizer: {}'.format(name) + name2)
      #     else:
      #       print(name2, type(y))

      optimizer.zero_grad()

if __name__ == '__main__':
    import fire
    fire.Fire(main)