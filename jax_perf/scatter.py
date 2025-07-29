import time
import torchax
import numpy as np
import jax.numpy as jnp
import jax
import time, os, sys, json, functools
from datetime import datetime
from inspect import currentframe, getframeinfo

from jax.sharding import NamedSharding, PartitionSpec as P, SingleDeviceSharding

mesh = jax.make_mesh((8, ), ('dp', ))
sharding = NamedSharding(mesh, P('dp'))
local_devices = jax.local_devices()
single_dev_sharding = SingleDeviceSharding(local_devices[0])


def tensor_info(tensor, msg=""):
    frameinfo = getframeinfo(currentframe().f_back)
    filename = frameinfo.filename.rsplit("/", 1)[-1]
    log_time = datetime.today().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    if tensor is None:
        print(f"{log_time} [INFO] {filename}:{frameinfo.lineno} [{msg}] None", flush=True)
    elif isinstance(tensor, jax.Array) or isinstance(tensor, torchax.tensor.Tensor):
        if isinstance(tensor, torchax.tensor.Tensor):
            env = torchax.default_env()
            array = env.t2j_iso(tensor)
        else:
            array = tensor
        array = jnp.abs(array)
        print(f"{log_time} [INFO] {filename}:{frameinfo.lineno} [{msg}] {array.shape} {array.dtype} {tensor.device} {array.sharding} mean={jnp.mean(array).item():.6f} max={jnp.max(array).item():.6f}", flush=True)
    else:
        print(f"{log_time} [INFO] {filename}:{frameinfo.lineno} [{msg}] {tensor.shape=} {tensor.dtype=} {tensor.device=} mean={tensor.abs().mean().item():.6f} max={tensor.abs().max().item():.6f}", flush=True)

def scatter0(array):
  return jax.device_put(array,sharding) 

def scatter1(array):
  length = array.shape[0] // jax.device_count()
  offset = jax.process_index() * len(jax.devices()) // len(jax.local_devices())
  arrays = [jax.device_put(array[(i+offset)*length:(i+offset+1)*length], d) 
            for i, d in enumerate(local_devices)]
  return jax.make_array_from_single_device_arrays(array.shape, sharding, arrays)

def scatter2(array):
  return jax.make_array_from_callback(array.shape, sharding, lambda i: array[i])


def scatter3(array):
  # https://docs.jax.dev/en/latest/_autosummary/jax.make_array_from_process_local_data.html
  return jax.make_array_from_process_local_data(sharding, array)

def scatter4(array):

  @functools.partial(
    jax.jit,
    out_shardings=sharding
  )
  def f(x):
    return x

  return f(array)

  
def first_to_one_dev_then(func):
  # first device put to one device then do the rest
  def newfunc(arr):
    arr = jax.device_put(arr, single_dev_sharding)
    return func(arr)

  newfunc.__name__ = 'single_then_' + func.__name__
  return newfunc



def run_and_measure(f, arr):
  # burn once?
  arr = f(arr)
  jax.block_until_ready(arr)

  start = time.perf_counter()
  for i in range(3):
    arr = f(arr)
    jax.block_until_ready(arr)
  end = time.perf_counter()
  # tensor_info(arr)
  print(f'{f.__name__} run: {(end - start) / 3}s')

    
# Goal:
# 4096 x 4096 on CPU -> to sharded 512 x 4096 on 8 chips

SIZE = 4096 * 4

source = np.ones((SIZE, SIZE))

run_and_measure(scatter0, source)
run_and_measure(scatter1, source)
run_and_measure(scatter2, source)
run_and_measure(scatter3, source)
run_and_measure(scatter4, source)

run_and_measure(first_to_one_dev_then(scatter0), source)
run_and_measure(first_to_one_dev_then(scatter1), source)
run_and_measure(first_to_one_dev_then(scatter2), source)
run_and_measure(first_to_one_dev_then(scatter3), source)
run_and_measure(first_to_one_dev_then(scatter4), source)
# measure device put

'''
Measurement on a single v6e-8:
scatter0 run: 9.928998770192266e-05s
scatter1 run: 0.12208282633218914s
scatter2 run: 0.008794211336256316s
scatter3 run: 7.440667832270265e-05s
scatter4 run: 0.08270870166597888s
single_then_scatter0 run: 0.017481056333053857s
single_then_scatter1 run: 0.01385946333175525s
single_then_scatter2 run: 0.014783153329820683s
single_then_scatter3 run: 0.014735486649442464s
jax.errors.SimplifiedTraceback: For simplicity, JAX has removed its internal frames from the traceback of the following exception. Set JAX_TRACEBACK_FILTERING=off to include these.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/hanq_google_com/learning_machine/jax_perf/scatter.py", line 107, in <module>
    run_and_measure(first_to_one_dev_then(scatter4), source)
  File "/home/hanq_google_com/learning_machine/jax_perf/scatter.py", line 78, in run_and_measure
    arr = f(arr)
  File "/home/hanq_google_com/learning_machine/jax_perf/scatter.py", line 69, in newfunc
    return func(arr)
  File "/home/hanq_google_com/learning_machine/jax_perf/scatter.py", line 62, in scatter4
    return f(array)
ValueError: Received incompatible devices for jitted computation. Got argument x of scatter4.<locals>.f with shape float32[4096,4096] and device ids [0] on platform TPU and explicit output sharding with device ids [0, 1, 2, 3, 7, 6, 5, 4] on platform TPU
'''


    

