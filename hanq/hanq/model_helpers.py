import math

def which_are_modules(pipe):
  for m in dir(pipe):
      module = getattr(pipe, m, None)
      if isinstance(module, torch.nn.Module):
          print(m)

          
def module_size(module):
  size = 0
  for k, v in module.state_dict():
    size += math.prod(v.shape) * v.dtype.itemsize
  return size
    