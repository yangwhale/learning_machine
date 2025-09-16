import jax
import torch
import torchax
from torchax.interop import jax_view

torchax.enable_globally()


class M(torch.nn.Module):

  def __init__(self, a: int):
    super().__init__()
    self.a = a  # 
    self.param = torch.nn.Parameter(torch.randn([2,2]))

  def forward(self, x):
    return x @ self.param * self.a

    

m = M(2)

def call_m(replace_a, weights, x):
  old = m.a
  m.a = replace_a
  res = torch.func.functional_call(m, weights, x)
  m.a = old
  return res

m.to('jax')
x = torch.randn((2,2), device='jax')

print(jax.jit(jax_view(call_m)).lower(2, jax_view(m.state_dict()), jax_view(x)).as_text())

"""
(py13) hanq_google_com@t1v-n-ffd511c3-w-0:~/learning_machine$ python module_with_scalar_state.py 
/home/hanq_google_com/miniconda3/envs/py13/lib/python3.13/site-packages/jax/_src/cloud_tpu_init.py:86: UserWarning: Transparent hugepages are not enabled. TPU runtime startup and shutdown time should be significantly improved on TPU v5e and newer. If not already set, you may need to enable transparent hugepages in your VM image (sudo sh -c "echo always > /sys/kernel/mm/transparent_hugepage/enabled")
  warnings.warn(
WARNING:root:Duplicate op registration for aten.__and__
module @jit_call_torch attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<i32>, %arg1: tensor<2x2xf32>, %arg2: tensor<2x2xf32>) -> (tensor<2x2xf32> {jax.result_info = "result"}) {
    %0 = stablehlo.dot_general %arg2, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %1 = stablehlo.convert %arg0 : (tensor<i32>) -> tensor<f32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<f32>) -> tensor<2x2xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<2x2xf32>
    return %3 : tensor<2x2xf32>
  }
}
"""