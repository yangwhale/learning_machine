import torch
from functorch.compile import aot_function



class M(torch.nn.Module):

  def forward(self, a, b):
    a[1:2].add_(1)
    a.add_(b)
    return a, b 



def backend_after_functionalization(fx, sample_inputs):
  print('---- after functionalization ----')
  print(fx.code)
  return fx


def backend(fx, sample_inputs):
  print('---- before functionalization ----')
  print(fx.code)
  return aot_function(fx, fw_compiler=backend_after_functionalization)


a = torch.randn((2,2))
b = torch.randn((2,2))

m = M()
m_compiled = torch.compile(m, backend=backend)
print(m_compiled(a, b))


"""
Output is 
---- before functionalization ----



def forward(self, L_a_ : torch.Tensor, L_b_ : torch.Tensor):
    l_a_ = L_a_
    l_b_ = L_b_
    getitem = l_a_[slice(1, 2, None)]
    add_ = getitem.add_(1);  getitem = add_ = None
    add__1 = l_a_.add_(l_b_);  l_a_ = l_b_ = add__1 = None
    return ()

---- after functionalization ----



def forward(self, arg0_1, arg1_1):
    slice_1 = torch.ops.aten.slice.Tensor(arg0_1, 0, 1, 2)
    add = torch.ops.aten.add.Tensor(slice_1, 1);  slice_1 = None
    slice_scatter = torch.ops.aten.slice_scatter.default(arg0_1, add, 0, 1, 2);  arg0_1 = add = None
    add_1 = torch.ops.aten.add.Tensor(slice_scatter, arg1_1);  slice_scatter = arg1_1 = None
    return (add_1,)

(tensor([[-1.1101, -1.8140],
        [ 1.2518,  2.0626]]), tensor([[-1.2198,  0.1516],
        [ 0.7614,  1.3113]]))
"""


