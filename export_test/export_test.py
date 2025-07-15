import copy
import time
from functools import wraps

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla.experimental.xla_mlir_debuginfo  # noqa: F401
from torch.ao.quantization.fake_quantize import FakeQuantize
from torch.ao.quantization.observer import MinMaxObserver
from torch.export import export
from torch.library import Library, impl, register_fake

import torchax


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"Function {func.__name__} took {total_time:.4f} seconds to execute.")
        return result

    return wrapper


NNMLIR_LIB = Library("nnmlir", "DEF")
NNMLIR_LIB.define("quant_spec(Tensor t, Tensor min, Tensor max) -> Tensor")
impl(f"{NNMLIR_LIB.ns}::quant_spec", "default", func=lambda x, _, __: x)
register_fake(f"{NNMLIR_LIB.ns}::quant_spec", lambda x, _, __: torch.empty_like(x))


def quant_spec_dummy(t, min, max):
  # Call with keyword arg to set composite attributes
  # https://docs.jax.dev/en/latest/_autosummary/jax.lax.composite.html
  return jax.lax.composite(
    lambda x, min, max: x, # no op
    name="nnmlir.quant_spec",
  )(t, min=min, max=max) 
  return t

from torchax.ops import jlibrary

jlibrary.register_jax_composite(
  "quant_spec",
  quant_spec_dummy,
  torch.ops.nnmlir.quant_spec.default,
  static_argnums=(1,2 ), # mark min and max as constant for composite
)


def trip_softmax(x, dim):
  res = F.softmax(x, dim)
  return res

jlibrary.register_torch_composite(
  "trip_softmax",
  trip_softmax
)


class MyModel(nn.Module):
    def __init__(self, loops=1000):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.loops = loops

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        for _ in range(self.loops):
            x = x + 1.0
        x = trip_softmax(x, -1)
        return x


m = MyModel(12000)
old_params = copy.deepcopy(list(m.parameters()))

# Execute this model using torch.
inputs = torch.randn(3, 3, 28, 28)
# Sample input is a tuple
sample_input = (inputs,)
output = m(*sample_input)

print("Benchmarking model for calibration")
exported = timeit(export)(m, sample_input)

print(f"Exported graph has {len(exported.graph.nodes)} nodes.")

@timeit
def quantization_saturation_pass(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    for node in reversed(gm.graph.nodes):
        if node.op == "output":
            continue

        with gm.graph.inserting_after(node):
            gm.add_submodule(f"{node}_obs", FakeQuantize(MinMaxObserver))
            new_observer = gm.graph.call_module(f"{node}_obs")
            node.replace_all_uses_with(new_observer)
            new_observer.args = (node,)


quantization_saturation_pass(exported.graph_module)


forward = exported.module()
output = forward(inputs)

# Let's train the dummy model
for a, b in zip(old_params, m.parameters()):
    assert torch.allclose(a, b)

optim = torch.optim.Adam(m.parameters(), lr=0.1)
for _ in range(10):
    t = torch.randn(3, 3, 28, 28)
    out = m(t)
    out.sum().backward()
    optim.step()
    optim.zero_grad()

for a, b in zip(old_params, m.parameters()):
    assert not torch.allclose(a, b)


# Either remove and store scale in metadata/"layerspec" or materialize the QDQs right
# here. Probably prefer materializing some sort of QDQ right here instead of maintaining
# a spec table. Keep the graphas the single source of truth instead of having to wrangle
# and pass metadata around.
@timeit
def quantization_module_to_quantspec_pass(
    gm: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    for node in gm.graph.find_nodes(op="call_module"):
        sm = gm.get_submodule(node.name)
        if not isinstance(sm, FakeQuantize):
            continue

        obs = sm.activation_post_process

        def quantstub(x, tmin, tmax):
          if isinstance(tmin, torch.Tensor):
            tmin = tmin.item()
          if isinstance(tmax, torch.Tensor):
            tmax = tmax.item()
          return torch.ops.nnmlir.quant_spec(x, tmin, tmax)
        
        #quantstub = torch.ops.nnmlir.quant_spec

        gm.register_buffer(f"{node}_min", obs.min_val)
        gm.register_buffer(f"{node}_max", obs.max_val)

        with gm.graph.inserting_after(node):
            quantspec = gm.graph.call_function(quantstub)
            min_node = gm.graph.get_attr(f"{node}_min")
            max_node = gm.graph.get_attr(f"{node}_max")

        quantspec.args = (node.args[0], min_node, max_node)

        node.replace_all_uses_with(quantspec)

    gm.graph.eliminate_dead_code()


quantization_module_to_quantspec_pass(exported.graph_module)

use_jax_for_the_last_step = True
if False:
  exported = timeit(export)(exported.module(), *exported.example_inputs)
  # exported = timeit(exported.run_decompositions)()
  stablehlo_graph_module = timeit(exported_program_to_stablehlo)(
      exported, StableHLOExportOptions(custom_ops_allowed_in_graph={"nnmlir"})
  )
else:
  import torchax
  import jax
  from torchax.ops import mappings
  import jax.export
  from torch.utils import _pytree as pytree
  from torchax import interop
  start = time.time()

  def make_shape_struct(x):
    return jax.ShapeDtypeStruct(x.shape, mappings.t2j_dtype(x.dtype))



  env = torchax.default_env()
  with env:
    model = exported.module().to('jax')
    def func_to_export(x):
      # hard code weights in model
      return model(x)
    example_inputs_jax = pytree.tree_map_only(torch.Tensor, lambda x: x.to('jax'), exported.example_inputs)
    res = jax.jit(interop.jax_view(func_to_export)).lower(*example_inputs_jax[0])
  end = time.time()
  print(res.as_text())
  print(f'Second export time = {end - start:.4f}')
  # print(stablehlo_graph_module.get_stablehlo_text())
