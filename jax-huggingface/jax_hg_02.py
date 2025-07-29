import time
import jax
from transformers import AutoModelForCausalLM, AutoTokenizer
import torchax 
from torchax.interop import torch_view

from jax.sharding import PartitionSpec as P, NamedSharding

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
model_inputs = tokenizer(["The secret to baking a good cake is "], return_tensors="jax")
print(model_inputs)

mesh = jax.make_mesh((jax.device_count(), ), ('axis', ))


# added later
from jax.tree_util import register_pytree_node
from transformers import modeling_outputs

def output_flatten(v):
  return v.to_tuple(), None

def output_unflatten(aux, children):
  return modeling_outputs.CausalLMOutputWithPast(*children)

register_pytree_node(
  modeling_outputs.CausalLMOutputWithPast,
  output_flatten,
  output_unflatten,
)

from transformers import cache_utils

def _flatten_dynamic_cache(dynamic_cache):
  return (
      dynamic_cache.key_cache,
      dynamic_cache.value_cache,
  ), None

def _unflatten_dynamic_cache(aux, children):
  cache = cache_utils.DynamicCache(),
  cache.key_cache, cache.value_cache = children
  return cache

register_pytree_node(
  cache_utils.DynamicCache,
  _flatten_dynamic_cache,
  _unflatten_dynamic_cache,
)

model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf", 
        torch_dtype="bfloat16", device_map="cpu")
        
weights, func = torchax.extract_jax(model)


# for name, w in weights.items():
#   print(name, w.shape)

def shard_weights_llama(mesh, weights):
  result = {}
  for k, v in weights.items():
    if (('q_proj' in k) or 
       ('k_proj' in k) or 
       ('v_proj' in k) or 
       ('gate_proj' in k) or 
       ('up_proj' in k)):
      sharding = P('axis', None)
    elif(('o_proj' in k) or 
       ('down_proj' in k) or 
       ('lm_head.weight' in k) or 
       ('embed_tokens' in k)):
      sharding = P(None, 'axis')
    else:
      sharding = P() # replicated
    result[k] = jax.device_put(v, NamedSharding(mesh, sharding))
  return result 


def func_with_constant(weights, input_ids):
  res = func(weights, (input_ids, ), {'use_cache': False})
  return res

res = func_with_constant(weights, model_inputs.input_ids)
print(res)

jitted_func = jax.jit(func_with_constant)

weights = shard_weights_llama(mesh, weights)
model_inputs.input_ids = jax.device_put(
  model_inputs.input_ids, NamedSharding(mesh, P())) # replicate

jax.block_until_ready(weights)

with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=False):
  for i in range(3):
    start = time.time()
    res = jitted_func(weights, model_inputs.input_ids)
    jax.block_until_ready(res)
    end = time.time()
    print(i, end - start, 'seconds')
 
# env = torchax.default_env()

# with env:
#   model.to('jax')
#   # ??
#   model.model.rotary_emb.original_inv_freq = model.model.rotary_emb.original_inv_freq.to('jax')
#   jmodel = torchax.interop.JittableModule(model)

#   model_inputs = dict(model_inputs)
#   generated_ids = jmodel.generate(**torch_view(model_inputs))
                                          

# print(generated_ids)
# print(tokenizer.batch_decode(generated_ids.cpu())[0])

