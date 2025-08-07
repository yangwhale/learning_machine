import sys
import time
import jax
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
import torchax as tx
from torchax.interop import torch_view

from jax.sharding import PartitionSpec as P, NamedSharding
import torch

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
model_inputs = tokenizer(["The secret to baking a good cake is "], return_tensors="pt")
print(model_inputs)

mesh = jax.make_mesh((jax.device_count(), ), ('axis', ))
env = tx.default_env()


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

    # apply_jax calls a jax function with the jax.Array inside of the tensor
    # as input.
    result[k] = v.apply_jax(jax.device_put, NamedSharding(mesh, sharding))
  return result 

def run_twice_and_print_cache(model, input_ids):
  res = model(input_ids)
  print('number of layers', len(res[1]))
  for k, v in res[1]:
    print('first kv cache')
    print(k.shape, v.shape)
    break

  next_token = torch.argmax(res[0][:, -1], dim=-1)

  res = model(next_token.unsqueeze(0), past_key_values=res[1])
  print('number of layers', len(res[1]))
  for k, v in res[1]:
    print('second kv cache')
    print(k.shape, v.shape)
    break


def run_twice_and_print_cache_static(model, input_ids):
  past_key_values = StaticCache(config=model.config, max_batch_size=1, max_cache_len=50, 
                                device='jax', dtype=model.dtype)
  res = model(input_ids, past_key_values=past_key_values)
  print('number of layers', len(res[1]))
  for k, v in res[1]:
    print('first kv cache')
    print(k.shape, v.shape)
    break

  next_token = torch.argmax(res[0][:, -1], dim=-1)

  res = model(next_token.unsqueeze(0), past_key_values=res[1])
  print('number of layers', len(res[1]))
  for k, v in res[1]:
    print('second kv cache')
    print(k.shape, v.shape)
    break


def autoregressive_decode(model, input_ids, tokenizer, max_tokens=50):
  start = time.perf_counter()
  res = model(input_ids)

  next_token = torch.argmax(res[0][:, -1], dim=-1)
  result_tokens = [int(next_token.item())]

  for _ in range(max_tokens):
    res = model(next_token.unsqueeze(0), past_key_values=res[1])
    next_token = torch.argmax(res[0][:, -1], dim=-1)
    if next_token.item() == tokenizer.eos_token:
      break
    result_tokens.append(next_token.item())
  end = time.perf_counter()
  
  print('decoded', tokenizer.batch_decode([result_tokens]))
  print(f'took {end - start} seconds')
  return result_tokens

# move the model's weight to 'jax' device. i.e. a tensor with a 
# jax array inside
with env:
  model.to('jax')
  weights = shard_weights_llama(mesh, model.state_dict())
  input_ids = model_inputs.input_ids.to('jax').apply_jax_(
    jax.device_put,
    NamedSharding(mesh, P()))
  tx.interop.call_jax(jax.block_until_ready, weights)

  run_twice_and_print_cache(model, input_ids)
  #run_twice_and_print_cache_static(model, input_ids)

  autoregressive_decode(model, input_ids, tokenizer)


sys.exit(0)
with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=False):
  for i in range(3):
    start = time.time()
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

