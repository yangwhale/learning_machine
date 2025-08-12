# How to Run a Hugging Face Model in JAX (Part 3)

In the previous posts, [part 1](01-run-huggingface-model-in-jax.md) and
[part 2](02-run-huggingface-model-in-jax.md), we explored how to 
call the `forward` function of a HuggingFace model. Now let's see
how we can run it's autoregressive decoding function. But before that, let's first
dive into how `torchax` works.

Before you begin, if we installed `torchax` by following previous examples, 
please reinstall from github via

```bash
pip install git+https://github.com/pytorch/xla.git#subdirectory=torchax
```

because there are some recent bugfixed (found while writing this post).

-----


## How torchax works

torchax seems to be converting PyTorch models into JAX functions, but actually
it is doing something different. Namely, it's dressing JAX Arrays with custome
to make it look like a `torch.Tensor`.

Then, when the `torch.nn.Module` is invoked, it thinks that it is accepting
`torch.Tensor` as input, but we actually sneaked in `jax.Array`'s!

![alt text](image-trojan.png)

Now instead of using `extract_jax` API, let's see what is really going on
by using `torchax`'s `Environment` and `Tensor` directly.

```python
import torch
import torchax as tx
import jax
import jax.numpy as jnp

# the torchax environment is what make pytorch operators work on torchax's Tensor
env = tx.default_env()

# Start with an jax array:

arr = jnp.ones((4, 4))

# call torch function on jax array is error
# torch.matmul(arr, arr)

# convert arr to a Tensor
tensor = tx.interop.torch_view(arr)

print(isinstance(tensor, torch.Tensor)) # prints True
print(tensor.__dict__)
```

And we get:

```
Is torch Tensor: True
inner data of my tensor {'_elem': Array([[1., 1., 1., 1.],
       [1., 1., 1., 1.],
       [1., 1., 1., 1.],
       [1., 1., 1., 1.]], dtype=float32), '_env': <torchax.tensor.Environment object at 0x772f8cd67fd0>}
```

Here we see that, the tensor converted from JAX Array is a `torch.Tensor`, and 
inside it is holding our original JAX Array. It also remembers the environment
object that it belongs, by default it's the same one as `tx.default_env()`.

Now we can try running some pytorch operators on this tensor, for that, first
we need to activate the environment, then run operations inside of it.

```python
with env:
  print(torch.matmul(tensor, tensor))
  print(torch.sin(tensor))
  print(torch.exp(tensor))
```

we get our result. Note that it's not regular `torch.Tensor`, but the one with 
a `jax.Array` inside of it.

```
Tensor(<class 'jaxlib._jax.ArrayImpl'> [[4. 4. 4. 4.]
 [4. 4. 4. 4.]
 [4. 4. 4. 4.]
 [4. 4. 4. 4.]])
Tensor(<class 'jaxlib._jax.ArrayImpl'> [[0.841471 0.841471 0.841471 0.841471]
 [0.841471 0.841471 0.841471 0.841471]
 [0.841471 0.841471 0.841471 0.841471]
 [0.841471 0.841471 0.841471 0.841471]])
Tensor(<class 'jaxlib._jax.ArrayImpl'> [[2.7182817 2.7182817 2.7182817 2.7182817]
 [2.7182817 2.7182817 2.7182817 2.7182817]
 [2.7182817 2.7182817 2.7182817 2.7182817]
 [2.7182817 2.7182817 2.7182817 2.7182817]])
```

Besides creatings a `torchax.Tensor` via wrapping a Jax Array, another way
is to call `.to('jax')` on a regular `torch.Tensor` (on CPU).

So another way to write the above example is:
```python
with env:
  tensor = torch.ones((4,4)).to('jax')
  print(torch.matmul(tensor, tensor))
  print(torch.sin(tensor))
  print(torch.exp(tensor))
```

You can repro the above example by running

```bash
python torchax-demo.py
```

**A ML model is a graph composed with torch operators.  Therefore,
if every torch operator runs on our variant of Tensor, then we achieved running
the torch model on top of Jax.**

Now, let's rewrite example we have seen in the previous posts by constructing
tensors and calling torch models.

```python
# move the model's weight to 'jax' device. i.e. a tensor with a 
# jax array inside
with env:
  model.to('jax')
  weights = shard_weights_llama(mesh, model.state_dict())
  input_ids = model_inputs.input_ids.to('jax').apply_jax_(
    jax.device_put,
    NamedSharding(mesh, P()))
  tx.interop.call_jax(jax.block_until_ready, weights)
  print(model(input_ids))
```

Let's disect what is going on above:
1. `model.to('jax')` moves the torch model's weight to a especial 'jax' device. This is akin to `model.to('cuda')` using the CUDA backend. Once this happens, the tensor type will be torchax.Tensor. This tensor class has an extra method: `apply_jax` which will apply any jax function to the inner jax array.
2. The weights in the model are still unsharded, so we shard them using the same
   sharding method we did last time.
3. We call the model as any Pytorch model, and we get the expected result.

```
CausalLMOutputWithPast(loss=None, logits=Tensor(<class 'jaxlib._jax.ArrayImpl'> [[[-12.950611    -7.4854484   -0.42371067 ...  -6.819363    -8.073828
    -7.5583534 ]
  [-13.508438   -11.716616    -6.9578876  ...  -9.135823   -10.237023
    -8.56888   ]
  [-12.8517685  -11.180469    -4.0543456  ...  -7.9564795  -11.546011
   -10.686134  ]
  ...
  [ -2.983235    -5.621302    11.553352   ...  -2.6286669   -2.8319468
    -1.9902805 ]
  [ -8.674949   -10.042385     3.4400458  ...  -3.7776647   -8.616567
    -5.7228904 ]
  [ -4.0748825   -4.706395     5.117742   ...   6.7174563    0.5748794
     2.506649  ]]]), past_key_values=DynamicCache(), hidden_states=None, attentions=None)
```

## Autoregressive decoding explained in shapes

LLMs are trained to predict the next token of a given input sentence. 

![alt text](llm-predict.png)

Given an input sequence of length `n`; the LLM will need to take a 
tensor of shape `(1, n)` as input (here 1 is the batch size), and
return a tensor of also shape `(1, n)` of output, for the next token. 
From which, we only care the last token.

Next step is to append this token to the original input, forming
a input sequence of shape `(1, n + 1)`, and we repeat the process
for next `m` iterations, or until the model produces a stop signal, usually
the `end of sentence` (`eos`) token.

In other words, the input and output shape of the LLM are:

```
iter 1: (1, n) -> (1, n)
iter 2: (1, n + 1) -> (1, n + 1)
iter 3: (1, n + 2) -> (1, n + 2)
...
```
The shape changes each iteration.

## Decoding with KV Cache

People familiar with LLM inference will point out that the usual autoregressive
decoding setup will employ a `KVCache`. [This article](https://medium.com/@plienhar/llm-inference-series-4-kv-caching-a-deeper-look-4ba9a77746c8) explains it quite well.
The main idea is that because iter 1 only produced one new token, and that is 
the only token the model hasn't seen before, we can encode the previous seen token
inside of a cache and reuse some compute. 

In a inference setup with KVCache, the inputs and outputs of the model roughly looks
like this
```
iter 1: (1, n) -> (1, n), kvcache(n)
iter 2: (1, 1), kvcache(n) -> (1, 1), kvcache(n + 1)
iter 3: (1, 1), kvcache(n + 1) -> (1, 1), kvcache(n + 2)
...
```

Here I used the notation `kvcache(n)` to signify a kvcache of sequence length `n`.
The full shape of kvcache usually `(batch size,  number of heads, sequence length, head dim) x num_layers x 2`.
(times 2 because K and V, and there are 2 of those per layer).

Let's actually run the model and inspect the shapes of the kv cache:

```python
  print('number of layers', len(res[1]))
  for k, v in res[1]:
    print(k.shape, v.shape)
```

We can see:

```
number of layers 32
torch.Size([1, 32, 12, 128]) torch.Size([1, 32, 12, 128])
torch.Size([1, 32, 12, 128]) torch.Size([1, 32, 12, 128])
torch.Size([1, 32, 12, 128]) torch.Size([1, 32, 12, 128])
torch.Size([1, 32, 12, 128]) torch.Size([1, 32, 12, 128])
...
```
for llama2 model, there are 32 layers, and number of heads is also 32.

We can do autoregressive decoding by passing in next token and the cache back:

```python
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
```

Here we are using greedy sampling. and we get:

```
number of layers 32
first kv cache
torch.Size([1, 32, 12, 128]) torch.Size([1, 32, 12, 128])
number of layers 32
second kv cache
torch.Size([1, 32, 13, 128]) torch.Size([1, 32, 13, 128])
```
We see that the dynamic cache has grown in size 1.

As we know, `jax.jit` doesn't like changing shapes (it will recompile!). 
So if we keep using `DynamicCache` we can do inference using eager mode.

```python
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
```

It outputs:
```
decoded ['100% in the ingredients.\nI’ve been baking cakes for as long as I can remember. I’ve always loved the process of baking and the smell of freshly baked cakes.\nI']
took 130.90283443999942 seconds
```

We got the model to speak. Although the 130seconds for one request is very slow.

Now let's see how can we speed it up with `jax.jit`.

## Static Cache and jax.jit

The issue with `jax.jit` and the `DynamicCache` used above is that
the input and output shape changes in every iteration. Applying 
`jax.jit` blindly will make it even slower than eager mode: it will need
recompile a graph to run once, then discarded.

Luckily, HuggingFace has a setting to use `StaticCache`, a cache 
with fixed maximal length so we can avoid recompilation. According to the
[LLM inference optimization](https://huggingface.co/docs/transformers/v4.44.0/en/llm_optims?static-kv=advanced+usage%3A+control+Static+Cache#static-kv-cache-and-torchcompile) doc, StaticCache is
precisely introduced to support `torch.compile`; which also loves static shape.

We write the following function to test out:
Note that the python code seems more involved by it is copied from 
[LLM inference optimization](https://huggingface.co/docs/transformers/v4.44.0/en/llm_optims?static-kv=advanced+usage%3A+control+Static+Cache#static-kv-cache-and-torchcompile) doc by HuggingFace.

```python
def autoregressive_decode_static(model, input_ids, tokenizer, max_tokens=50):

  def decode_one_tokens(cur_token, input_pos, cache_position, past_key_values):
    logits, cache = model(
        cur_token,
        position_ids=input_pos,
        cache_position=cache_position,
        past_key_values=past_key_values,
        return_dict=False,
        use_cache=True
    )
    new_token = torch.argmax(logits[:, -1], dim=-1)[:, None]
    return new_token, cache

  batch_size, seq_length = input_ids.shape
  with torch.no_grad():
    start = time.perf_counter()
    past_key_values = StaticCache(
        config=model.config, 
        max_batch_size=1, max_cache_len=max_tokens, 
        device='jax', dtype=model.dtype
    )
    cache_position = torch.arange(seq_length, device='jax')
    generated_ids = []

    logits, past_key_values = model(
        input_ids, 
        cache_position=cache_position, 
        past_key_values=past_key_values, 
        return_dict=False, 
        use_cache=True
    )
    next_token = torch.argmax(logits[:, -1], dim=-1)[:, None]
    generated_ids.append(next_token[:, 0].item())

    cache_position = torch.tensor([seq_length + 1], device='jax')
    for _ in range(1, max_tokens):
        next_token, past_key_values = decode_one_tokens(
          next_token.clone(), None, cache_position, past_key_values)
        generated_ids.append(next_token.int().item())
        cache_position += 1
    end = time.perf_counter()

  text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
  print(text)
  print('Time: ', end - start)
```

output:

```
['1', '0', '0', '%', 'but', 'ter', '.', '\n', 'I', '’', 'm', 'not', 'sure', 'if', 'it', '’', 's', 'the', 'but', 'ter', 'or', 'the', 'eggs', ',', 'but', 'I', '’', 'm', 'pretty', 'sure', 'it', '’', 's', 'the', 'but', 'ter', '.', '\n', 'I', '’', 's', '\n', 'I', '’', 's', '\n', 'I', '’', '\n', 'I']
Time:  88.39702287199907
```

We got the same output and faster time because of the staticness. We haven't even tried to compile yet!

## Now let's jit it.

To compile the function using `jax.jit`, we can use the helper function at `torchax.interop.jax_jit`.

We make the following changes from the function above:

```python
  jitted = tx.interop.jax_jit(decode_one_tokens) # add this after defining decode_one_tokens
  # replace this line
-       next_token, past_key_values = decode_one_tokens(
  # with this:
+       next_token, past_key_values = jitted(
```
in other words, we are jitting `decode_one_tokens`, 
and replacing call to it with the jitted function.

Here we use `tx.interop.jax_jit` instead of `jax.jit` because `jax.jit` acts on JAX functions (functions that takes jax arrays as input and output), here, we 
are acting on torch functions (functions that takes and returns `torch.Tensor`).

Running it, we have found this error:
```
Traceback (most recent call last):
  File "/home/hanq_google_com/learning_machine/jax-huggingface/jax_hg_03.py", line 201, in <module>
    autoregressive_decode_static(model, input_ids, tokenizer)
  File "/home/hanq_google_com/learning_machine/jax-huggingface/jax_hg_03.py", line 177, in autoregressive_decode_static
    next_token, past_key_values = jitted(
  File "/home/hanq_google_com/pytorch/xla/torchax/torchax/interop.py", line 220, in call_jax
    res: JaxValue = jax_func(*args, **kwargs)
TypeError: Error interpreting argument to functools.partial(<function call_torch at 0x7d1cea648af0>, <function autoregressive_decode_static.<locals>.decode_one_tokens at 0x7d1c86d0e440>) as an abstract array. The problematic value is of type <class 'transformers.cache_utils.StaticCache'> and was passed to the function at path args[3].
This typically means that a jit-wrapped function was called with a non-array argument, and this argument was not marked as static using the static_argnums or static_argnames parameters of jax.jit.
```

Recall [episode 1](jax-huggingface/01-run-huggingface-model-in-jax.md) we encountered exactly the same issue, namely, StaticCache need to be registered in pytree.

To do that, we add the following:

```python
def _flatten_static_cache(cache):
  return (
      cache.key_cache,
      cache.value_cache,
  ), (cache._config, cache.max_batch_size, cache.max_cache_len)

def _unflatten_static_cache(aux, children):
  cache = cache_utils.StaticCache(*aux)
  cache._config = aux[0]
  cache.key_cache, cache.value_cache = children
  return cache

register_pytree_node(
  cache_utils.StaticCache,
  _flatten_static_cache,
  _unflatten_static_cache,
)
```

Running again it seems stuck with the following message:

```
/home/hanq_google_com/venv/lib/python3.10/site-packages/jax/_src/interpreters/mlir.py:1135: UserWarning: A large amount of constants were captured during lowering (13.48GB total). If this is intentional, disable this warning by setting JAX_CAPTURED_CONSTANTS_WARN_BYTES=-1. To obtain a report of where these constants were encountered, set JAX_CAPTURED_CONSTANTS_REPORT_FRAMES=-1.
```

What is happening here? 
So turns out, when you `jax.jit`, any data that is used but NOT passed in in
function input arguments will be inlined in the graph as constants. 
Having large constant will make the Graph big, and can make the compile time 
longer. Sometimes, it can also OOM the instruction cache.

Here we only have one function that is applied with `jax.jit` (through `tx.interop.jax_jit`)
which is `  def decode_one_tokens(cur_token, input_pos, cache_position, past_key_values):`
Looking it carefully, we notice that the model weights, which is a big chunk of input, 
is not listed in the input args. Let's fix that.

Replace decode_one_tokens with:
```python
  def decode_one_tokens(model_weights, cur_token, input_pos, cache_position, past_key_values):
    logits, cache = torch.func.functional_call(
        model, 
        model_weights, # weight state_dict
        (cur_token,), # args as tuple
        dict(
          position_ids=input_pos,
          cache_position=cache_position,
          past_key_values=past_key_values,
          return_dict=False,
          use_cache=True) # kwargs as dict
    )
    new_token = torch.argmax(logits[:, -1], dim=-1)[:, None]
    return new_token, cache
```

Here, we added an input arg to `decode_one_tokens`. Now, we need to use
this weight when computing the next logits. We can do so with 
[`torch.func.functional_call`](https://docs.pytorch.org/docs/stable/generated/torch.func.functional_call.html)

Running it again we got:

```
['1', '0', '0', '%', 'but', 'ter', '.', '\n', 'I', '’', 'm', 'not', 'sure', 'if', 'it', '’', 's', 'the', 'but', 'ter', 'or', 'the', 'eggs', ',', 'but', 'I', '’', 'm', 'pretty', 'sure', 'it', '’', 's', 'the', 'but', 'ter', '.', '\n', 'I', '’', 's', '\n', 'I', '’', 's', '\n', 'I', '’', '\n', 'I']
Time:  14.7717966591008
```

Much much faster! The full repro is located at [jax_hg_03.py](jax_hg_03.py).
