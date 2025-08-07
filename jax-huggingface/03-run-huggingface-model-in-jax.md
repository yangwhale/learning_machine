# How to Run a Hugging Face Model in JAX (Part 3)

In the previous posts, [part 1](jax-huggingface/0]1-run-huggingface-model-in-jax.md) and
[part 2](jax-huggingface/02-run-huggingface-model-in-jax.md), we explored how to 
call the `forward` function of a HuggingFace model. Now let's see
how we can run it's autoregressive decoding function. But before that, let's first
dive into how `torchax` works.

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