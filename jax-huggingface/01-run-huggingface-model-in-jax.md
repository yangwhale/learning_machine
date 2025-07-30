# How to Run a Hugging Face Model in JAX (Part 1)

Hugging Face recently removed native JAX and TensorFlow support from its `transformers` library, aiming to streamline its codebase. This decision left many JAX users wondering how they could continue leveraging Hugging Face's vast collection of models without reimplementing everything in JAX.

This blog post explores a solution: running PyTorch-based Hugging Face models with JAX inputs. This approach offers a valuable "way out" for JAX users who rely on Hugging Face models.

## Background & Approach

As the author of [torchax](https://github.com/pytorch/xla/tree/master/torchax), a nascent library designed for seamless interoperability between JAX and PyTorch, this exploration serves as an excellent stress test for `torchax`. Let's dive in\!

-----

## Setup

We'll begin with the standard Hugging Face quickstart setup. If you haven't already, set up your environment:

```bash
# Create venv / conda env; activate etc.
pip install huggingface-cli
huggingface-cli login # Set up your Hugging Face token
pip install -U transformers datasets evaluate accelerate timm flax
```

Next, install `torchax` directly from its latest development version:

```bash
pip install torchax
pip install jax[tpu] # or jax[cuda12] if you are on GPU https://docs.jax.dev/en/latest/installation.html
```

-----

## First Attempt: Eager Mode

Let's start by instantiating a model and tokenizer. We'll create a script named `jax_hg_01.py` with the following code:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import jax # Import jax here for later use

# Load a PyTorch model and tokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype="bfloat16", device_map="cpu")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Tokenize input, requesting JAX arrays
model_inputs = tokenizer(["The secret to baking a good cake is "], return_tensors="jax")
print(model_inputs)
```

Notice the crucial `return_tensors="jax"` in the tokenizer call. This instructs Hugging Face to return JAX arrays directly, which is essential for our goal of using JAX inputs with a PyTorch model. Running the above script will output:

```
{'input_ids': Array([[    1,   450,  7035,   304,   289,  5086,   263,  1781,   274,
         1296,   338, 29871]], dtype=int32), 'attention_mask': Array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=int32)}
```

Now, let's employ `torchax` to convert this PyTorch model into a JAX callable. Modify your script as follows:

```python
import torchax
# ... (previous code)

weights, func = torchax.extract_jax(model)
```

The `torchax.extract_jax` function converts the model's `forward` method into a JAX-compatible callable. It also returns the model's weights as a Pytree of JAX arrays (which is essentially the `model.state_dict()` converted to JAX arrays).

With `func` and `weights` in hand, we can now call this JAX function. The convention is to pass the `weights` as the first argument, followed by a tuple for positional arguments (`args`), and finally an optional dictionary for keyword arguments (`kwargs`).

Let's add the call to our script:

```python
# ... (previous code)

print(func(weights, (model_inputs.input_ids, )))
```

Executing this will produce the following output, demonstrating successful eager-mode execution:

```
In [2]: import torchax

In [3]: weights, func = torchax.extract_jax(model)
WARNING:root:Duplicate op registration for aten.__and__

In [4]: print(func(weights, (model_inputs.input_ids, )))
CausalLMOutputWithPast(loss=None, logits=Array([[[-12.950611  ,  -7.4854484 ,  -0.42371067, ...,  -6.819363  ,
          -8.073828  ,  -7.5583534 ],
        [-13.508438  , -11.716616  ,  -6.9578876 , ...,  -9.135823  ,
         -10.237023  ,  -8.56888   ],
        [-12.8517685 , -11.180469  ,  -4.0543456 , ...,  -7.9564795 ,
         -11.546011  , -10.686134  ],
        ...,
        [ -2.983235  ,  -5.621302  ,  11.553352  , ...,  -2.6286669 ,
          -2.8319468 ,  -1.9902805 ],
        [ -8.674949  , -10.042385  ,   3.4400458 , ...,  -3.7776647 ,
          -8.616567  ,  -5.7228904 ],
        [ -4.0748825 ,  -4.706395  ,   5.117742  , ...,   6.7174563 ,
           0.5748794 ,   2.506649  ]]], dtype=float32), past_key_values=DynamicCache(), hidden_states=None, attentions=None)
```

To pass keyword arguments (kwargs) to the function, you can simply add them as the third argument:

```python
print(func(weights, (model_inputs.input_ids, ), {'use_cache': False}))
```

While this demonstrates basic functionality, the true power of JAX lies in its **JIT compilation**. Just-In-Time (JIT) compilation can significantly accelerate computations, especially on accelerators like GPUs and TPUs. So, our next logical step is to `jax.jit` our function.

-----

## Jitting - Fiddling with Pytrees

In JAX, JIT compilation is as simple as wrapping your function with `jax.jit`. Let's try that:

```python
import jax
# ... (previous code)

func_jit = jax.jit(func)
res = func_jit(weights, (model_inputs.input_ids,))
```

Running this code will likely result in a `TypeError`:

```
The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/hanq_google_com/learning_machine/jax-huggingface/script.py", line 18, in <module>
    res = func_jit(weights, (model_inputs.input_ids,))
TypeError: function jax_func at /home/hanq_google_com/pytorch/xla/torchax/torchax/__init__.py:52 traced for jit returned a value of type <class 'transformers.modeling_outputs.CausalLMOutputWithPast'>, which is not a valid JAX type
```

The error message indicates that JAX doesn't understand the `CausalLMOutputWithPast` type. When you `jax.jit` a function, JAX requires that all inputs and outputs are "JAX types"—meaning they can be flattened into a list of JAX-understood elements using `jax.tree.flatten`.

To resolve this, we need to register these custom types with **JAX's Pytree system**. Pytrees are nested data structures (like tuples, lists, and dictionaries) that JAX can traverse and apply transformations to. By registering a custom type, we tell JAX how to decompose it into its constituent parts (children) and reconstruct it.

Add the following to your script:

```python
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
```

This code snippet defines how `CausalLMOutputWithPast` objects should be flattened (into a tuple of their internal components) and unflattened (reconstructed from those components). Now, JAX can properly handle this type.

However, upon running the script again, you'll encounter a similar error:

```
Traceback (most recent call last):
  File "/home/hanq_google_com/learning_machine/jax-huggingface/script.py", line 33, in <module>
    res = func_jit(weights, (model_inputs.input_ids,))
TypeError: function jax_func at /home/hanq_google_com/pytorch/xla/torchax/torchax/__init__.py:52 traced for jit returned a value of type <class 'transformers.cache_utils.DynamicCache'> at output component [1], which is not a valid JAX type
```

The same Pytree registration trick is needed for `transformers.cache_utils.DynamicCache`:

```python
from transformers import cache_utils

def _flatten_dynamic_cache(dynamic_cache):
  return (
      dynamic_cache.key_cache,
      dynamic_cache.value_cache,
  ), None

def _unflatten_dynamic_cache(aux, children):
  cache = cache_utils.DynamicCache()
  cache.key_cache, cache.value_cache = children
  return cache

register_pytree_node(
  cache_utils.DynamicCache,
  _flatten_dynamic_cache,
  _unflatten_dynamic_cache,
)
```

With these registrations, we're past the Pytree type issues. However, another common JAX error arises:

```
jax.errors.ConcretizationTypeError: Abstract tracer value encountered where concrete value is expected: traced array with shape bool[]
This occurred in the item() method of jax.Array
The error occurred while tracing the function jax_func at /home/hanq_google_com/pytorch/xla/torchax/torchax/__init__.py:52 for jit. This concrete value was not available in Python because it depends on the value of the argument kwargs['use_cache'].

See https://docs.jax.dev/en/latest/errors.html#jax.errors.ConcretizationTypeError
```

-----

## Static Arguments

This `ConcretizationTypeError` is a classic JAX issue. When you `jax.jit` a function, JAX traces its execution to build a computation graph. During this tracing, it treats all inputs as *tracers*—symbolic representations of values—rather than their concrete values. The error arises because the `if use_cache and past_key_values is None:` condition attempts to read the actual boolean value of `use_cache`, which is not available during tracing.

There are two primary ways to fix this:

1.  Using `static_argnums` in `jax.jit` to explicitly tell JAX which arguments are compile-time constants.
2.  Using a **closure** to "bake in" the constant value.

For this example, we'll demonstrate the closure method. We'll define a new function that encapsulates the constant `use_cache` value and then JIT that function:

```python
import time
# ... (previous code including jax.tree_util imports and pytree registrations)

def func_with_constant(weights, input_ids):
  res = func(weights, (input_inputs_ids, ), {'use_cache': False}) # Pass use_cache as a fixed value
  return res

jitted_func = jax.jit(func_with_constant)
res = jitted_func(weights, model_inputs.input_ids)
print(res)
```

Running this updated script finally yields the expected output, matching our eager-mode results:

```
CausalLMOutputWithPast(loss=Array([[[-12.926737  ,  -7.455758  ,  -0.42932802, ...,  -6.822556  ,
          -8.060653  ,  -7.5620213 ],
        [-13.511845  , -11.716769  ,  -6.9498663 , ...,  -9.14628   ,
         -10.245605  ,  -8.572137  ],
        [-12.842418  , -11.174898  ,  -4.0682483 , ...,  -7.9594035 ,
         -11.54412   , -10.675278  ],
        ...,
        [ -2.9683495 ,  -5.5914016 ,  11.563716  , ...,  -2.6254666 ,
          -2.8206763 ,  -1.9780521 ],
        [ -8.675585  , -10.044738  ,   3.4449315 , ...,  -3.7793014 ,
          -8.6158495 ,  -5.729558  ],
        [ -4.0751734 ,  -4.69619   ,   5.111123  , ...,   6.733637  ,
           0.57132554,   2.524692  ]]], dtype=float32), logits=None, past_key_values=None, hidden_states=None, attentions=None)
```

We've successfully converted a PyTorch model into a JAX function, made it compatible with `jax.jit`, and executed it\!

A key characteristic of JIT-compiled functions is their performance profile: the first run is typically slower due to compilation, but subsequent runs are significantly faster. Let's verify this by timing a few runs:

```python
for i in range(3):
  start = time.time()
  res = jitted_func(weights, model_inputs.input_ids)
  jax.block_until_ready(res) # Ensure computation is complete
  end = time.time()
  print(i, end - start, 'seconds')
```

On a Google Cloud TPU v6e, the results clearly demonstrate the JIT advantage:

```
0 4.365400552749634 seconds
1 0.01341700553894043 seconds
2 0.013022422790527344 seconds
```

The first run took over 4 seconds, while subsequent runs completed in milliseconds. This is the power of JAX's compilation\!

The full script for this example can be found at `jax_hg_01.py` in the accompanying repository.

-----

## Conclusion

This exploration demonstrates that running a `torch.nn.Module` from Hugging Face within JAX is indeed feasible, though it requires addressing a few "rough edges." The primary challenges involved registering Hugging Face's custom output types with JAX's Pytree system and managing static arguments for JIT compilation.

In the future, an adapter library could pre-register common Hugging Face pytrees and provide a smoother integration experience for JAX users.

## Next Steps

We've laid the groundwork\! In the next installment, we'll delve into:

  * **Decoding a Sentence:** Demonstrating how to use `model.generate` for text generation within this JAX-PyTorch setup.
  * **Tensor Parallelism:** Showing how to scale this solution to run on multiple TPUs (e.g., 8 TPUs) for accelerated inference.

Stay tuned\!
