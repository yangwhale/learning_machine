# How to Run a Hugging Face Model in JAX (Part 4) - Diffusers

In the previous episodes ([ep01](01-run-huggingface-model-in-jax.md), [ep02](02-run-huggingface-model-in-jax.md), [ep03](03-run-huggingface-model-in-jax.md)), we have run the Llama model in Jax using
PyTorch model definitions from HuggingFace transformers and
torchax as the interoperability layer. In this episode, we will
do so for a image generation model.

## The stable diffusion

Let's start with instatiating a Stable Diffusion model from
HuggingFace diffusers, in a simple script. (my case it's saved in `jax_hg_04.py`)



```python
import time
import functools
import jax
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-base")
print(type(pipe))
print(isinstantance(pipe, torch.nn.Module))
```

Running the above you will see

```
<class 'diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline'>
False
```

There we found something unusual: the `StableDiffusionPipeline` is NOT a `torch.nn.Module`.

Recall previously (part 1), to convert a torch model to a JAX callable, we use `torchax.extract_jax`
which only works with `torch.nn.Module`s.

## Components of StableDiffusion Pipeline:

Looking at the `pipe` object above:

```
In [6]: pipe
Out[6]:
StableDiffusionPipeline {
  "_class_name": "StableDiffusionPipeline",
  "_diffusers_version": "0.35.1",
  "_name_or_path": "stabilityai/stable-diffusion-2-base",
  "feature_extractor": [
    "transformers",
    "CLIPImageProcessor"
  ],
  "image_encoder": [
    null,
    null
  ],
  "requires_safety_checker": false,
  "safety_checker": [
    null,
    null
  ],
  "scheduler": [
    "diffusers",
    "PNDMScheduler"
  ],
  "text_encoder": [
    "transformers",
    "CLIPTextModel"
  ],
  "tokenizer": [
    "transformers",
    "CLIPTokenizer"
  ],
  "unet": [
    "diffusers",
    "UNet2DConditionModel"
  ],
  "vae": [
    "diffusers",
    "AutoencoderKL"
  ]
}
```

We can see that the pipeline actually have many components, inspecting it
in REPL, we can see that the components `vae`, `unet` and `text_encoder`
are `torch.nn.Module`'s. They will be our starting point.

## The `torchax.compile` API

For this blog post, we will show case `compile` API in torchax.
This API is like `torch.compile`; except, instead of using torch-inductor to compile
your model; it is powered with `jax.jit`. This way, we will get jax compiled performance
instead of Jax eager model.

This is an wrapper over a `torch.nn.Module`, but will use `jax.jit` on the
forward function of this module. The wrapped `JittableModule` is still a
`torch.nn.Module`; so we could substitute it into the pipeline.

So let's modify the above script to

```python
import time
import functools
import jax
import torch
from diffusers import StableDiffusionPipeline
import torchax

pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-base")

env = torchax.default_env()

prompt = "a photograph of an astronaut riding a horse"

with env:
    # Moves the weights to 'jax' device: i.e. to tensors backed by jax.Array's
    pipe.to('jax')

    pipe.unet = torchax.compile(
        pipe.unet
    )
    pipe.vae = torchax.compile(pipe.vae)
    pipe.text_encoder = torchax.compile(pipe.text_encoder)

    image = pipe(prompt, num_inference_steps=20).images[0]
    image.save('astronaut_png')

```

Running it we got:

```
TypeError: function call_torch at /mnt/disks/hanq/torch_xla/torchax/torchax/interop.py:224 traced for jit returned a value of type <class 'transformers.modeling_outputs.BaseModelOutputWithPooling'> at output component jit, which is not a valid JAX type
```

### Error 1: Pytree's again

Again, we got pytree issues that we deal many times throughout the posts.
This is as simple as registrying it as follows:

```python
from jax.tree_util import register_pytree_node
import jax
def base_model_output_with_pooling_flatten(v):
  return (v.last_hidden_state, v.pooler_output, v.hidden_states, v.attentions), None

def base_model_output_with_pooling_unflatten(aux_data, children):
  return BaseModelOutputWithPooling(*children)

register_pytree_node(
  BaseModelOutputWithPooling,
  base_model_output_with_pooling_flatten,
  base_model_output_with_pooling_unflatten
)
```

Running it again we hit the second error:

```
jax.errors.ConcretizationTypeError: Abstract tracer value encountered where concrete value is expected: traced array with shape bool[]
This occurred in the item() method of jax.Array
The error occurred while tracing the function call_torch at /mnt/disks/hanq/torch_xla/torchax/torchax/interop.py:224 for jit. This concrete value was not available in Python because it depends on the value of the argument kwargs['return_dict'].
```

### Error 2: Static argnames

This error also looks familiar: the pipeline is calling the model passing `return_dict=True` (or `False`, we don't really care);
by default, JAX thinks it's a variable (that can change); but here we should really treat it as a constant (so that) different
values should trigger recompile.

If we are calling `jax.jit` ourselves, we would be passing `static_argnames` to `jax.jit`( See API doc [here](https://docs.jax.dev/en/latest/_autosummary/jax.jit.html)); but here, we are using `torchax.compile`; how do we
do that?

We can do that by passing `CompileOptions` to `torchax.compile`, as so:

```python
    pipe.unet = torchax.compile(
        pipe.unet, torchax.CompileOptions(
            jax_jit_kwargs={'static_argnames': ('return_dict', )}
        )
    )
```

So basically we can pass arbitrary `kwargs` to the underlying `jax.jit` as a dictionary.

Now, let's try again. This time we are greeted with the following, more scary error:

```
Traceback (most recent call last):
  File "/mnt/disks/hanq/learning_machine/jax-huggingface/jax_hg_04.py", line 43, in <module>
    image = pipe(prompt, num_inference_steps=20).images[0]
            ~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/disks/hanq/miniconda3/envs/py13/lib/python3.13/site-packages/torch/utils/_contextlib.py", line 120, in decorate_context
    return func(*args, **kwargs)
  File "/mnt/disks/hanq/miniconda3/envs/py13/lib/python3.13/site-packages/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py", line 1061, in __call__
    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
              ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/disks/hanq/miniconda3/envs/py13/lib/python3.13/site-packages/diffusers/schedulers/scheduling_pndm.py", line 257, in step
    return self.step_plms(model_output=model_output, timestep=timestep, sample=sample, return_dict=return_dict)
           ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/disks/hanq/miniconda3/envs/py13/lib/python3.13/site-packages/diffusers/schedulers/scheduling_pndm.py", line 382, in step_plms
    prev_sample = self._get_prev_sample(sample, timestep, prev_timestep, model_output)
  File "/mnt/disks/hanq/miniconda3/envs/py13/lib/python3.13/site-packages/diffusers/schedulers/scheduling_pndm.py", line 418, in _get_prev_sample
    alpha_prod_t = self.alphas_cumprod[timestep]
                   ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^
  File "/mnt/disks/hanq/torch_xla/torchax/torchax/tensor.py", line 235, in __torch_function__
    return self.env.dispatch(func, types, args, kwargs)
           ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/disks/hanq/torch_xla/torchax/torchax/tensor.py", line 589, in dispatch
    res = op.func(*args, **kwargs)
  File "/mnt/disks/hanq/torch_xla/torchax/torchax/ops/jtorch.py", line 290, in getitem
    indexes = self._env.t2j_iso(indexes)
              ^^^^^^^^^
AttributeError: 'Tensor' object has no attribute '_env'
```

### error 4: need to move scheduler

This almost looks like a bug, so let's investigate it throughly.

Let's run our script under pdb:

```
python -m pdb jax_hf_04.py
```

Going up the stack and printing out the `self.alphas_cumprod`:

```
(Pdb) p type(self.alphas_cumprod)
<class 'torch.Tensor'>
```

we notice that this variable alphas_cumprod did not move the `jax` device
as `pipe.to('jax')` should have done.

Turns out, the scheduler object in `pipe`, of type `PNDMScheduler` is not
a `torch.nn.Module` so tensors on it doesn't get moved with the `to` syntax:

```
(Pdb) p self
PNDMScheduler {
  "_class_name": "PNDMScheduler",
  "_diffusers_version": "0.35.1",
  "beta_end": 0.012,
  "beta_schedule": "scaled_linear",
  "beta_start": 0.00085,
  "clip_sample": false,
  "num_train_timesteps": 1000,
  "prediction_type": "epsilon",
  "set_alpha_to_one": false,
  "skip_prk_steps": true,
  "steps_offset": 1,
  "timestep_spacing": "leading",
  "trained_betas": null
}

(Pdb) p isinstance(self, torch.nn.Module)
False
```

With this identified, we can move it to ourselves.

Let's add the following function
```python

def move_scheduler(scheduler):
  for k, v in scheduler.__dict__.items():
    if isinstance(v, torch.Tensor):
      setattr(scheduler, k, v.to('jax'))

```
and call it right after `pipe.to('jax')`

```python
  pipe.to('jax')
+ move_scheduler(pipe.scheduler)
  ...
```

Adding this makes above script run successfully, and we got our first image:

![](astronaut.png)


## Compiling VAE

Despite that the above script finish running, and despite of this line

```python
  pipe.vae = torchax.compile(pipe.vae)
```

We actually didn't run the compiled version of VAE. Why? Because by default
`torchax.compile` only compiles the `forward` method of the module (as specified [here](https://github.com/pytorch/xla/blob/master/torchax/torchax/__init__.py#L107)). VAE when it 
is getting called, the `decode` method is getting called instead.

```python
    pipe.vae = torchax.compile(
      pipe.vae, 
      torchax.CompileOptions(
        methods_to_compile=['decode'],
        jax_jit_kwargs={'static_argnames': ('return_dict', )}
      )
    )
```
the correct version is to replace it with the above.
Here, the `method_to_compile` option specifies which methods to compile.

Let's add JAX profiling and see the picture:

Let's add the above again after computing the first time, so that 
we are not capturing compile times.

iteration 2 took: 5.942152s

Having the above really makes a difference:

Before compiling the VAE, it takes 5.9 seconds to generate one image on A100 GPU,
after it, it takes 1.07s instead.

```
iteration 0 took: 53.946763s
100%|█████████████████████████████████████████| 20/20 [00:01<00:00, 19.73it/s]
iteration 1 took: 1.074522s
100%|█████████████████████████████████████████| 20/20 [00:01<00:00, 19.82it/s]
iteration 2 took: 1.067044s
```

(Like before, we run the code 3 times to show both the time with and without compilation)

# Conclusion

In the blog we show case that we can run the Stable Diffusion model 
from Huggingface using JAX. The only issues we need to deal with is 
again, pytree registration and static args for compilation. This shows 
that HuggingFace can support JAX as a framework without needing to reimplement
all its model using JAX/flax! 
