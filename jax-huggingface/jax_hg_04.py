import time
import functools
import jax
import torch
from diffusers import StableDiffusionPipeline
import torchax

from jax.tree_util import register_pytree_node
import jax
from transformers.modeling_outputs import BaseModelOutputWithPooling

jax.config.update('jax_default_matmul_precision', 'high')


def base_model_output_with_pooling_flatten(v):
  return (v.last_hidden_state, v.pooler_output, v.hidden_states, v.attentions), None

def base_model_output_with_pooling_unflatten(aux_data, children):
  return BaseModelOutputWithPooling(*children)

register_pytree_node(
  BaseModelOutputWithPooling,
  base_model_output_with_pooling_flatten,
  base_model_output_with_pooling_unflatten
)

pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-base")

env = torchax.default_env()

prompt = "a photograph of an astronaut riding a horse"

def move_scheduler(scheduler):
  for k, v in scheduler.__dict__.items():
    if isinstance(v, torch.Tensor):
      setattr(scheduler, k, v.to('jax'))


with env:
    # Moves the weights to 'jax' device: i.e. to tensors backed by jax.Array's
    pipe.to('jax')
    move_scheduler(pipe.scheduler)

    pipe.unet = torchax.compile(
        pipe.unet, torchax.CompileOptions(
            jax_jit_kwargs={'static_argnames': ('return_dict', )}
        )
    )
    pipe.vae = torchax.compile(
      pipe.vae, 
      torchax.CompileOptions(
        methods_to_compile=['decode'],
        jax_jit_kwargs={'static_argnames': ('return_dict', )}
      )
    )
    pipe.text_encoder = torchax.compile(pipe.text_encoder)

    for i in range(3):
      start = time.perf_counter()
      image = pipe(prompt, num_inference_steps=20).images[0]
      end =time.perf_counter()
      print(f'iteration {i} took: {end - start:04f}s')



    image.save('astronaut.png')
