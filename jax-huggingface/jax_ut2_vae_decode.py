import time
import numpy as np
import jax
import torch
from diffusers import AutoencoderKLCogVideoX
from diffusers.models.autoencoders.vae import DecoderOutput
import torchax
from jax.tree_util import register_pytree_node
from jax.sharding import PartitionSpec as P, NamedSharding, Mesh
from jax.experimental import mesh_utils
from flax.linen import partitioning as nn_partitioning
from matplotlib.ticker import FormatStrFormatter
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import csv
import os

MODEL_NAME = "zai-org/CogVideoX1.5-5B"
VAE_SUBFOLDER = "vae"

LOGICAL_AXIS_RULES = (
    ('conv_out', ('tp', 'dp', 'sp')),
    ('conv_in', ('tp', 'dp', 'sp'))
)

USE_DP = False
SP_NUM = 1
USE_FSDP = True

def setup_pytree_registrations():
    print("注册PyTree节点...")
    def model_output_flatten(obj):
        return obj.to_tuple(), type(obj)
    def model_output_unflatten(aux, children):
        return aux(*children)
    for cls in [DecoderOutput]:
        register_pytree_node(cls, model_output_flatten, model_output_unflatten)
        print(f"  - {cls.__name__} 已注册")

def record_time(call_method):
    start = time.time()
    output = call_method()
    if hasattr(output, 'sample'):
        jax.block_until_ready(output.sample)
    else:
        jax.block_until_ready(output)
    return output, (time.time() - start) * 1000

def plot_results(csv_file='memory_usage.csv'):
    df = pd.read_csv(csv_file)
    df['time'] = df['time'] / 1000
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_xlabel('Number of Frames')
    ax1.set_ylabel('Time Cost (seconds)', color='tab:blue')
    ax1.plot(df['frames'], df['time'], marker='o', linestyle='-', 
             color='tab:blue', label='Time Cost')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    
    plt.title('JAX VAE Decode Performance')
    ax1.legend(loc='upper left')
    fig.tight_layout()
    
    plot_filename = os.path.splitext(csv_file)[0] + '.png'
    plt.savefig(plot_filename, dpi=150)
    plt.close()
    print(f"Plot saved to {plot_filename}")

def save_to_csv(results, filename='memory_usage.csv'):
    if not results:
        print(f"WARNING: No results to save. Creating empty CSV file: {filename}")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w', newline='') as f:
            csv.DictWriter(f, ['frames', 'time']).writeheader()
        return
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, results[0].keys())
        writer.writeheader()
        writer.writerows(results)

def load_vae(model_name=MODEL_NAME):
    print(f"正在加载VAE模型: {model_name}")
    vae = AutoencoderKLCogVideoX.from_pretrained(model_name, subfolder=VAE_SUBFOLDER).to(dtype=torch.bfloat16)
    vae.enable_slicing()
    vae.enable_tiling()
    print("VAE模型加载完成")
    return vae

def shard_weights_vae(mesh, weights):
    result = {}
    for k, v in weights.items():
        v.apply_jax_(jax.device_put, NamedSharding(mesh, P()))
        result[k] = v
    return result

def setup_vae_for_jax(vae):
    print("\n配置VAE以使用JAX...")
    tp_dim, dp_dim, sp_dim = jax.device_count(), 1, 1
    
    if USE_DP:
        tp_dim //= 2
        dp_dim = 2
    if SP_NUM > 1:
        tp_dim //= SP_NUM
        sp_dim = SP_NUM
    
    print(f"  Mesh 维度: tp_dim={tp_dim}, dp_dim={dp_dim}, sp_dim={sp_dim}")
    
    mesh_devices = mesh_utils.create_device_mesh((tp_dim, dp_dim, sp_dim), allow_split_physical_axes=True)
    mesh = Mesh(mesh_devices, ('tp', 'dp', 'sp'))
    env = torchax.default_env()
    env._mesh = mesh

    def _move_module_to_xla(module):
        with jax.default_device('cpu'):
            state_dict = module.state_dict()
            for k, v in state_dict.items():
                if hasattr(v, 'dtype') and v.dtype == torch.float32:
                    state_dict[k] = v.to(torch.bfloat16)
            state_dict = env.to_xla(state_dict)
            module.load_state_dict(state_dict, assign=True)
    
    with env:
        print("- 将VAE移到XLA并进行分片...")
        _move_module_to_xla(vae)
        vae_weights = shard_weights_vae(mesh, vae.state_dict())
        vae.load_state_dict(vae_weights, assign=True, strict=False)
        torchax.interop.call_jax(jax.block_until_ready, vae_weights)
        vae = torchax.compile(vae, torchax.CompileOptions(jax_jit_kwargs={'static_argnames': ('return_dict', )}))
    
    print("VAE配置完成")
    return vae, env, mesh

def vae_decode_test(vae, env, max_frames=100, step=10, height=240, width=432):
    latent_channels, latent_height, latent_width = 16, height // 8, width // 8
    results = []
    
    for frames in tqdm(range(10, max_frames + 1, step), desc="Testing JAX VAE Decode"):
        latent_frames = max(1, frames // 4)
        try:
            with env:
                latents = torch.randn(1, latent_channels, latent_frames, latent_height, latent_width, dtype=torch.bfloat16).to('jax')
                output, time_cost = record_time(lambda: vae.decode(latents).sample)
                del output, latents
            results.append({'frames': frames, 'time': time_cost})
        except Exception as e:
            print(f"Error at {frames} frames: {str(e)}")
            break
    return results

def main():
    jax.config.update("jax_compilation_cache_dir", "/dev/shm/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    torch.set_default_dtype(torch.bfloat16)
    
    setup_pytree_registrations()
    vae = load_vae(MODEL_NAME)
    vae, env, mesh = setup_vae_for_jax(vae)
    
    print("--- Starting JAX VAE Decode Performance Test ---")
    
    with mesh, nn_partitioning.axis_rules(LOGICAL_AXIS_RULES), env:
        results = vae_decode_test(vae, env, max_frames=21, step=10, height=240, width=432)
        
        csv_filename = 'test_result/jax_vae_decode_performance.csv'
        save_to_csv(results, csv_filename)
        print(f"Results saved to {csv_filename}")
        plot_results(csv_filename)
    
    print("\n✅ JAX VAE Decode测试完成！")

if __name__ == "__main__":
    main()