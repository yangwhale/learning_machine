"""
VAE Decoder 独立测试
测试 CogVideoX VAE 的解码能力
"""

import time
import numpy as np
import jax
import torch
import imageio
from diffusers import AutoencoderKLCogVideoX
from diffusers.models.autoencoders.vae import DecoderOutput
import torchax
from jax.tree_util import register_pytree_node
from jax.sharding import PartitionSpec as P, NamedSharding, Mesh
from jax.experimental import mesh_utils
from flax.linen import partitioning as nn_partitioning

# 模型配置
MODEL_NAME = "zai-org/CogVideoX1.5-5B"
VAE_SUBFOLDER = "vae"

# VAE sharding 配置
LOGICAL_AXIS_RULES = (
    ('conv_out', ('tp', 'dp', 'sp')),
    ('conv_in', ('tp', 'dp', 'sp'))
)

# Mesh 分片配置
USE_DP = False
SP_NUM = 1
USE_FSDP = True


def setup_pytree_registrations():
    """注册必要的pytree节点以支持JAX转换"""
    print("注册PyTree节点...")
    
    def model_output_flatten(obj):
        return obj.to_tuple(), type(obj)

    def model_output_unflatten(aux, children):
        return aux(*children)
    
    OUTPUT_CLASSES = [DecoderOutput]
    
    for cls in OUTPUT_CLASSES:
        register_pytree_node(cls, model_output_flatten, model_output_unflatten)
        print(f"  - {cls.__name__} 已注册")


def load_vae(model_name=MODEL_NAME):
    """加载 VAE 模型"""
    print(f"正在加载VAE模型: {model_name}")
    vae = AutoencoderKLCogVideoX.from_pretrained(
        model_name,
        subfolder=VAE_SUBFOLDER
    )
    print("VAE模型加载完成")
    return vae


def shard_weights_vae(mesh, weights):
    """对VAE模型的权重进行分片"""
    result = {}
    for k, v in weights.items():
        v.apply_jax_(jax.device_put, NamedSharding(mesh, P()))
        result[k] = v
    return result


def setup_vae_for_jax(vae):
    """设置VAE以在JAX环境中运行"""
    print("\n配置VAE以使用JAX...")

    tp_dim, dp_dim, sp_dim = jax.device_count(), 1, 1
    
    if USE_DP:
        tp_dim //= 2
        dp_dim = 2
    
    if SP_NUM > 1:
        tp_dim //= SP_NUM
        sp_dim = SP_NUM
    
    print(f"  Mesh 维度: tp_dim={tp_dim}, dp_dim={dp_dim}, sp_dim={sp_dim}")
    
    mesh_devices = mesh_utils.create_device_mesh(
        (tp_dim, dp_dim, sp_dim), 
        allow_split_physical_axes=True
    )
    mesh = Mesh(mesh_devices, ('tp', 'dp', 'sp'))
    
    env = torchax.default_env()
    env._mesh = mesh

    def _move_module_to_xla(module):
        with jax.default_device('cpu'):
            state_dict = module.state_dict()
            state_dict = env.to_xla(state_dict)
            module.load_state_dict(state_dict, assign=True)
    
    with env:
        print("- 将VAE移到XLA并进行分片...")
        _move_module_to_xla(vae)
        vae_weights = shard_weights_vae(mesh, vae.state_dict())
        vae.load_state_dict(vae_weights, assign=True, strict=False)
        torchax.interop.call_jax(jax.block_until_ready, vae_weights)
        
        # print("- 编译VAE...")
        vae = torchax.compile(
            vae,
            torchax.CompileOptions(
                jax_jit_kwargs={'static_argnames': ('return_dict', )}
            )
        )
    
    print("VAE配置完成")
    return vae, env, mesh


def decode_latents(vae, latents, env, verbose=True):
    """使用VAE解码潜在表示"""
    if verbose:
        print("\n解码潜在表示...")
        print(f"  潜在表示形状: {latents.shape}")
    
    with env:
        video = vae.decode(latents).sample
    
    if verbose:
        print(f"  输出视频形状: {video.shape}")
    return video


def test_vae_decoder(vae, env, num_frames=1, height=240, width=432, num_iterations=2):
    """
    测试VAE Decoder的解码能力
    
    Args:
        vae: VAE模型
        env: torchax环境
        num_frames: 视频帧数
        height: 视频高度
        width: 视频宽度
        num_iterations: 运行次数
        
    Returns:
        decoded_video: 解码后的视频
        times: 每次迭代的时间列表
    """
    print(f"\n测试VAE Decoder (帧数={num_frames}, 分辨率={height}x{width}, 迭代次数={num_iterations})...")
    
    # 创建随机latent表示 [B, C_latent, F, H//8, W//8]
    # CogVideoX VAE的latent通道数是16，空间压缩比是8
    latent_channels = 16
    latent_height = height // 8
    latent_width = width // 8
    
    with env:
        latents = torch.randn(
            1, latent_channels, num_frames, latent_height, latent_width,
            dtype=torch.bfloat16
        )
        latents = latents.to('jax')
    
    print(f"创建随机latent表示，形状: {latents.shape}")
    print(f"  - Latent通道数: {latent_channels}")
    print(f"  - Latent空间尺寸: {latent_height}x{latent_width}")
    
    times = []
    decoded_video = None
    
    for i in range(num_iterations):
        print(f"\n{'='*60}")
        if i == 0:
            print(f"迭代 {i+1}/{num_iterations} (包含JIT编译)")
        else:
            print(f"迭代 {i+1}/{num_iterations} (使用已编译代码)")
        print('='*60)
        
        # Decoder解码过程计时
        start = time.perf_counter()
        
        # 解码
        decoded_video = decode_latents(vae, latents, env, verbose=(i==0))
        
        end = time.perf_counter()
        elapsed = end - start
        times.append(elapsed)
        
        print(f"\nDecoder耗时: {elapsed:.4f} 秒")
    
    # 打印性能总结
    print(f"\n{'='*60}")
    print("Decoder性能总结:")
    print('='*60)
    print(f"第一次运行（含编译）: {times[0]:.4f} 秒")
    if len(times) > 1:
        avg_time = sum(times[1:]) / len(times[1:])
        print(f"后续运行平均时间: {avg_time:.4f} 秒")
        print(f"加速比: {times[0] / avg_time:.2f}x")
    
    print(f"\n各次迭代详细时间:")
    for i, t in enumerate(times):
        print(f"  迭代 {i+1}: {t:.4f} 秒")
    
    return decoded_video, times


def save_video_frames(video_tensor, output_path, fps=8):
    """
    保存视频帧到文件
    
    Args:
        video_tensor: 视频张量 [B, C, F, H, W]
        output_path: 输出文件路径
        fps: 帧率
    """
    print(f"\n保存视频到: {output_path}")
    
    # 先转换为float32（因为bfloat16不支持某些操作）
    video_tensor = video_tensor.to(torch.float32)
    
    # 转换为numpy并调整范围，使用copy()避免不可写警告
    video_np = video_tensor[0].permute(1, 2, 3, 0).cpu().numpy().copy()  # [F, H, W, C]
    video_np = ((video_np + 1.0) / 2.0 * 255).clip(0, 255).astype(np.uint8)
    
    # 保存 - 使用macro_block_size=1避免自动调整分辨率
    imageio.mimsave(output_path, video_np, fps=fps, macro_block_size=1)
    print(f"视频已保存 (共{len(video_np)}帧)")
    print(f"  分辨率: {video_np.shape[2]}x{video_np.shape[1]}")


def main():
    """主函数"""
    # 设置JAX编译缓存
    jax.config.update("jax_compilation_cache_dir", "/dev/shm/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    
    # 设置默认精度
    torch.set_default_dtype(torch.bfloat16)
    
    # 注册PyTree节点
    setup_pytree_registrations()
    
    # 加载VAE
    vae = load_vae(MODEL_NAME)
    
    # 配置VAE使用JAX
    vae, env, mesh = setup_vae_for_jax(vae)
    
    # 在mesh上下文中运行测试
    with mesh, nn_partitioning.axis_rules(LOGICAL_AXIS_RULES), env:
        # 测试VAE Decoder（运行2次：第1次含编译，第2次使用编译后代码）
        # 使用16的倍数分辨率避免视频编码警告 (432x240, 16:9比例)
        decoded_video, times = test_vae_decoder(
            vae,
            env,
            num_frames=11,
            height=56,
            width=104,
            num_iterations=2
        )
        
        # 保存解码后的视频
        save_video_frames(decoded_video, 'vae_decoded.mp4')
    
    print("\n✅ VAE Decoder独立运行测试完成！")
    print("\n生成的文件:")
    print("  - vae_decoded.mp4: Decoder解码后的视频")


if __name__ == "__main__":
    main()