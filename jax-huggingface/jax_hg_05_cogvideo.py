import time
import functools
import jax
import torch
from diffusers import CogVideoXPipeline
from diffusers.models.autoencoders.vae import DecoderOutput
import torchax
from jax.tree_util import register_pytree_node
from transformers.modeling_outputs import BaseModelOutputWithPooling, BaseModelOutputWithPastAndCrossAttentions


def setup_jax_config():
    """配置JAX环境参数"""
    jax.config.update('jax_default_matmul_precision', 'high')
    print("JAX配置: 使用高精度矩阵乘法")


def setup_pytree_registrations():
    """
    注册必要的pytree节点以支持JAX转换
    
    为各种transformers输出类型注册flatten和unflatten方法，
    使其可以在JAX的函数转换中正常使用
    """
    print("注册transformers输出类型为pytree节点...")
    
    # 注册 BaseModelOutputWithPooling
    def base_model_output_with_pooling_flatten(v):
        """将BaseModelOutputWithPooling展平为元组"""
        return (v.last_hidden_state, v.pooler_output, v.hidden_states, v.attentions), None

    def base_model_output_with_pooling_unflatten(aux_data, children):
        """从元组重建BaseModelOutputWithPooling"""
        return BaseModelOutputWithPooling(*children)

    register_pytree_node(
        BaseModelOutputWithPooling,
        base_model_output_with_pooling_flatten,
        base_model_output_with_pooling_unflatten
    )
    print("  - BaseModelOutputWithPooling 已注册")
    
    # 注册 BaseModelOutputWithPastAndCrossAttentions
    def base_model_output_with_past_flatten(v):
        """将BaseModelOutputWithPastAndCrossAttentions展平为元组"""
        return (
            v.last_hidden_state,
            v.past_key_values,
            v.hidden_states,
            v.attentions,
            v.cross_attentions
        ), None

    def base_model_output_with_past_unflatten(aux_data, children):
        """从元组重建BaseModelOutputWithPastAndCrossAttentions"""
        return BaseModelOutputWithPastAndCrossAttentions(*children)

    register_pytree_node(
        BaseModelOutputWithPastAndCrossAttentions,
        base_model_output_with_past_flatten,
        base_model_output_with_past_unflatten
    )
    print("  - BaseModelOutputWithPastAndCrossAttentions 已注册")
    
    # 注册 DecoderOutput
    def decoder_output_flatten(v):
        """将DecoderOutput展平为元组"""
        return (v.sample,), None

    def decoder_output_unflatten(aux_data, children):
        """从元组重建DecoderOutput"""
        return DecoderOutput(sample=children[0])

    register_pytree_node(
        DecoderOutput,
        decoder_output_flatten,
        decoder_output_unflatten
    )
    print("  - DecoderOutput 已注册")


def load_cogvideo_pipeline(model_name="zai-org/CogVideoX-2b"):
    """
    加载CogVideoX Pipeline
    
    Args:
        model_name: 预训练模型名称
        
    Returns:
        pipe: CogVideoX Pipeline实例
    """
    print(f"正在加载模型: {model_name}")
    pipe = CogVideoXPipeline.from_pretrained(model_name)
    print("模型加载完成")
    return pipe


def move_scheduler_to_jax(scheduler):
    """
    将scheduler的所有Tensor移动到JAX设备
    
    Args:
        scheduler: 调度器对象
    """
    print("将scheduler参数移动到JAX设备...")
    for k, v in scheduler.__dict__.items():
        if isinstance(v, torch.Tensor):
            setattr(scheduler, k, v.to('jax'))


def setup_pipeline_for_jax(pipe):
    """
    设置Pipeline以在JAX环境中运行
    
    将所有模型权重移动到JAX设备并编译关键组件：
    - Transformer: DiT模型的核心网络
    - VAE: 用于编码/解码视频帧
    - Text Encoder: 文本编码器
    
    Args:
        pipe: CogVideoX Pipeline
        
    Returns:
        pipe: 配置后的Pipeline
    """
    print("\n配置Pipeline以使用JAX...")
    env = torchax.default_env()
    
    with env:
        # 将权重移动到JAX设备（即：使用jax.Array作为底层存储的tensors）
        print("- 移动模型权重到JAX设备...")
        pipe.to('jax')
        
        # 移动scheduler参数
        move_scheduler_to_jax(pipe.scheduler)
        
        # 编译transformer（DiT的核心网络）
        print("- 编译transformer...")
        pipe.transformer = torchax.compile(
            pipe.transformer, 
            torchax.CompileOptions(
                jax_jit_kwargs={'static_argnames': ('return_dict', )}
            )
        )
        
        # 编译VAE的decode方法（用于将latent转换为视频帧）
        print("- 编译VAE...")
        pipe.vae = torchax.compile(
            pipe.vae, 
            torchax.CompileOptions(
                methods_to_compile=['decode'],
                jax_jit_kwargs={'static_argnames': ('return_dict', )}
            )
        )
        
        # 编译文本编码器
        print("- 编译Text Encoder...")
        pipe.text_encoder = torchax.compile(pipe.text_encoder)
    
    print("Pipeline配置完成")
    return pipe, env


def run_generation_benchmark(pipe, prompt, num_inference_steps=50, num_frames=49, num_iterations=3):
    """
    运行视频生成基准测试
    
    Args:
        pipe: CogVideoX Pipeline
        prompt: 文本提示
        num_inference_steps: 推理步数
        num_frames: 视频帧数
        num_iterations: 迭代次数
        
    Returns:
        frames: 最后生成的视频帧
        times: 各次迭代的时间列表
    """
    print(f"\n运行{num_iterations}次视频生成基准测试...")
    print(f"提示词: '{prompt}'")
    print(f"推理步数: {num_inference_steps}")
    print(f"视频帧数: {num_frames}")
    
    times = []
    frames = None
    
    for i in range(num_iterations):
        start = time.perf_counter()
        result = pipe(prompt, num_inference_steps=num_inference_steps, num_frames=num_frames)
        frames = result.frames[0]  # CogVideoX 返回 frames 而不是 images
        end = time.perf_counter()
        elapsed = end - start
        times.append(elapsed)
        print(f"迭代 {i}: {elapsed:.4f} 秒")
    
    return frames, times


def print_performance_summary(times):
    """
    打印性能统计摘要
    
    Args:
        times: 时间列表
    """
    print("\n" + "=" * 60)
    print("性能统计摘要")
    print("=" * 60)
    
    if len(times) > 0:
        print(f"总迭代次数: {len(times)}")
        print(f"第一次运行（含编译）: {times[0]:.4f} 秒")
        
        if len(times) > 1:
            avg_time = sum(times[1:]) / len(times[1:])
            print(f"后续运行平均时间: {avg_time:.4f} 秒")
            print(f"加速比: {times[0] / avg_time:.2f}x")
            print(f"\n说明:")
            print(f"- 第一次运行包含JIT编译时间，因此较慢")
            print(f"- 后续运行使用编译后的代码，速度显著提升")
        
        print(f"\n各次迭代详细时间:")
        for i, t in enumerate(times):
            print(f"  迭代 {i}: {t:.4f} 秒")


def main():
    """主函数"""
    print("=" * 60)
    print("JAX + CogVideoX 视频生成示例")
    print("=" * 60)
    
    # 1. 配置JAX环境
    print("\n1. 配置JAX环境...")
    setup_jax_config()
    
    # 2. 设置pytree注册
    print("\n2. 设置pytree注册...")
    setup_pytree_registrations()
    
    # 3. 加载CogVideoX模型
    print("\n3. 加载CogVideoX模型...")
    pipe = load_cogvideo_pipeline("zai-org/CogVideoX-2b")
    
    # 4. 配置Pipeline以使用JAX
    print("\n4. 配置Pipeline以使用JAX...")
    pipe, env = setup_pipeline_for_jax(pipe)
    
    # 5. 运行视频生成基准测试
    print("\n" + "=" * 60)
    print("5. 运行视频生成基准测试")
    print("=" * 60)
    
    prompt = "A cat walks on the grass, realistic style."
    
    with env:
        frames, times = run_generation_benchmark(
            pipe,
            prompt,
            num_inference_steps=5,
            num_frames=2,  # 减少帧数以降低显存需求
            num_iterations=3
        )
    
    # 6. 保存生成的视频
    print("\n6. 保存生成的视频...")
    output_path = 'output_video.mp4'
    # 导入必要的库来保存视频
    import imageio
    imageio.mimsave(output_path, frames, fps=8)
    print(f"视频已保存到: {output_path}")
    
    # 7. 打印性能摘要
    print_performance_summary(times)
    
    print("\n" + "=" * 60)
    print("视频生成完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
