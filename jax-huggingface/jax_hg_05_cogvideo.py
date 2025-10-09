import time
import re
import jax
import torch
from diffusers import CogVideoXPipeline
from diffusers.models.autoencoders.vae import DecoderOutput
import torchax
from jax.tree_util import register_pytree_node
from jax.sharding import PartitionSpec as P, NamedSharding
from transformers.modeling_outputs import BaseModelOutputWithPooling, BaseModelOutputWithPastAndCrossAttentions


def setup_jax_config():
    """配置JAX环境参数"""
    # jax.config.update('jax_default_matmul_precision', 'high')
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


# Transformer sharding策略 - FSDP模式（默认）
# 参考: diffusers/cog_tx_splash_attn.py
# 注意：所有模式都以 .weight$ 结尾，这样不会匹配到 bias 等1维参数
transformer_shardings_fsdp = {
    # Attention layers - 在输出维度分片
    r'.*\.to_q\.weight$': (None, 'axis'),
    r'.*\.to_k\.weight$': (None, 'axis'),
    r'.*\.to_v\.weight$': (None, 'axis'),
    r'.*\.to_out.*\.weight$': ('axis', None),
    # Feedforward layers
    r'.*\.ff\.net\.0\.weight$': (None, 'axis'),
    r'.*\.ff\.net\.2\.weight$': ('axis', None),
}

# Transformer sharding策略 - Tensor Parallel模式
transformer_shardings_tp = {
    # Attention layers - 在输入维度分片
    r'.*\.to_q\.weight$': ('axis', None),
    r'.*\.to_k\.weight$': ('axis', None),
    r'.*\.to_v\.weight$': ('axis', None),
    r'.*\.to_out.*\.weight$': (None, 'axis'),
    # Feedforward layers
    r'.*\.ff\.net\.0\.weight$': ('axis', None),
    r'.*\.ff\.net\.2\.weight$': (None, 'axis'),
}

# Text Encoder (T5) sharding策略
# 参考: diffusers/cog_tx_splash_attn.py
# 简化版本，只使用 axis 维度（原始版本使用 axis,dp,sp）
text_encoder_shardings = {
    r'shared\.weight$': ('axis',),
    r'encoder\.block\.\d+\.layer\.\d+\.SelfAttention\.q\.weight$': ('axis',),
    r'encoder\.block\.\d+\.layer\.\d+\.SelfAttention\.k\.weight$': ('axis',),
    r'encoder\.block\.\d+\.layer\.\d+\.SelfAttention\.v\.weight$': ('axis',),
    r'encoder\.block\.\d+\.layer\.\d+\.SelfAttention\.o\.weight$': (None, 'axis'),
    r'encoder\.block\.\d+\.layer\.\d+\.DenseReluDense\.wi_0\.weight$': ('axis',),
    r'encoder\.block\.\d+\.layer\.\d+\.DenseReluDense\.wi_1\.weight$': ('axis',),
    r'encoder\.block\.\d+\.layer\.\d+\.DenseReluDense\.wo\.weight$': (None, 'axis'),
}

# VAE sharding策略
# 对于卷积层，在输出通道维度分片（第0维）
# 参考: diffusers/cog_tx_splash_attn.py 中的 LOGICAL_AXIS_RULES
vae_shardings = {
    # Encoder 卷积层 - 在输出通道分片
    r'encoder\..*\.conv\.weight$': ('axis', None, None, None),
    r'encoder\..*\.conv_in\.weight$': ('axis', None, None, None),
    r'encoder\..*\.conv_out\.weight$': ('axis', None, None, None),
    # Decoder 卷积层 - 在输出通道分片
    r'decoder\..*\.conv\.weight$': ('axis', None, None, None),
    r'decoder\..*\.conv_in\.weight$': ('axis', None, None, None),
    r'decoder\..*\.conv_out\.weight$': ('axis', None, None, None),
    # 其他卷积层
    r'.*\.conv_shortcut\.weight$': ('axis', None, None, None),
}


def shard_weights_transformer(mesh, weights, use_fsdp=True):
    """
    对CogVideoX Transformer模型的权重进行分片
    
    参考 diffusers/cog_tx_splash_attn.py 中的 _shard_weight_dict 函数
    使用正则表达式匹配权重名称，应用对应的分片规则
    
    关键点：
    - 只匹配以 .weight 结尾的参数，避免对1维的 bias 等参数进行错误分片
    - 使用 apply_jax_ (in-place) 而不是 apply_jax
    - 未匹配的参数会被复制到所有设备
    
    Args:
        mesh: JAX设备网格
        weights: 模型权重字典
        use_fsdp: 是否使用FSDP模式（默认True），否则使用Tensor Parallel模式
        
    Returns:
        分片后的权重字典
    """
    # 选择分片策略
    sharding_dict = transformer_shardings_fsdp if use_fsdp else transformer_shardings_tp
    
    result = {}
    for k, v in weights.items():
        # 尝试匹配分片规则
        matched = False
        for target, sharding in sharding_dict.items():
            if re.fullmatch(target, k) is not None:
                # 找到匹配的模式，应用分片（使用 apply_jax_ 进行 in-place 操作）
                v.apply_jax_(jax.device_put, NamedSharding(mesh, P(*sharding)))
                matched = True
                break
        
        if not matched:
            # 没有匹配到任何模式，复制到所有设备
            v.apply_jax_(jax.device_put, NamedSharding(mesh, P()))
        
        result[k] = v
    return result


def shard_weights_text_encoder(mesh, weights):
    """
    对Text Encoder (T5)模型的权重进行分片
    
    参考 diffusers/cog_tx_splash_attn.py 中的text_encoder_shardings
    
    Args:
        mesh: JAX设备网格
        weights: Text Encoder权重字典
        
    Returns:
        分片后的权重字典
    """
    result = {}
    for k, v in weights.items():
        # 尝试匹配分片规则
        matched = False
        for target, sharding in text_encoder_shardings.items():
            if re.fullmatch(target, k) is not None:
                # 找到匹配的模式，应用分片
                v.apply_jax_(jax.device_put, NamedSharding(mesh, P(*sharding)))
                matched = True
                break
        
        if not matched:
            # 没有匹配到任何模式，复制到所有设备
            v.apply_jax_(jax.device_put, NamedSharding(mesh, P()))
        
        result[k] = v
    return result


def shard_weights_vae(mesh, weights):
    """
    对VAE模型的权重进行分片
    
    注意：VAE的卷积层通道数可能不规则（如3、128等），
    在不同设备数（4、8、16等）下可能无法整除。
    因此当前策略是将所有VAE权重复制到所有设备，不进行分片。
    
    Args:
        mesh: JAX设备网格
        weights: VAE权重字典
        
    Returns:
        分片后的权重字典（实际上是复制到所有设备）
    """
    result = {}
    for k, v in weights.items():
        # VAE权重全部复制到所有设备，不进行分片
        # 这样可以在任意数量的设备上工作（4、8、16等）
        v.apply_jax_(jax.device_put, NamedSharding(mesh, P()))
        result[k] = v
    return result


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
    
    将所有模型权重移动到JAX设备并编译关键组件:
    - Transformer: DiT模型的核心网络
    - VAE: 用于编码/解码视频帧
    - Text Encoder: 文本编码器
    
    Args:
        pipe: CogVideoX Pipeline
        
    Returns:
        pipe: 配置后的Pipeline
        env: torchax环境
        mesh: JAX设备网格
    """
    print("\n配置Pipeline以使用JAX...")
    env = torchax.default_env()
    
    # 创建设备网格（用于分片）
    print(f"- 创建设备网格（设备数: {jax.device_count()}）...")
    mesh = jax.make_mesh((jax.device_count(),), ('axis',))
    
    with env:
        # 将权重移动到JAX设备（即：使用jax.Array作为底层存储的tensors）
        print("- 移动模型权重到JAX设备...")
        pipe.to('jax')
        
        # 移动scheduler参数
        move_scheduler_to_jax(pipe.scheduler)
        
        # 对transformer权重进行分片
        print("- 对Transformer权重进行分片...")
        transformer_weights = shard_weights_transformer(mesh, pipe.transformer.state_dict())
        pipe.transformer.load_state_dict(transformer_weights, assign=True, strict=False)
        # 确保所有权重已分片完成
        torchax.interop.call_jax(jax.block_until_ready, transformer_weights)
        
        # 对text_encoder权重进行分片
        print("- 对Text Encoder权重进行分片...")
        text_encoder_weights = shard_weights_text_encoder(mesh, pipe.text_encoder.state_dict())
        pipe.text_encoder.load_state_dict(text_encoder_weights, assign=True, strict=False)
        # 确保所有权重已分片完成
        torchax.interop.call_jax(jax.block_until_ready, text_encoder_weights)
        
        # 对VAE权重进行分片
        print("- 对VAE权重进行分片...")
        vae_weights = shard_weights_vae(mesh, pipe.vae.state_dict())
        pipe.vae.load_state_dict(vae_weights, assign=True, strict=False)
        # 确保所有权重已分片完成
        torchax.interop.call_jax(jax.block_until_ready, vae_weights)
        
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
    return pipe, env, mesh


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
            print(f"- 第一次运行包含JIT编译时间, 因此较慢")
            print(f"- 后续运行使用编译后的代码, 速度显著提升")
        
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
    pipe, env, mesh = setup_pipeline_for_jax(pipe)
    
    # 5. 运行视频生成基准测试
    print("\n" + "=" * 60)
    print("5. 运行视频生成基准测试")
    print("=" * 60)
    
    prompt = "A cat walks on the grass, realistic style."
    
    with env:
        frames, times = run_generation_benchmark(
            pipe,
            prompt,
            num_inference_steps=50,
            num_frames=16,  # 减少帧数以降低显存需求
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
