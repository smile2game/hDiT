from functools import partial
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.configuration_utils import ConfigMixin,register_to_config
from diffusers.utils import is_torch_version
from diffusers.utils.torch_utils import maybe_allow_in_graph

from diffusers.models.attention import Attention,FeedForward
from diffusers.models.embeddings import TimestepEmbedding,Timesteps,get_3d_sincos_pos_embed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin #混用与复合

from hfuser.core.distributed.comm import all_to_all_comm,gather_squence,split_sequence,get_pad,set_pad
from hfuser.core.distributed.parallel_mgr import ParallelManager
from ..modules.embeddings import CogVideoXPatchEmbed,apply_rotary_emb
from ..modules.normalization import AdaLayerNorm,CogVideoXLayerNormZero

class CogVideoXAttnProcessor2_0: 
    """
    Transformer3DModel >> Block >> Attention >> CogVideoXAttnProcessor2_0
    """
    def __init__(self):
        if not hasattr(F,"scaled_dot_product_attention"):
            raise ImportError("CogVideoXAttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")




@maybe_allow_in_graph
class CogVideoXBlock(nn.Module):
    """
    Parameters:
        dim (`int`):
            The number of channels in the input and output.
        qk_norm (`bool`):
            Whether to apply layer normalization to the query and key.
        norm_elementwise_affine (`bool`):
            Whether to apply elementwise affine(逐元素仿射变换) transformation to the normalized output.
        norm_eps (`float`):
            The epsilon value for layer normalization.
        final_dropout (`bool`):
            Whether to apply dropout after the final feed-forward layer.
        ff_inner_dim (`int`):
            The inner dimension of the feed-forward layer.
            FFN(x) = Gelu(xW1 + b1)W2 + b2
        ff_bias (`bool`):
            Whether to use bias in the feed-forward layer.
        attention_out_bias (`bool`):
            Whether to use bias in the attention output projection layer.
        block_idx (`int`):
            The index of the block in the transformer.
    """
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: int,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = False,
        qk_norm: bool = True,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = True,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
        block_idx: int = 0,
    ):
        super().__init__()
        
        #1. Self Attention
        self.norm1 = CogVideoXLayerNormZero(time_embed_dim,dim,norm_elementwise_affine,norm_eps,bias=True)

        self.attn1 = Attention(
            query_dim=dim,
            dim_head = attention_head_dim,
            heads = num_attention_heads,
            qk_norm = "layer_norm" if qk_norm else None,
            eps = 1e-6,
            bias = attention_bias,
            out_bias = attention_out_bias,
            processor = CogVideoXAttnProcessor2_0(),
        )

        # parallel
        self.attn1.parallel_manager = None 

        # 2. Feed Forward
        self.norm2 = CogVideoXLayerNormZero(time_embed_dim,dim,norm_elementwise_affine,norm_eps,bias=True)

        self.ff = FeedForward(
            dim,
            dropout = dropout, #不同于attention的地方,dropout和inner_dim
            activation_fn = activation_fn,
            final_dropout = final_dropout,
            inner_dim = ff_inner_dim,
            bias = ff_bias,
        )
        #无需pab
        self.block_idx = block_idx


    def forward





class CogVideoXTransformer3DModel(ModelMixin,ConfigMixin):
    """
    inherite from diffusers ModelMixin,ConfigMixin

    Parameters:
        num_attention_heads (`int`, defaults to `30`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, defaults to `64`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, *optional*, defaults to `16`):
            The number of channels in the output.
        flip_sin_to_cos (`bool`, defaults to `True`):
            Whether to flip the sin and cos in the positional embeddings.
        time_embed_dim (`int`, defaults to `512`):
            The dimension of the time embeddings.
        text_embed_dim (`int`, defaults to `4096`):
            The dimension of the text embeddings.
        num_layers (`int`, defaults to `30`):
            The number of layers in the transformer.
        dropout (`float`, defaults to `0.0`):
            The dropout rate.
        attention_bias (`bool`, defaults to `True`):
            Whether to use bias in the attention layer.
        sample_width (`int`, defaults to `90`):
            The width of the sample.
        sample_height (`int`, defaults to `60`):
            The height of the sample.
        sample_frames (`int`, defaults to `49`):
            The number of frames in the sample.
        patch_size (`int`, defaults to `2`):
            The size of the patches.
        temporal_compression_ratio (`int`, defaults to `4`):
            The temporal compression ratio.
    """
    @register_to_config #decorator to inherit from ConfigMixin's __init__, send args to register_to_config
    def __init__(
        self,
        num_attention_heads: int = 30,
        attention_head_dim: int = 64,
        in_channels: int = 16,
        out_channels: Optional[int] = 16,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        time_embed_dim: int = 512,
        text_embed_dim: int = 4096,
        num_layers: int = 30,
        dropout: float = 0.0,
        attention_bias: bool = True,
        sample_width: int = 90,
        sample_height: int = 60,
        sample_frames: int = 49,
        patch_size: int = 2,
        temporal_compression_ratio: int = 4, #时间压缩比
        max_text_seq_length: int = 226,
        activation_fn: str = "gelu-approximate",
        timestep_activation_fn: str = "silu",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        spatial_interpolation_scale: float = 1.875, #空间插值比例
        temporal_interpolation_scale: float = 1.0, #时间插值比例
        use_rotary_positional_embeddings: bool = False,
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim #总维度 = 30 x 64 = 1920

        post_patch_height = sample_height // patch_size #60 // 2 = 30
        post_patch_width = sample_width // patch_size #90 // 2 = 45
        post_time_compression_frames = (sample_frames - 1) // temporal_compression_ratio + 1 #(49 - 1) // 4 + 1 = 13

        self.num_patches = post_patch_height * post_patch_width * post_time_compression_frames #30 * 45 * 13 = 17550

        #1. Patch embedding
        self.patch_embed = CogVideoXPatchEmbed(patch_size,in_channels,inner_dim,text_embed_dim,bias=True) 
        self.embedding_dropout = nn.Dropout(dropout)

        #2. 3D positional embeddings 13 x 1350 x 1920,增添位置信息,和文本信息，patch_embed拼接起来
        spatial_pos_embedding = get_3d_sincos_pos_embed(
            inner_dim,
            (post_patch_width,post_patch_height),
            post_time_compression_frames,
            spatial_interpolation_scale,
            temporal_interpolation_scale,
        ) 
        spatial_pos_embedding = torch.from_numpy(spatial_pos_embedding).flatten(0,1) #17550,1920
        pos_embedding = torch.zeros(1,max_text_seq_length + self.num_patches,inner_dim,requires_grad=False) #[1,226+17550,]
        pos_embedding.data[:,max_text_seq_length:].copy_(spatial_pos_embedding)  #先创建0矩阵，再拷贝
        self.register_buffer("pos_embedding",pos_embedding,persistent=False) #

        #3. Time embeddings
        self.time_proj = Timesteps(inner_dim,flip_sin_to_cos,freq_shift)
        self.time_embedding = TimestepEmbedding(inner_dim,time_embed_dim,timestep_activation_fn)

        # 4. Define spatio-temporal transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                CogVideoXBlock(
                    dim = inner_dim,
                    num_attention_heads = num_attention_heads,
                    attention_head_dim = attention_head_dim,
                    time_embed_dim = time_embed_dim,
                    dropout = dropout,
                    activation_fn = activation_fn,
                    attention_bias = attention_bias,
                    norm_elementwise_affine = norm_elementwise_affine,
                    norm_eps = norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_final = nn.LayerNorm(inner_dim,norm_eps,norm_elementwise_affine)

        # 5. Output blocks
        self.norm_out = AdaLayerNorm( 
            embedding_dim = time_embed_dim,
            output_dim = 2 * inner_dim,
            norm_elementwise_affine = norm_elementwise_affine,
            norm_eps = norm_eps,
            chunk_dim = 1,
        )
        self.proj_out = nn.Linear(inner_dim,patch_size * patch_size * out_channels) #重新展开回去

        self.gradient_checkpointing = False
        self.parallel_manager = None

    def enable_parallel(self,dp_size,sp_size,enable_cp):
        #update cfg parallel,sp能被2整除，就分一半到cfg para上
        if enable_cp and sp_size % 2 == 0:
            sp_size = sp_size // 2
            cp_size = 2
        else:
            cp_size = 1
        
        self.parallel_manager: ParallelManager = ParallelManager(dp_size,cp_size,sp_size)