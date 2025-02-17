from typing import Optional,Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CogVideoXLayerNormZero(nn.Module):
    def __init__(
            self,
            conditioning_dim: int, # 512
            embedding_dim: int, #1920 = 64*30
            elementwise_affine: bool = True,
            eps: float = 1e-5,
            bias: bool = True,
    ):
        super().__init__()

        self.silu = nn.SiLU() # SiLU(x) = x&logist_sigmoid(x)
        self.linear = nn.Linear(conditioning_dim,6*embedding_dim,bias = bias)
        self.norm = nn.LayerNorm(embedding_dim,eps=eps,elementwise_affine=elementwise_affine)

    def forward(
        self,
        hidden_states:torch.Tensor,
        encoder_hidden_states:torch.Tensor,
        temb:torch.Tensor,
    ):
        #TODO:
        raise NotImplementedError("CogVideoXLayerNormZero forward 尚未实现")


class AdaLayerNorm(nn.Module):
    def __init__(
                self,
                embedding_dim:int, #和LayerNorm不一样
                num_embeddings:Optional[int]=None,
                output_dim:Optional[int]=None,
                norm_elementwise_affine:bool=False,
                norm_eps:float=1e-5,
                chunk_dim:int=0,
                ):
        super().__init__()

        self.chunk_dim = chunk_dim
        output_dim = output_dim or embedding_dim * 2

        if num_embeddings is not None:
            self.emb = nn.Embedding(num_embeddings,embedding_dim)
        else:
            self.emb = None
        
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim,output_dim)
        self.norm = nn.LayerNorm(output_dim // 2,norm_eps,norm_elementwise_affine)

    def forward(
        self,
        x:torch.Tensor,
        timestep:Optional[torch.Tensor]=None,
        temb:Optional[torch.Tensor]=None,
    ):
        #TODO:
        raise NotImplementedError("AdaLayerNorm forward 尚未实现")
        
