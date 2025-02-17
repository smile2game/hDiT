import functools
import math
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange
from timm.models.vision_transformer import Mlp

class CogVideoXPatchEmbed(nn.Module):
    def __init__(self, 
                 patch_size:int=2, 
                 in_channels:int=16, 
                 embed_dim:int=1920, 
                 text_embed_dim:int=4096, 
                 bias:bool=True):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d( #这样patchfy???
            in_channels,embed_dim,kernel_size = (patch_size,patch_size),stride = patch_size,bias = bias
        )
        self.text_proj = nn.Linear(text_embed_dim,embed_dim) 

    def forward(
        self,
        text_embeds:torch.Tensor,
        image_embeds:torch.Tensor,
    ):
        #TODO:
        raise NotImplementedError("CogVideoXPatchEmbed forward 尚未实现")



def apply_rotary_emb(x:torch.Tensor, 
                     freqs_cis:Union[torch.Tensor,Tuple[torch.Tensor]], 
                     use_real:bool=True, 
                     use_real_unbind_dim:int=-1):
    #TODO:
    raise NotImplementedError("apply_rotary_emb 尚未实现")




