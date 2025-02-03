import logging
from typing import Optional,Tuple,Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoencoderKLCogVideoX(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 3,
            down_block_types: Tuple[str] = (
                "CogVideoXDownBlock3D",
                "CogVideoXDownBlock3D",
                "CogVideoXDownBlock3D",
                "CogVideoXDownBlock3D",
            ),
    )