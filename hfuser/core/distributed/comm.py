from typing import Any, Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup

# ======================================================
# AllGather & ReduceScatter
# ======================================================

def all_to_all_comm(input_, process_group=None, scatter_dim=2, gather_dim=1):


def gather_squence(input_, process_group=None, dim=2, grad_scale=1.0, pad=0):

def split_sequence(input_, process_group=None, dim=2, grad_scale=1.0, pad=0, pad_val=0):

def get_pad(name:str, parallel_group:dist.ProcessGroup):

def set_pad(name:str, dim_size:int, parallel_group:dist.ProcessGroup):