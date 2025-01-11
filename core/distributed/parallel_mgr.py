import os
import logging
import torch
import torch.distributed as dist

from colossalai.cluster.process_group_mesh import ProcessGroupMesh 
from torch.distributed import ProcessGroup

from hDiT.hfuser.utils.logging import init_logger

class ParallelManager(ProcessGroupMesh):
    def __init__(self, dp_size, cp_size, sp_size):
        #TODO time is 205.1.10 16:37
        pass


def initialize(rank=0, world_size=1, init_method=None):
    if not dist.is_initialized():
        #先删除原来的
        try:
            dist.destroy_process_group()
        except Exception:
            pass
        #初始阿进程组,帮助每个rank找到自己的编号
        dist.init_process_group(backend="nccl", init_method=init_method, world_size=world_size, rank=rank)
        torch.cuda.set_device(rank)
        init_logger()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        

