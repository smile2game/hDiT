import os
import random

import imageio
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig, ListConfig, OmegaConf


def set_seed(seed, dp_rank=None):
    if seed == -1:
        seed = random.randint(0, 1000000)

    if dp_rank is not None:
        seed = torch.tensor(seed, dtype=torch.int64).cuda()
        if dist.get_world_size() > 1:
            dist.broadcast(seed, 0)
        seed = seed + dp_rank

    seed = int(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def save_video(video, output_path, fps):
    """
    Save a video to disk.
    """
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imageio.mimwrite(output_path, video, fps=fps)