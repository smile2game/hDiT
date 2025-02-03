# --------------------------------------------------------
# References:
# CogVideo: https://github.com/THUDM/CogVideo
# diffusers: https://github.com/huggingface/diffusers
#  --------------------------------------------------------

import inspect
import logging
import math
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from transformers import T5EncoderModel, T5Tokenizer 

#pipeline
from hfuser.core.pipeline.pipeline import hfuserPipeline,hfuserPipelineOutput 
#models
from hfuser.models.autoencoders.autoencoder_kl_cogvideox import AutoencoderKLCogVideoX #vae
from hfuser.models.transformers.cogvideox_transformer_3d import CogVideoXTransformer3DModel #transformer
from hfuser.schedulers.scheduling_ddim_cogvideox import CogVideoXDDIMScheduler #scheduler
from hfuser.utils.utils import save_video, set_seed


class CogVideoXConfig: #仅研究并行功能，只需传入model_path和num_gpus
    """
    config is to instanciate a 'CogVideoXPipeline'
    config >> engine >> pipeline >> generate 
    """
    def __init__(
            self, 
            model_path: str = "THUDM/CogVideoX-2b",  
            num_gpus: int = 1, 
            ):
        # ======= model =======
        self.model_path = model_path  
        self.pipeline_cls = CogVideoXPipeline #这是个类,还没用实例化
        # ======= distributed =======
        self.num_gpus = num_gpus



class CogVideoXPipeline(hfuserPipeline):
        """
        engine._create_pipeline >> pipeline_cls(self.config)
        """
        def __init__(
                self, 
                config: CogVideoXConfig,
                tokenizer: Optional[T5Tokenizer] = None,
                text_encoder: Optional[T5EncoderModel] = None, #embedding文本
                vae: Optional[AutoencoderKLCogVideoX] = None,
                transformer: Optional[CogVideoXTransformer3DModel] = None,
                scheduler: Optional[CogVideoXDDIMScheduler] = None,
                device: torch.device = torch.device("cuda"),
                dtype: torch.dtype = torch.bfloat16,
        ):
                super().__init__() 
                #instance varaibles
                self._config = config
                self._device = device
                small_cogvideo = True
                if small_cogvideo: #bfloat16 >> float16
                    dtype = torch.float16
                self._dtype = dtype
                #load models
                if transformer is None:
                    transformer = CogVideoXTransformer3DModel.from_pretrained(
                        config.model_path, subfolder="transformer", torch_dtype=self._dtype
                    )
                if vae is None:
                    vae = AutoencoderKLCogVideoX.from_pretrained(config.model_path, subfolder="vae", torch_dtype=self._dtype)
                if tokenizer is None:
                    tokenizer = T5Tokenizer.from_pretrained(config.model_path, subfolder = "tokenizer")
                if text_encoder is None:
                    text_encoder = T5EncoderModel.from_pretrained(config.model_path,subfolder = "text_encoder",torch_dtype=self._dtype)
                if scheduler is None:
                     scheduler = CogVideoXDDIMScheduler.from_pretrained(config.model_path,subfolder="scheduler")
                #注册modules
                self.register_modules(tokenizer=tokenizer,text_encoder=text_encoder,vae=vae,transformer=transformer,scheduler=scheduler)
                #不使用cpu_offload以及vae_tilling
                self.set_eval_and_device(self._device,text_encoder,vae,transformer)

                self.vae_scale_factor_spatial = (
                     2 ** (len(self.vae.config.block_out_channels) -1) if hasattr(self, "vae") and self.vae is not None else 8
                )
                self.vae_scale_factor_temporal = (
                    self.vae.config.temporal_compression_ratio if hasattr(self, "vae") and self.vae is not None else 4
                )
                self.video_processor = VideoProcessor(vae_scale_factor = self.vae_scale_factor_spatial)
                #parallel
                self._set_parallel()
            
            def _set_seed(self,seed):
                if dist.get_world_size() == 1:
                     set_seed(seed)
                else:
                     set_seed(seed,self.transformer.parallel_manager.dp_rank)
            
            def _set_parallel(self,dp_size:Optional[int]=None,sp_size:Optional[int]=None,enable_cp:Optional[bool]=False):
                #init sequence parallel