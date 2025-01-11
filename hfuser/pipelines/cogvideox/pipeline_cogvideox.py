# --------------------------------------------------------
# References:
# CogVideo: https://github.com/THUDM/CogVideo
# diffusers: https://github.com/huggingface/diffusers
#  --------------------------------------------------------


class CogVideoConfig: #仅研究并行功能，只需传入model_path和num_gpus
    def __init__(
            self, 
            model_path: str = "THUDM/CogVideoX-2b",  
            num_gpus: int = 1, 
            ):
        self.model_path = model_path
        self.num_gpus = num_gpus
        self.pipeline_cls = CogVideoXPipeline

class CogVideoXPipeline(VideoSysPipeline):
        def __init__(self, config: CogVideoConfig):
