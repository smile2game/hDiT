from hfuser import CogVideoXConfig,hfuserEngine #1.config 2.engine
import argparse
import time

model_path = "/root/shared-nvme/CogVideoX-2b"
def run_base():
    config = CogVideoXConfig(model_path=model_path, num_gpus=1) #1.args: model_path, num_gpus
    engine = hfuserEngine(config) #2.engine args: config

    prompt = "Sunset over the sea."
    seed = 10
    start=  time.time()  
    #生成视频，注意guidance_scale
    video = engine.generate(
                            prompt, 
                            guidance_scale=6,
                            num_inference_steps=50, 
                            num_frames=49, 
                            seed=seed
                            ).video[0]
    end = time.time()
    print(f"CogVideoX-2b run_base generate video cost time:{end-start}")
    engine.save_video(video, f"./outputs/{prompt}-{end-start}s.mp4")#2.save_video args: video, save_path

if __name__ == "__main__":
    run_base()