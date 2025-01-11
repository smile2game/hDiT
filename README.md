# H-DiT

## Report

12.19 attention & mlp 支持，torchrun 双卡推理

## 理念

1. 这不是一个用于商业推理的系统，而是一个用于科研实验的 playground,因此我会尽可能精简代码(删除掉多 workers，降低鲁棒性)
2. 致敬 vllm 并学习
3. 尽量少的错误检测,如果不是必要的
4. 学习使用 logger

## 🪕Note

本项目仅用于课题组内部个人科研，无商用用途,专注于科研实验，研究扩散模型在特定模型(组内项目以及通用文生图、文生视频模型)。

项目架构借鉴了[XDiT](https://github.com/xdit-project/xDiT)和[VideoSys](https://github.com/NUS-HPC-AI-Lab/VideoSys)。特别感谢方佳瑞和尤洋两位老师团队的开源共享，给我提供了能够借鉴了框架和实现方案，我在此基础上搭建完成我个人的毕设实验平台。

