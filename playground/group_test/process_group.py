import torch
import torch.distributed as dist

# 初始化进程组
def init_process_group(backend='nccl', rank=0, world_size=1):
    dist.init_process_group(
        backend=backend,
        init_method='env://',  # 使用环境变量初始化
        rank=rank,
        world_size=world_size
    )

# 示例：将所有进程的梯度进行聚合（AllReduce）
def allreduce_example(tensor):
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()  # 计算平均值

# 销毁进程组
def destroy_process_group():
    dist.destroy_process_group()

# 例子: 在每个进程中进行某些操作
def main():
    rank = 0  # 假设当前进程的rank是0
    world_size = 1  # 假设总共有1个进程

    init_process_group(rank=rank, world_size=world_size)

    # 假设每个进程有一个张量
    tensor = torch.randn(10).cuda()

    # 进行AllReduce操作
    allreduce_example(tensor)

    # 输出操作结果
    print(f"Rank {rank}, tensor after all-reduce: {tensor}")

    destroy_process_group()

if __name__ == "__main__":
    main()
