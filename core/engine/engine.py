import os
import torch
import torch.distributed as dist

from .mp_utils import get_distributed_init_method, get_open_port

class hfuserEngine:
    """
    worker 
    """
    def __init__(self, config): #利用config初始化
        self.config = config
        self.parallel_worker_tasks = None
        self._init_worker(config.pipeline_cls) #先进行初始化

    def _init_worker(self, pipeline_cls):
        world_size = self.config.num_gpus #获取world_size方便后续使用
        #错误检测与环境变量
        assert world_size <= torch.cuda.device_count() #确保gpu数量小于等于设备数量
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1" #torchinductor 编译线程数
        if "OMP_NUM_THREADS" not in os.environ:
            os.environ["OMP_NUM_THREADS"] = "1" #设置OMP_NUM_THREADS为1，避免CPU争用

        #多节点的地址初始化
        distributed_init_method = get_distributed_init_method("127.0.0.1", get_open_port()) 

        #列表存放workers
        if world_size == 1:
            self.workers = []
            self.worker_monitor = None
        else:
            #TODO: 
            print("to do for multi process")
        print(f"create a driver_worker with pipeline_cls:{pipeline_cls}")

        self.driver_worker = self._create_pipeline( #driver_worker就是pipeline
        pipeline_cls=pipeline_cls, distributed_init_method=distributed_init_method
            )

    def _create_pipeline(self, pipeline_cls, distributed_init_method=None):
        
