import os
from functools import partial
from typing import Any, Optional

import torch
import torch.distributed as dist

from .mp_utils import get_distributed_init_method, get_open_port
import hfuser

class hfuserEngine:
    """
    worker 
    """
    def __init__(self, config): 
        self.config = config #获取config (model_path,pipeline_cls,num_gpus) 
        self.parallel_worker_tasks = None
        self._init_worker(config.pipeline_cls) 

    def _init_worker(self, pipeline_cls):
        world_size = self.config.num_gpus 
        #set env_path,avoid torchinductor compile error and cpu contention
        assert world_size <= torch.cuda.device_count() 
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
        if "OMP_NUM_THREADS" not in os.environ:
            os.environ["OMP_NUM_THREADS"] = "1"

        #multi-node init_method
        distributed_init_method = get_distributed_init_method("127.0.0.1", get_open_port()) 

        #列表存放workers
        if world_size == 1:
            self.workers = []
            self.worker_monitor = None
        else:
            #TODO: 
            print("to do for multi process")
        print(f"create a driver_worker with pipeline_cls:{pipeline_cls}")

        #pp_clas实例化成pipeline返回给driver_worker,distri_init_method介入
        self.driver_worker = self._create_pipeline( 
        pipeline_cls=pipeline_cls, distributed_init_method=distributed_init_method
            )

    def _create_pipeline(self, pipeline_cls, distributed_init_method=None):
        hfuser.initialize(rank=0, world_size=self.config.num_gpus, init_method=distributed_init_method)
        pipeline = pipeline_cls(self.config) #调用__init__方法进行实例化
        return pipeline
    
    def _run_workers(
        self,
        method: str,
        *args,
        async_run_tensor_parallel_workers_only: bool = False,
        max_concurrent_workers: Optional[int] = None,
        **kwargs,
    ) -> Any:
        #多进程执行
        worker_outputs = [worker.execute_method(method, *args, **kwargs) for worker in self.workers]
        if async_run_tensor_parallel_workers_only:
            return worker_outputs
        #主进程执行
        driver_worker_method = getattr(self.driver_worker, method) #获取方法
        driver_worker_output = driver_worker_method(*args, **kwargs) #执行方法

        return [driver_worker_output] + [output.get() for output in worker_outputs]
        

    def generate(self, *args, **kwargs):
        return self._run_workers("generate", *args, **kwargs)[0] #返回driver_worker_output
