import yaml
import torch
from src.training.training import run_multi_gpu_training, run_single_gpu_training


"""
Run training
"""


def main():
    # Open config and get the gpu type
    with open("config/config.yaml", "r") as file:
        config_main = yaml.safe_load(file)

    multi_gpu = config_main["multi_gpu"]

    # Select single or multi GPU
    if multi_gpu:
        # World size is number of GPUs
        world_size = torch.cuda.device_count()

        torch.multiprocessing.spawn(
            run_multi_gpu_training,
            args=(world_size,),
            nprocs=world_size,
            join=True,
        )

    else:
        run_single_gpu_training()


if __name__ == "__main__":
    main()
