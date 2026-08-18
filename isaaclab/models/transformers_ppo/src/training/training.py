import os
import time
import yaml
import numpy as np
import torch
import torch.distributed as dist
import gymnasium as gym
from ..agent import PPOAgent
from ..agent import (
    kungfu_reward,
    carracing_reward,
    pong_reward,
    breakout_reward,
)
from ..training import Training


def run_multi_gpu_training(
    rank,
    world_size,
):
    """
    This class handles multi-gpu distributed training

    Args:
        rank (int): Rank is the GPU from the world size. This is set
            by the distrubuted training class.
        world_size (int): The world size is the number of
            GPUs.
    """

    # Get time id
    group_id = str(int(time.time()))

    # Load configs for base training and model
    with open("config/config.yaml", "r") as file:
        config_main = yaml.safe_load(file)

    # Distributed env network
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12357"
    os.environ["NCCL_TIMEOUT"] = "1200"

    # Set different seeds for each thread
    torch.manual_seed(int(config_main["seed_base"]) + rank)
    np.random.seed(int(config_main["seed_base"]) + rank)

    # Architect setup
    architecture = config_main["architecture"]
    final_layer_pooling = config_main["final_layer_pooling"]

    if final_layer_pooling:
        final_pool_skip = config_main["final_pool_skip"]
    else:
        final_pool_skip = False

    # Open config
    with open(
        f"config/{config_main['game']}/config_{config_main['game']}_{architecture}_flp_{str(final_layer_pooling).lower()}.yaml",
        "r",
    ) as file:
        config = yaml.safe_load(file)

    # Game setup and rewards
    if config_main["game"] == "carracing":
        env = gym.make(
            "CarRacing-v3",
            render_mode="rgb_array",
            lap_complete_percent=0.95,
            continuous=True,
        )
        reward_func = carracing_reward

    elif config_main["game"] == "pong":
        import ale_py

        env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
        reward_func = pong_reward

    elif config_main["game"] == "breakout":
        import ale_py

        env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
        reward_func = breakout_reward

    elif config_main["game"] == "kungfu":
        import ale_py

        env = gym.make(
            "ALE/KungFuMaster-v5",
            render_mode="rgb_array",
        )
        reward_func = kungfu_reward

    # Game frame input information
    lookback_frames = config["lookback_frames"]
    height = config["height"]
    observation_space = (lookback_frames, height, height)

    # Action space
    discrete = config["discrete"]
    if discrete:
        n_actions = env.action_space.n
    else:
        n_actions = env.action_space.shape[0]

    # Training hyperparameters
    batch_size = config["training_config"]["batch_size"]
    n_epochs = config["training_config"]["n_epochs"]
    alpha = config["training_config"]["alpha"]
    max_norm = config["training_config"]["max_norm"]
    lr_decay = config["training_config"]["lr_decay"]
    lr_decay_step_size = config["training_config"]["lr_decay_step_size"]
    log_clamp_lower = config["model_config"].get("log_clamp_lower")
    log_clamp_upper = config["model_config"].get("log_clamp_upper")
    n_steps = config["training_config"]["n_steps"]
    save_interval = config["training_config"]["save_interval"]
    save_video_interval = config["training_config"]["save_video_interval"]

    # Model hyperparameters
    d_model = config["model_config"]["d_model"]
    num_heads = config.get("model_config").get("num_heads")
    num_layers = config["model_config"]["num_layers"]
    d_ff = config["model_config"]["d_ff"]
    dropout = config["model_config"]["dropout"]
    gae_lambda = config["model_config"]["gae_lambda"]
    policy_clip = config["model_config"]["policy_clip"]
    value_clip_range = config["model_config"]["value_clip_range"]
    gamma = config["model_config"]["gamma"]
    entropy_coef = config["model_config"]["entropy_coef"]
    min_entropy_coef = config["model_config"]["min_entropy_coef"]
    entropy_decay = config["model_config"]["entropy_decay"]
    N = config["model_config"]["N"]

    # Load the agent
    agent = PPOAgent(
        lookback_frames=lookback_frames,
        d_model=d_model,
        input_shape=(height, height),
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=dropout,
        num_actions=n_actions,
        alpha=alpha,
        lr_decay=lr_decay,
        lr_decay_step_size=lr_decay_step_size,
        gae_lambda=gae_lambda,
        gamma=gamma,
        entropy_coef=entropy_coef,
        min_entropy_coef=min_entropy_coef,
        entropy_decay=entropy_decay,
        value_clip_range=value_clip_range,
        policy_clip=policy_clip,
        observation_space=observation_space,
        batch_size=batch_size,
        N=N,
        n_epochs=n_epochs,
        discrete=discrete,
        architecture=architecture,
        final_layer_pooling=final_layer_pooling,
        final_pool_skip=final_pool_skip,
        max_norm=max_norm,
        log_clamp_lower=log_clamp_lower,
        log_clamp_upper=log_clamp_upper,
    )

    # Setup cuda and rank
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        backend = "nccl"
    else:
        backend = "gloo"

    # Load dist process
    print(f"Rank {rank}: Initialising process group")
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
    )
    print(f"Rank {rank}: Process group initialised")

    # Load trainer class
    trainer = Training(
        agent=agent,
        env=env,
        n_steps=n_steps,
        multi_gpu=True,
        height=height,
        rank=rank,
        world_size=world_size,
        lookback_frames=lookback_frames,
        save_interval=save_interval,
        save_video_interval=save_video_interval,
        group_id=group_id,
        config_main=config_main,
        config=config,
        final_layer_pooling=final_layer_pooling,
        final_pool_skip=final_pool_skip,
        reward_func=reward_func,
    )

    # Train the model
    trainer.training_loop()

    # Close out dist process
    dist.destroy_process_group()


def run_single_gpu_training(rank=0):
    """
    This class handles running single GPU training

    Args:
        rank (int): Set to zero as the GPU being used is
            GPU 0.
    """
    # Load main configs for training and model
    with open("config/config.yaml", "r") as file:
        config_main = yaml.safe_load(file)

    # Architect setup
    architecture = config_main["architecture"]
    final_layer_pooling = config_main["final_layer_pooling"]

    with open(
        f"config/{config_main['game']}/config_{config_main['game']}_{architecture}_flp_{str(final_layer_pooling).lower()}.yaml",
        "r",
    ) as file:
        config = yaml.safe_load(file)

    # Game setup
    if config_main["game"] == "carracing":
        env = gym.make(
            "CarRacing-v3",
            render_mode="rgb_array",
            lap_complete_percent=0.95,
            continuous=True,
        )

    elif config_main["game"] == "pong":
        import ale_py

        env = gym.make("ALE/Pong-v5", render_mode="rgb_array")

    elif config_main["game"] == "breakout":
        import ale_py

        env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")

    elif config_main["game"] == "kungfu":
        import ale_py

        env = gym.make(
            "ALE/KungFuMaster-v5",
            render_mode="rgb_array",
        )

    # Game frame input information
    lookback_frames = config["lookback_frames"]
    height = config["height"]
    observation_space = (lookback_frames, height, height)

    # Action space
    discrete = config["discrete"]
    if discrete:
        n_actions = env.action_space.n
    else:
        n_actions = env.action_space.shape[0]

    # Training hyperparameters
    batch_size = config["training_config"]["batch_size"]
    n_epochs = config["training_config"]["n_epochs"]
    alpha = config["training_config"]["alpha"]
    max_norm = config.get("max_norm", 0.5)
    lr_decay = config["training_config"]["lr_decay"]
    lr_decay_step_size = config["training_config"]["lr_decay_step_size"]
    n_steps = config["training_config"]["n_steps"]
    log_clamp_lower = config["model_config"].get("log_clamp_lower")
    log_clamp_upper = config["model_config"].get("log_clamp_upper")
    save_interval = config["training_config"]["save_interval"]
    save_video_interval = config["training_config"]["save_video_interval"]
    group_id = str(int(time.time()))

    # Model hyperparameters
    d_model = config["model_config"]["d_model"]
    num_heads = config.get("model_config").get("num_heads")
    num_layers = config["model_config"]["num_layers"]
    d_ff = config["model_config"]["d_ff"]
    dropout = config["model_config"]["dropout"]
    gae_lambda = config["model_config"]["gae_lambda"]
    policy_clip = config["model_config"]["policy_clip"]
    value_clip_range = config["model_config"]["value_clip_range"]
    gamma = config["model_config"]["gamma"]
    entropy_coef = config["model_config"]["entropy_coef"]
    min_entropy_coef = config["model_config"]["min_entropy_coef"]
    entropy_decay = config["model_config"]["entropy_decay"]
    N = config["model_config"]["N"]
    final_layer_pooling = config_main["final_layer_pooling"]

    # Instantiate the agent
    agent = PPOAgent(
        lookback_frames=lookback_frames,
        d_model=d_model,
        input_shape=(height, height),
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=dropout,
        num_actions=n_actions,
        alpha=alpha,
        lr_decay=lr_decay,
        lr_decay_step_size=lr_decay_step_size,
        gae_lambda=gae_lambda,
        gamma=gamma,
        entropy_coef=entropy_coef,
        min_entropy_coef=min_entropy_coef,
        entropy_decay=entropy_decay,
        value_clip_range=value_clip_range,
        policy_clip=policy_clip,
        observation_space=observation_space,
        batch_size=batch_size,
        N=N,
        n_epochs=n_epochs,
        discrete=discrete,
        architecture=architecture,
        final_layer_pooling=final_layer_pooling,
        max_norm=max_norm,
        log_clamp_lower=log_clamp_lower,
        log_clamp_upper=log_clamp_upper,
    )

    # Instantiate trainer class
    trainer = Training(
        agent=agent,
        env=env,
        n_steps=n_steps,
        multi_gpu=True,
        height=height,
        rank=rank,
        world_size=1,
        lookback_frames=lookback_frames,
        save_interval=save_interval,
        save_video_interval=save_video_interval,
        group_id=group_id,
        config_main=config_main,
        config=config,
    )
    # Train the model
    trainer.training_loop()
