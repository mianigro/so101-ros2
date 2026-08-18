import os
import time
import yaml
from gymnasium.wrappers import RecordVideo
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from ..utils import stack_frames
from ..utils import SimpleFrameClient


def convert_activations_to_frame(activations):
    """Convert CNN activations to RGB frame [H,W,3] uint8"""
    # Take first sample in batch and first 3 channels
    frame = activations[0][:3]  # [3, H, W]

    # Normalize each channel separately
    frame = frame.transpose(1, 2, 0)  # [H, W, 3]
    frame = (
        (frame - frame.min(axis=(0, 1)))
        / (frame.max(axis=(0, 1)) - frame.min(axis=(0, 1)))
        * 255
    )

    return frame.astype(np.uint8)


class Training:
    def __init__(
        self,
        agent,
        env,
        height,
        n_steps,
        multi_gpu,
        lookback_frames,
        rank,
        world_size,
        save_interval,
        save_video_interval,
        group_id,
        config_main,
        config,
        final_layer_pooling,
        final_pool_skip,
        reward_func,
    ):
        # Training params
        self.agent = agent
        self.reward_func = reward_func
        self.env = env
        self.height = height
        self.n_steps = n_steps
        self.lookback_frames = lookback_frames
        self.save_interval = save_interval
        self.save_video_interval = save_video_interval
        self.group_id = group_id

        # Multi gpu params
        self.multi_gpu = multi_gpu
        self.rank = rank
        self.world_size = world_size
        self.config_main = config_main

        if self.rank == 0:
            self.stream_client = SimpleFrameClient()
            self.time_start = 0.0

        # Score tracking
        self.score_history = []

        # Save dir setup
        if final_layer_pooling:
            self.save_training_dir = os.path.join(
                "runs",
                f'training_{config_main["game"]}_{config_main["architecture"]}_flp_{str(final_layer_pooling).lower()}_fls_{str(final_pool_skip).lower()}_{group_id}',
            )
        else:
            self.save_training_dir = os.path.join(
                "runs",
                f'training_{config_main["game"]}_{config_main["architecture"]}_flp_{str(final_layer_pooling).lower()}_{group_id}',
            )
        self.model_dir = os.path.join(self.save_training_dir, "saved_models")
        self.logs_dir = os.path.join(self.save_training_dir, "logs")
        self.video_dir = os.path.join(self.save_training_dir, "saved_video")

        # TensorBoard setup
        if self.multi_gpu:
            self.log_dir = os.path.join(f"{self.save_training_dir}", f"gpu_{rank}")
        else:
            self.log_dir = self.save_training_dir
        self.writer = SummaryWriter(log_dir=self.log_dir)

        """self.agent.load_model(
            "runs/training_breakout_mamba_flp_false_1750339847/saved_models/model_step_1750339847_learn_iter_150.pth"
        )"""

        # Make dirs needed for logging
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.video_dir, exist_ok=True)

        # Save config files to training log dir
        # Write the main config data to the yaml file
        with open(f"{self.save_training_dir}/config.yaml", "w") as file:
            yaml.dump(config_main, file, default_flow_style=False)

        # Write the game and model data to the yaml file
        with open(
            f'{self.save_training_dir}/config_{config_main["game"]}.yaml', "w"
        ) as file:
            yaml.dump(config, file, default_flow_style=False)

        # Setup multi gpu mode - Single GPU not implemented completely
        if self.multi_gpu:
            # Move models to device
            device = torch.device(
                f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
            )

            # Set the model type for transformer
            self.agent.actor_critic.device = device
            self.agent.actor_critic.to(device)

            if self.rank == 0:
                # Create dummy input (adjust dimensions based on your input shape)
                # Example: (batch_size, channels, height, width)
                self.agent.actor_critic.eval()
                dummy_input = torch.randn(
                    1, self.lookback_frames, self.height, self.height
                ).to(device)
                self.writer.add_graph(self.agent.actor_critic, dummy_input)
                self.agent.actor_critic.train()

            # Wrap in DDP
            self.agent.actor_critic = DDP(self.agent.actor_critic)

    def save_models(self, step_count):
        """
        Save the models locally at a specific point, for
        multi-gpu training, the model on GPU 0 is saved.

        Args:
            step_count: Global step count at the point of saving
        """
        if not self.multi_gpu or self.rank == 0:
            self.agent.save_model(self.model_dir, step_count, self.group_id)

    def log_to_tensorboard(self, log_dict, step):
        """
        Log metrics to TensorBoard.

        Args:
            log_dict: Dictionary containing metrics to log
            step: Current step count for x-axis in TensorBoard
        """
        # Add GPU prefix for multi-gpu setup
        if self.multi_gpu:
            for key, value in log_dict.items():
                self.writer.add_scalar(f"GPU_{self.rank}/{key}", value, step)
        else:
            for key, value in log_dict.items():
                self.writer.add_scalar(key, value, step)

    def training_loop(self):
        """
        Training loop for the model, this is called at N
        intervals to train from memory. At the end, memory
        is cleared.
        """
        # Tracking information
        learn_iters = 0
        avg_score = 0
        step_count = 0
        game = 0
        global_steps = torch.tensor(0)

        if self.rank == 0:
            self.stream_client.connect()

        if self.rank == 0:
            self.env = RecordVideo(
                self.env,
                video_folder=self.video_dir,
                episode_trigger=lambda ep: ep % self.save_video_interval == 0,
            )

        # Start a game
        while True:
            # Get starting point at t=0
            observation, _ = self.env.reset()
            stacked_frames = None
            done = False
            new_episode = True
            truncated = False
            score = 0

            # Create the stacked frames for a new episode as a buffer
            stacked_frames = stack_frames(
                stacked_frames,
                observation,
                new_episode,
                target_height=self.height,
                num_stack=self.lookback_frames,
            )

            # When game playing
            while not done and not truncated:
                # Flag to indicate if new episode or not for stacked_frames()
                new_episode = False

                # Choose an action from the policy
                action, prob, val = self.agent.choose_action(stacked_frames)

                activations = self.agent.actor_critic.module.cnn_encoder.activations

                # Execute action and accumulate rewards and observations
                observation_, reward, done, truncated, _ = self.env.step(action)

                if self.rank == 0:
                    frame = convert_activations_to_frame(activations)

                    self.time_start += 0.067  # Approx time per step

                    time_taken = self.time_start * 2  # Two games at once

                    self.stream_client.send_frame(
                        observation_[:, :, ::-1],
                        frame,
                        time_taken,
                        global_steps,
                        avg_score,
                    )

                # Keep track of score
                score += reward

                # Store memory of state, action, reward, etc
                reward = self.reward_func(reward)
                self.agent.remember(stacked_frames, action, prob, val, reward, done)

                # Add frames to stack
                stacked_frames_ = stack_frames(
                    stacked_frames,
                    observation_,
                    new_episode,
                    target_height=self.height,
                    num_stack=self.lookback_frames,
                )

                # For bootstrappiong GAE value
                if not done:
                    last_val = val
                else:
                    last_val = 0

                # Learning interval
                if (step_count + self.lookback_frames) % self.agent.N == 0:
                    if self.rank == 0:
                        stream_tuple = (
                            observation_[:, :, ::-1],
                            frame,
                            self.stream_client,
                        )
                        (
                            learn_actor_loss,
                            learn_critic_loss,
                            learn_total_loss,
                            entropy,
                            entropy_mod,
                            lr,
                            learn_grad_norms,
                        ) = self.agent.learn(last_val, stream_tuple)

                    else:
                        (
                            learn_actor_loss,
                            learn_critic_loss,
                            learn_total_loss,
                            entropy,
                            entropy_mod,
                            lr,
                            learn_grad_norms,
                        ) = self.agent.learn(last_val)

                    # Multi gpu tracking of global steps
                    if self.multi_gpu:
                        # Global steps
                        global_steps = torch.tensor(
                            [step_count], device=f"cuda:{self.rank}"
                        )
                        dist.all_reduce(global_steps, op=dist.ReduceOp.SUM)
                    # Single GPU tracking and saving
                    else:
                        global_steps = np.array(step_count)

                    # Log to tensorboard
                    log_dict = {
                        "learn_iter": learn_iters,
                        "loss_mean_actor": learn_actor_loss,
                        "loss_mean_critic": learn_critic_loss,
                        "loss_mean_total": learn_total_loss,
                        "entropy": entropy,
                        "entropy_actor_ratio": abs(
                            entropy_mod / learn_actor_loss * 100
                        ),
                        "entropy_critic_ratio": abs(
                            entropy_mod / learn_critic_loss * 100
                        ),
                        "entropy_total_ratio": abs(
                            entropy_mod
                            / (learn_actor_loss + 0.5 * learn_critic_loss)
                            * 100
                        ),
                        "learning_rate": lr,
                        "grad_norms": learn_grad_norms,
                        "gpu_episode": game,
                        "gpu_steps": step_count,
                        "gpu_mean_score": avg_score,
                    }
                    self.log_to_tensorboard(log_dict, global_steps.item())

                    # Text log
                    with open(f"{self.logs_dir}/training_{self.rank}.txt", "a") as f:
                        log_line = f"gpu_no: {self.rank}, global_steps: {global_steps.item()}, gpu_time_steps: {step_count}, learn_iters: {learn_iters}, gpu_mean_score: {avg_score}, mean_actor_loss: {learn_actor_loss}, mean_critic_loss: {learn_critic_loss}, mean_total_loss: {learn_total_loss}, entropy: {entropy}, entropy_actor_ratio: {abs(entropy_mod/learn_actor_loss)}, entropy_critic_ratio: {abs(entropy_mod/learn_critic_loss)}, entropy_total_ratio: {abs(entropy_mod/learn_total_loss)}, learning_rate: {lr}, learn_grad_norms: {learn_grad_norms}\n"
                        f.write(log_line)

                    # Increment learning iter
                    learn_iters += 1

                    # Save at learn iters
                    if learn_iters % self.save_interval == 0:
                        if not self.multi_gpu or self.rank == 0:
                            self.save_models(f"learn_iter_{learn_iters}")

                    # Reset score at the end of each learn iter
                    self.score_history = []

                    # Check to finish training and save
                    if global_steps.item() >= self.n_steps:
                        if self.rank == 0 or not self.multi_gpu:
                            self.save_models(f"global_steps_{global_steps.item()}")

                        if self.multi_gpu:
                            dist.destroy_process_group()

                        return

                # Iterate into next observation
                stacked_frames = stacked_frames_

                # Incremet time step
                step_count += 1

            # Log game information
            self.score_history.append(score)
            avg_score = np.mean(self.score_history).item()

            # Log game in text
            if self.multi_gpu:
                with open(f"{self.logs_dir}/game_{self.rank}.txt", "a") as f:
                    log_line = f"gpu_no: {self.rank}, gpu_episode: {game}, score: {score:.2f}, gpu_mean_score: {avg_score:.2f}, gpu_time_steps: {step_count}, learning_steps: {learn_iters}\n"
                    f.write(log_line)

            else:
                with open(f"{self.logs_dir}/game.txt", "a") as f:
                    log_line = f"gpu_no: {self.rank}, gpu_episode: {game}, score: {score:.2f}, gpu_mean_score: {avg_score:.2f}, gpu_time_steps: {step_count}, learning_steps: {learn_iters}\n"
                    f.write(log_line)

            # Increment game counter for the GPU
            game += 1
