import os
import torch
import argparse
import numpy as np
import cv2
import glob
import yaml
from pathlib import Path
import gymnasium as gym
from typing import Dict, Tuple, List
from torch.distributions import Normal, Categorical
from utils import stack_frames
from models import TransformerModel
from models import MambaCVModel


class CNNFeatureExtractor(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.activation = {}

        # Register hooks for each CNN layer
        self.model.cnn_encoder.conv1.register_forward_hook(
            self.get_activation("initial")
        )
        self.model.cnn_encoder.layer1[-1].conv2.register_forward_hook(
            self.get_activation("layer1")
        )
        self.model.cnn_encoder.layer2[-1].conv2.register_forward_hook(
            self.get_activation("layer2")
        )

    def get_activation(self, name):
        def hook(model, input, output):
            # Reshape output to separate lookback frames
            batch_size = output.size(0) // self.model.lookback_frames
            channels = output.size(1)
            height = output.size(2)
            width = output.size(3)
            # Reshape to [lookback_frames, channels, height, width]
            reshaped = output.view(
                batch_size, self.model.lookback_frames, channels, height, width
            )[0]
            self.activation[name] = reshaped.detach()

        return hook

    def forward(self, x):
        return self.model(x)


class MultiFrameFeatureWriter:
    def __init__(
        self,
        output_path: str,
        fps: int,
        frame_size: Tuple[int, int],
        lookback_frames: int,
    ):
        self.frame_size = frame_size
        self.lookback_frames = lookback_frames
        os.makedirs(output_path, exist_ok=True)

        # Calculate grid layout (2 rows x 3 columns for 6 frames)
        self.rows = 2
        self.cols = 3
        self.combined_size = (frame_size[0] * self.cols, frame_size[1] * self.rows)

        # Initialize writers for each feature layer
        codecs = [("mp4v", ".mp4"), ("MJPG", ".avi"), ("XVID", ".avi")]
        layer_names = ["initial", "layer1", "layer2"]

        self.writers = {}
        for codec, extension in codecs:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            test_writers = {
                layer: cv2.VideoWriter(
                    f"{output_path}/{layer}_activation{extension}",
                    fourcc,
                    fps,
                    self.combined_size,
                    isColor=True,
                )
                for layer in layer_names
            }

            if all(writer.isOpened() for writer in test_writers.values()):
                self.writers = test_writers
                print(f"Using codec: {codec}")
                break
            else:
                for writer in test_writers.values():
                    writer.release()

        if not self.writers:
            raise RuntimeError("Failed to initialize video writers")

    def create_feature_overlay(
        self, frame: np.ndarray, feature_map: np.ndarray
    ) -> np.ndarray:
        """Create feature map overlay for a single frame."""
        # Handle NaN values and normalize
        feature_map = np.nan_to_num(feature_map)
        if feature_map.max() != feature_map.min():  # Avoid division by zero
            feature_map = (feature_map - feature_map.min()) / (
                feature_map.max() - feature_map.min()
            )

        # Apply non-linear transformation for better visibility
        feature_map = np.power(feature_map, 0.5)  # Reduced power for better contrast

        # Convert to heatmap
        feature_map_uint8 = (feature_map * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(feature_map_uint8, cv2.COLORMAP_VIRIDIS)
        heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))

        # Blend with original frame
        darkened_frame = cv2.addWeighted(frame, 0.7, np.zeros_like(frame), 0.3, 0)
        return cv2.addWeighted(darkened_frame, 0.5, heatmap, 0.5, 0)

    def create_grid_frame(
        self,
        frames: List[np.ndarray],
        feature_maps: Dict[str, List[np.ndarray]],
        layer_name: str,
    ) -> np.ndarray:
        """Create grid of frames with feature maps for a specific layer."""
        grid = np.zeros(
            (self.combined_size[1], self.combined_size[0], 3), dtype=np.uint8
        )

        for idx, (frame, feature_map) in enumerate(
            zip(frames, feature_maps[layer_name])
        ):
            if idx >= self.lookback_frames:
                break

            # Calculate position in 2x3 grid
            row = idx // self.cols
            col = idx % self.cols

            # Ensure frame is in correct format
            if frame.shape[:2] != self.frame_size:
                frame = cv2.resize(frame, self.frame_size)
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            # Create overlay with feature map
            frame_with_features = self.create_feature_overlay(frame, feature_map)

            # Place frame in grid
            y_start = row * self.frame_size[1]
            y_end = (row + 1) * self.frame_size[1]
            x_start = col * self.frame_size[0]
            x_end = (col + 1) * self.frame_size[0]

            grid[y_start:y_end, x_start:x_end] = frame_with_features

            # Add frame number
            cv2.putText(
                grid,
                f"Frame {idx}",
                (x_start + 10, y_start + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

        return grid

    def write_frame(
        self,
        game_frames: List[np.ndarray],
        feature_maps_dict: Dict[str, List[np.ndarray]],
    ):
        """Write a frame for each feature layer with all lookback frames in a grid."""
        for layer_name in self.writers.keys():
            try:
                grid_frame = self.create_grid_frame(
                    game_frames, feature_maps_dict, layer_name
                )
                self.writers[layer_name].write(grid_frame)
            except Exception as e:
                print(f"Error writing {layer_name}: {str(e)}")

    def release(self):
        for writer in self.writers.values():
            if writer is not None:
                writer.release()


class InferenceAgent:
    def __init__(
        self,
        model_path: str,
        discrete: bool,
        input_shape: Tuple[int, int],
        lookback_frames: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        num_actions: int,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lookback_frames = lookback_frames
        self.discrete = discrete

        # Load config
        with open(f"{run_dir}/config.yaml", "r") as file:
            config_main = yaml.safe_load(file)

        self.architecture = config_main["architecture"]

        final_layer_pooling = config_main["final_layer_pooling"]

        log_clamp_lower = None
        log_clamp_upper = None

        if config_main["architecture"] == "transformer":
            # Create and load model
            self.model = TransformerModel(
                lookback_frames=lookback_frames,
                input_shape=input_shape,
                d_model=d_model,
                num_heads=num_heads,
                num_layers=num_layers,
                d_ff=d_ff,
                num_actions=num_actions,
                discrete=discrete,
                final_layer_pooling=final_layer_pooling,
                log_clamp_lower=log_clamp_lower,
                log_clamp_upper=log_clamp_upper,
            ).to(self.device)

        elif config_main["architecture"] == "mamba":
            self.model = MambaCVModel(
                lookback_frames=lookback_frames,
                input_shape=input_shape,
                d_model=d_model,
                num_layers=num_layers,
                d_ff=d_ff,
                num_actions=num_actions,
                discrete=discrete,
                final_layer_pooling=final_layer_pooling,
                log_clamp_lower=log_clamp_lower,
                log_clamp_upper=log_clamp_upper,
            ).to(self.device)

        # Load state dict
        print(f"Loading model from {model_path}")
        state_dict = torch.load(model_path)
        # Remove 'module.' prefix from keys
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # Load the corrected state_dict
        self.model.load_state_dict(new_state_dict)
        print(f"Loaded model from {model_path}")

        self.feature_extractor = CNNFeatureExtractor(self.model)
        self.feature_extractor.eval()

    def get_action_and_features(
        self, observation: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, List[np.ndarray]]]:
        with torch.no_grad():
            # First reshape observation to [batch=1, lookback, height, width]
            frames = torch.FloatTensor(observation).unsqueeze(0).to(self.device)

            # Process frames through feature extractor
            dist_params, value = self.feature_extractor(frames)

            if self.discrete:
                dist = Categorical(logits=dist_params)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                action = np.array(action.item(), dtype=np.int32)

            else:
                mean, log_std = dist_params.chunk(2, dim=-1)
                dist = Normal(mean, torch.exp(log_std))
                action = dist.sample()
                log_prob = dist.log_prob(action)

                if log_prob.dim() > 1:
                    log_prob = log_prob.sum(dim=-1)

                action = action.cpu().numpy().flatten()
                log_prob = log_prob.item()

            # Extract feature maps for each frame
            feature_maps = {}
            for name, activation in self.feature_extractor.activation.items():
                n_frames = observation.shape[0]  # Number of lookback frames
                # For each CNN layer activation, extract per-frame features
                per_frame_activations = []
                for i in range(n_frames):
                    # Take the mean across channels for visualization
                    feature_map = activation[i].mean(dim=0).cpu().numpy()
                    per_frame_activations.append(feature_map)
                feature_maps[name] = per_frame_activations

            return action, feature_maps


def extract_frames_from_stacked(
    stacked_frames: np.ndarray, lookback_frames: int
) -> List[np.ndarray]:
    """Extract individual frames from stacked frames tensor."""
    frames = []
    for i in range(lookback_frames):
        frame = stacked_frames[i]
        # Convert to uint8 and ensure proper format for visualization
        frame = (frame * 255).astype(np.uint8)
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        frames.append(frame)
    return frames


def run_inference_with_recording(
    model_path: str, run_dir: str, num_episodes: int = 1, fps: int = 30
):

    output_path = os.path.join(run_dir, "analyse_result/cnn_plots")
    os.makedirs(output_path, exist_ok=True)

    with open(f"{run_dir}/config.yaml", "r") as file:
        config_main = yaml.safe_load(file)

    if config_main["game"] == "carracing":
        with open(f"{run_dir}/config_{config_main['game']}.yaml", "r") as file:
            config = yaml.safe_load(file)

    elif config_main["game"] == "pong":
        with open(f"{run_dir}/config_{config_main['game']}.yaml", "r") as file:
            config = yaml.safe_load(file)
        import ale_py

        env = gym.make("ALE/Pong-v5", render_mode="rgb_array")

    elif config_main["game"] == "breakout":
        with open(f"{run_dir}/config_{config_main['game']}.yaml", "r") as file:
            config = yaml.safe_load(file)

        import ale_py

        env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")

    # Extract parameters
    lookback_frames = config["lookback_frames"]
    height = config["height"]
    discrete = config["discrete"]
    d_model = config["model_config"]["d_model"]
    num_heads = config.get("model_config").get("num_heads")
    num_layers = config["model_config"]["num_layers"]
    d_ff = config["model_config"]["d_ff"]

    if discrete:
        n_actions = env.action_space.n
    else:
        n_actions = env.action_space.shape[0]

    # Initialize agent and video writer
    agent = InferenceAgent(
        discrete=discrete,
        model_path=model_path,
        input_shape=(height, height),
        lookback_frames=lookback_frames,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        num_actions=n_actions,
    )

    video_writer = MultiFrameFeatureWriter(
        output_path, fps, (height, height), lookback_frames
    )

    try:
        for episode in range(num_episodes):
            observation, _ = env.reset()
            stacked_frames = None
            done = False
            truncated = False
            total_reward = 0

            # Initial frame stacking
            stacked_frames = stack_frames(
                stacked_frames,
                observation,
                target_height=height,
                is_new_episode=True,
                num_stack=agent.lookback_frames,
            )

            while not done and not truncated:
                # Extract individual frames from stacked frames
                frame_list = extract_frames_from_stacked(
                    stacked_frames, agent.lookback_frames
                )

                # Get action and feature maps
                action, feature_maps = agent.get_action_and_features(stacked_frames)

                # Record frames with feature maps
                video_writer.write_frame(frame_list, feature_maps)

                # Step environment
                observation, reward, done, truncated, _ = env.step(action)

                # Update frames
                stacked_frames = stack_frames(
                    stacked_frames,
                    observation,
                    target_height=height,
                    is_new_episode=False,
                    num_stack=agent.lookback_frames,
                )

                total_reward += reward

            print(f"Episode {episode + 1} finished with reward: {total_reward}")

    finally:
        env.close()
        video_writer.release()
        print(f"Videos saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse attention laters")
    parser.add_argument("run_name", help="The name of the run")

    args = parser.parse_args()
    run_name = args.run_name

    run_dir = os.path.join("runs", run_name)

    # Load saved model if exists
    model_files = glob.glob(os.path.join(run_dir, "saved_models", "model_step_*.pth"))
    if model_files:
        model_path = max(model_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))

    run_inference_with_recording(
        model_path=model_path, run_dir=run_dir, num_episodes=1, fps=30
    )
