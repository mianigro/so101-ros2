import os
import glob
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import gymnasium as gym
import yaml
from mamba_ssm import Mamba
from models import MambaCVModel


class MambaAnalyzer:
    def __init__(self, model_path, analysis_dir):
        self.model_path = model_path
        self.analysis_dir = analysis_dir
        self.activation_maps = {}
        self.state_maps = {}
        self.hooks = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with open(f"{run_dir}/config.yaml", "r") as file:
            config_main = yaml.safe_load(file)

        self.architecture = config_main["architecture"]
        final_layer_pooling = config_main["final_layer_pooling"]
        log_clamp_lower = None
        log_clamp_upper = None

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
        input_shape = (height, height)

        if discrete:
            n_actions = env.action_space.n
        else:
            n_actions = env.action_space.shape[0]

        self.model = MambaCVModel(
            lookback_frames=lookback_frames,
            input_shape=input_shape,
            d_model=d_model,
            num_layers=num_layers,
            d_ff=d_ff,
            num_actions=n_actions,
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

        self._register_hooks()

    def _register_hooks(self):
        def hook_fn(name):
            def _hook(module, input, output):
                # Handle Mamba's tuple output (output, state)
                if isinstance(module, Mamba):
                    # Ensure we capture the full output tensor
                    output_tensor = output[
                        0
                    ]  # Shape: (batch_size, sequence_length, d_model)
                    print(f"Capturing {name} output with shape: {output_tensor.shape}")
                    self.activation_maps[name] = (
                        output_tensor.detach()
                    )  # Full output tensor
                    self.state_maps[name] = output[1].detach()  # Hidden state
                else:
                    print(f"Capturing {name} output with shape: {output.shape}")
                    self.activation_maps[name] = output.detach()

            return _hook

        # Register hooks for CNN encoder
        self.hooks.append(
            self.model.cnn_encoder.register_forward_hook(hook_fn("cnn_encoder"))
        )

        # Register hooks for each Mamba layer
        for idx, mamba_block in enumerate(self.model.mamba):
            # Access the Mamba layer directly from the Sequential block
            mamba_layer = mamba_block[0]
            self.hooks.append(
                mamba_layer.register_forward_hook(hook_fn(f"mamba_layer_{idx}"))
            )

    def visualize_cnn_features(self, batch_idx=0):
        """Visualize 1D CNN features using a bar plot"""
        features = self.activation_maps["cnn_encoder"][batch_idx].cpu().numpy()

        plt.figure(figsize=(15, 6))
        plt.bar(range(len(features)), features)
        plt.title("CNN Feature Vector Activations")
        plt.xlabel("Feature Channel")
        plt.ylabel("Activation Magnitude")
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{self.analysis_dir}/test_cnn.png")
        return plt.gcf()

    def visualize_mamba_states(self, layer_idx=0, batch_idx=0):
        """Visualize Mamba state dynamics"""
        state = self.state_maps[f"mamba_layer_{layer_idx}"][batch_idx].cpu().numpy()

        plt.figure(figsize=(10, 6))
        plt.plot(state.T)
        plt.title(f"Mamba State Dynamics - Layer {layer_idx}")
        plt.xlabel("Time Step")
        plt.ylabel("State Value")
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{self.analysis_dir}/mamba_state_layer_{layer_idx}.png")
        return plt.gcf()

    def analyze_mamba_outputs(self, batch_idx=0):
        """Analyze Mamba layer outputs"""
        outputs = []

        for idx in range(self.model.num_layers):
            layer_name = f"mamba_layer_{idx}"
            if layer_name not in self.activation_maps:
                print(f"Skipping {layer_name} - no activations captured")
                continue

            # Get the full output tensor for the specified batch
            layer_output = self.activation_maps[layer_name][batch_idx].cpu().numpy()

            # Debug: Print output shape and statistics
            print(f"Layer {idx} output shape: {layer_output.shape}")
            print(f"Min: {layer_output.min():.4f}, Max: {layer_output.max():.4f}")
            print(f"Mean: {layer_output.mean():.4f}, Std: {layer_output.std():.4f}")

            # Ensure the output is 2D for heatmap visualization
            if layer_output.ndim == 3:
                # If 3D, use the full sequence (sequence_length, d_model)
                layer_output = layer_output.squeeze(0)  # Remove batch dimension
            elif layer_output.ndim == 1:
                # If 1D, reshape to (sequence_length, d_model)
                layer_output = layer_output.reshape(-1, self.model.d_model)
            elif layer_output.ndim == 2:
                # If 2D, assume it's (sequence_length, d_model)
                pass
            else:
                # If >2D, flatten extra dimensions
                layer_output = layer_output.reshape(layer_output.shape[0], -1)

            outputs.append(layer_output)

        # Visualize outputs
        fig, axes = plt.subplots(1, len(outputs), figsize=(20, 4))
        if len(outputs) == 1:
            axes = [axes]  # Ensure axes is always a list

        for idx, (output, ax) in enumerate(zip(outputs, axes)):
            sns.heatmap(
                output,
                ax=ax,
                cmap="viridis",
                cbar=True,
                xticklabels=False,  # Hide x-axis labels for clarity
                yticklabels=False,  # Hide y-axis labels for clarity
            )
            ax.set_title(f"Layer {idx} Output Activations")
            ax.set_xlabel("Feature Dimension")
            ax.set_ylabel("Time Step")

        plt.tight_layout()
        plt.savefig(f"{self.analysis_dir}/mamba_outputs.png")
        return fig

    def compute_layer_statistics(self):
        """Compute statistics for each layer's activations"""
        stats = {}
        for name, activations in self.activation_maps.items():
            stats[name] = {
                "mean": activations.mean().item(),
                "std": activations.std().item(),
                "max": activations.max().item(),
                "min": activations.min().item(),
                "sparsity": (activations == 0).float().mean().item(),
            }
        return stats

    def cleanup(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()


def analyze_model(model_path, analysis_dir):
    """Main analysis function"""
    analyzer = MambaAnalyzer(model_path=model_path, analysis_dir=analysis_dir)

    # Forward pass with input
    with torch.no_grad():
        batch_size = 4
        input_shape = (6, 210, 210)  # Adjust to your dimensions
        sample_input = torch.randn(batch_size, *input_shape).to(analyzer.device)
        analyzer.model(sample_input)

    # Generate visualizations
    results = {
        "cnn_visualization": analyzer.visualize_cnn_features(),
        "mamba_states": [
            analyzer.visualize_mamba_states(i) for i in range(analyzer.model.num_layers)
        ],
        "mamba_outputs": analyzer.analyze_mamba_outputs(),
        "layer_statistics": analyzer.compute_layer_statistics(),
    }

    print(results)

    analyzer.cleanup()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Mamba layers")
    parser.add_argument("run_name", help="The name of the run")

    args = parser.parse_args()
    run_name = args.run_name

    run_dir = os.path.join("runs", run_name)
    analysis_dir = os.path.join(run_dir, "analyse_result/mamba_plots")
    os.makedirs(analysis_dir, exist_ok=True)

    # Load saved model if exists
    model_files = glob.glob(os.path.join(run_dir, "saved_models", "model_step_*.pth"))
    if model_files:
        model_path = max(model_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))

    analyze_model(model_path, analysis_dir)
