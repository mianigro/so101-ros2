import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import yaml
import torch
import gymnasium as gym
from models import TransformerModel
from utils import stack_frames


def extract_attention_weights(model, input_state):
    """
    Extract attention weights from all layers of the transformer model.

    Args:
        model: TransformerModel instance
        input_state: Input tensor of shape (batch_size, lookback_frames, height, width)

    Returns:
        list of numpy arrays containing attention weights from each layer
    """
    # Ensure model is in eval mode
    model.eval()

    # Process input through model
    with torch.no_grad():
        # Store for attention weights
        all_attention_weights = []

        # Forward pass to get attention weights
        # The MultiHeadAttention modules store weights in last_attention_weights
        action_dist, _ = model(input_state)

        # Extract weights from each encoder layer
        for layer in model.encoder_layers:
            # Get weights from the self-attention module
            # Shape: (batch_size, num_heads, seq_len, seq_len)
            attn_weights = layer.self_attn.last_attention_weights

            # Convert to numpy and store
            all_attention_weights.append(attn_weights.cpu().numpy())

    return all_attention_weights


def plot_attention_heatmap(attention_weights, layer_idx=0, head_idx=0, save_path=None):
    """
    Plot attention heatmap for a specific layer and attention head.

    Args:
        attention_weights: List of attention weight arrays from extract_attention_weights
        layer_idx: Index of the transformer layer to visualize
        head_idx: Index of the attention head to visualize
        save_path: Optional path to save the figure
    """
    # Get weights for specified layer and head
    # Shape: (batch_size, num_heads, seq_len, seq_len)
    layer_weights = attention_weights[layer_idx]

    # Get weights for first batch item and specified head
    # Shape: (seq_len, seq_len)
    head_weights = layer_weights[0, head_idx]

    # Create figure
    plt.figure(figsize=(10, 8))

    # Create heatmap
    sns.heatmap(
        head_weights,
        cmap="Blues",
        xticklabels=range(1, head_weights.shape[1] + 1),
        yticklabels=range(1, head_weights.shape[0] + 1),
        cbar_kws={"label": "Attention Weight"},
    )

    # More descriptive labels
    plt.title(f"Frame Attention Pattern - Layer {layer_idx + 1}, Head {head_idx + 1}")
    plt.xlabel("Frame Being Attended To (Past Frames)")
    plt.ylabel("Frame Doing the Attending")

    # Add timestamp indicators (0 is current, -7 is oldest for 8 frame window)
    num_frames = head_weights.shape[0]
    plt.xticks(range(num_frames), range(-(num_frames - 1), 1)[::-1])
    plt.yticks(range(num_frames), range(-(num_frames - 1), 1)[::-1])

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def visualize_all_attention_heads(save_dir, attention_weights):
    """
    Create heatmaps for all layers and heads in the model.

    Args:
        attention_weights: List of attention weight arrays from extract_attention_weights
        save_dir: Directory to save plots
    """
    num_layers = len(attention_weights)
    num_heads = attention_weights[0].shape[1]

    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            save_path = os.path.join(
                save_dir, f"layer_{layer_idx+1}_head_{head_idx+1}.png"
            )
            plot_attention_heatmap(
                attention_weights,
                layer_idx=layer_idx,
                head_idx=head_idx,
                save_path=save_path,
            )


def analyze_attention_patterns(attention_weights):
    """
    Analyze attention patterns across all layers and heads.

    Args:
        attention_weights: List of attention weight arrays from extract_attention_weights

    Returns:
        dict containing various attention pattern metrics
    """
    metrics = {}

    for layer_idx, layer_weights in enumerate(attention_weights):
        layer_metrics = {
            "avg_attention": layer_weights.mean(axis=(0, 2, 3)),  # Average per head
            "max_attention": layer_weights.max(axis=(0, 2, 3)),  # Max per head
            "sparsity": (layer_weights < 0.1).mean(axis=(0, 2, 3)),  # Sparsity per head
        }

        metrics[f"layer_{layer_idx+1}"] = layer_metrics

    return metrics


def load_analyze_attention(run_dir, model_path):
    save_dir = os.path.join(run_dir, "analyse_result/attention_plots")
    os.makedirs(save_dir, exist_ok=True)

    # Load the state_dict
    state_dict = torch.load(model_path)
    # Remove 'module.' prefix from keys
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # Get model parameters from config
    # Load config
    with open(f"{run_dir}/config.yaml", "r") as file:
        config_main = yaml.safe_load(file)

    if config_main["game"] == "carracing":
        with open(f"{run_dir}/config_{config_main['game']}.yaml", "r") as file:
            config = yaml.safe_load(file)
        env = gym.make(
            "CarRacing-v3",
            render_mode="rgb_array",
            lap_complete_percent=0.95,
            continuous=True,
        )

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

    lookback_frames = config["lookback_frames"]
    height = config["height"]
    final_height = config.get("final_height", height)
    d_model = config["model_config"]["d_model"]
    num_heads = config["model_config"]["num_heads"]
    num_layers = config["model_config"]["num_layers"]
    d_ff = config["model_config"]["d_ff"]
    discrete = config["discrete"]
    final_layer_pooling = config_main["final_layer_pooling"]

    if discrete:
        n_actions = env.action_space.n
        log_clamp_lower = None
        log_clamp_upper = None
    else:
        n_actions = env.action_space.shape[0]
        log_clamp_lower = config["model_config"]["log_clamp_lower"]
        log_clamp_upper = config["model_config"]["log_clamp_upper"]

    # Create model
    model = TransformerModel(
        lookback_frames=lookback_frames,
        input_shape=(final_height, final_height),
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        num_actions=n_actions,
        discrete=discrete,
        final_layer_pooling=final_layer_pooling,
        log_clamp_lower=log_clamp_lower,
        log_clamp_upper=log_clamp_upper,
    )

    model.load_state_dict(new_state_dict)
    model.eval()
    print(f"Loaded model from {model_path}")

    # Create input state
    observation, _ = env.reset()
    stacked_frames = None
    new_episode = True

    # Stack frames
    stacked_frames = stack_frames(
        stacked_frames,
        observation,
        new_episode,
        target_height=height,
        num_stack=lookback_frames,
    )

    # Convert to tensor and add batch dimension
    input_state = torch.FloatTensor(stacked_frames).unsqueeze(0)

    # Extract attention weights
    attention_weights = extract_attention_weights(model, input_state)

    # Plot individual heatmap
    plot_attention_heatmap(attention_weights, layer_idx=0, head_idx=0)

    # Generate plots for all layers and heads
    visualize_all_attention_heads(save_dir, attention_weights)

    # Analyze attention patterns
    metrics = analyze_attention_patterns(attention_weights)

    # Print analysis
    for layer, layer_metrics in metrics.items():
        print(f"\n{layer} Analysis:")
        print(f"Average attention per head: {layer_metrics['avg_attention']}")
        print(f"Maximum attention per head: {layer_metrics['max_attention']}")
        print(f"Attention sparsity per head: {layer_metrics['sparsity']}")


# Example usage:
def main():
    parser = argparse.ArgumentParser(description="Analyse attention layers")
    parser.add_argument("run_name", help="The name of the run")

    args = parser.parse_args()
    run_name = args.run_name

    run_dir = os.path.join("runs", run_name)

    # Load saved model
    model_files = glob.glob(os.path.join(run_dir, "saved_models", "model_step_*.pth"))
    if model_files:
        latest_model = max(
            model_files, key=lambda x: int(x.split("_")[-1].split(".")[0])
        )

    load_analyze_attention(run_dir, model_path=latest_model)


if __name__ == "__main__":
    main()
