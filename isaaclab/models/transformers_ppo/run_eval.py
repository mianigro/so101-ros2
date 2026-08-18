# Exmaple python3 run_eval.py pong --n_episodes 50

import argparse
import os
import re
import torch
import importlib
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from src.utils import stack_frames
from src.agent import PPOAgent


def load_model(model_path, agent):
    """
    Load a saved model from path

    Args:
        model_path: Path to the saved model file
        agent: Initialized agent

    Returns:
        Agent with loaded model
    """
    # Load model state from saved path
    agent.load_model(model_path)
    return agent


def evaluate_model(
    agent, env, height, lookback_frames, n_episodes=100, reward_func=lambda x: x
):
    """
    Evaluate a model by running episodes and tracking scores

    Args:
        agent: Agent with loaded model
        env: Environment to run evaluation on
        height: Frame height for preprocessing
        lookback_frames: Number of frames to stack
        n_episodes: Number of episodes to evaluate
        reward_func: Function to transform rewards (optional)

    Returns:
        Dictionary with evaluation results
    """
    scores = []
    episode_lengths = []

    for episode in range(n_episodes):
        observation, _ = env.reset()
        stacked_frames = None
        done = False
        truncated = False
        score = 0
        steps = 0

        # Create initial stacked frames buffer
        stacked_frames = stack_frames(
            stacked_frames,
            observation,
            True,
            target_height=height,
            num_stack=lookback_frames,
        )

        while not done and not truncated:
            # Choose action based on loaded model
            action, _, _ = agent.choose_action(stacked_frames)

            # Execute action
            observation_, reward, done, truncated, _ = env.step(action)

            # Track score
            score += reward
            steps += 1

            # Update stacked frames
            stacked_frames = stack_frames(
                stacked_frames,
                observation_,
                False,
                target_height=height,
                num_stack=lookback_frames,
            )

        # Record results
        scores.append(score)
        episode_lengths.append(steps)

        print(f"Episode {episode+1}/{n_episodes}, Score: {score:.2f}, Steps: {steps}")

    return {
        "scores": scores,
        "mean_score": np.mean(scores),
        "std_score": np.std(scores),
        "max_score": np.max(scores),
        "min_score": np.min(scores),
        "episode_lengths": episode_lengths,
        "mean_episode_length": np.mean(episode_lengths),
    }


def find_best_model(models_dir):
    """
    Find the best model file in the given directory based on filename
    We consider "best" to be the model with the highest step count

    Args:
        models_dir: Directory containing model files

    Returns:
        Path to the best model file
    """
    best_model = None
    highest_steps = -1

    # Walk through directories
    for root, _, files in os.walk(models_dir):
        for file in files:
            if file.endswith(".pth") and "model" in file:
                model_path = os.path.join(root, file)

                # Extract step count
                step_match = re.search(r"global_steps_(\d+)", file)
                if step_match:
                    steps = int(step_match.group(1))
                    if steps > highest_steps:
                        highest_steps = steps
                        best_model = model_path

    return best_model


def main():
    parser = argparse.ArgumentParser(description="Evaluate best RL models")
    parser.add_argument("game_name", help="The name of the game")
    parser.add_argument("--run_dir", help="Specific run directory to evaluate")
    parser.add_argument(
        "--models_dir",
        default="saved_models",
        help="Directory containing models within run dir",
    )
    parser.add_argument(
        "--n_episodes",
        type=int,
        default=100,
        help="Number of episodes to evaluate each model",
    )
    parser.add_argument(
        "--output", default="evaluation_results", help="Output directory for results"
    )
    args = parser.parse_args()

    game_name = args.game_name
    n_episodes = args.n_episodes
    base_run_dir = "runs"

    # Either use specified run dir or find matching ones
    run_dirs = []
    if args.run_dir:
        run_dirs = [args.run_dir]
    else:
        for dir_name in os.listdir(base_run_dir):
            if f"training_{game_name}_" in dir_name:
                run_dirs.append(os.path.join(base_run_dir, dir_name))

    if not run_dirs:
        print(f"No runs found for game '{game_name}'")
        return

    print(f"Found {len(run_dirs)} runs for game '{game_name}'")

    # Store results for final comparison
    all_run_results = []

    # Load configuration from each run
    for run_dir in run_dirs:
        print(f"\nEvaluating models in {run_dir}")

        # Load config files
        config_path = os.path.join(run_dir, "config.yaml")
        game_config_path = os.path.join(run_dir, f"config_{game_name}.yaml")

        if not os.path.exists(config_path) or not os.path.exists(game_config_path):
            print(f"Configuration files not found for {run_dir}, skipping...")
            continue

        # Load configuration using yaml
        import yaml

        with open(config_path, "r") as f:
            config_main = yaml.safe_load(f)

        with open(game_config_path, "r") as f:
            config = yaml.safe_load(f)

        # Set up environment and reward function based on game
        try:
            if game_name == "carracing":
                env = gym.make(
                    "CarRacing-v3",
                    render_mode="rgb_array",
                    lap_complete_percent=0.95,
                    continuous=True,
                )
                from src.agent import carracing_reward

                reward_func = carracing_reward
                discrete = False
            elif game_name == "pong" or game_name == "ponglong":
                import ale_py

                env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
                from src.agent import pong_reward

                reward_func = pong_reward
                discrete = True

            elif game_name == "breakout" or game_name == "breakoutlong":
                import ale_py

                env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
                from src.agent import breakout_reward

                reward_func = breakout_reward
                discrete = True

            elif game_name == "kungfu" or game_name == "kungfulong":
                import ale_py

                env = gym.make("ALE/KungFuMaster-v5", render_mode="rgb_array")
                from src.agent import kungfu_reward

                reward_func = kungfu_reward
                discrete = True
            else:
                # Fallback to config-specified environment
                import importlib

                env_module = importlib.import_module(
                    f"gymnasium.envs.{config_main['env_module']}"
                )
                env = gym.make(config_main["env_id"])
                reward_func = lambda x: x  # Default identity function
                discrete = config["discrete"]
        except (ImportError, AttributeError) as e:
            print(f"Error setting up environment for {run_dir}: {e}")
            continue

        # Extract all parameters needed for agent initialization
        architecture = config_main["architecture"]
        final_layer_pooling = config_main["final_layer_pooling"]
        final_pool_skip = config_main["final_pool_skip"]

        # Game frame input information
        lookback_frames = config["lookback_frames"]
        height = config["height"]
        observation_space = (lookback_frames, height, height)

        # Action space
        if discrete:
            n_actions = env.action_space.n
        else:
            n_actions = env.action_space.shape[0]

        # Training hyperparameters
        batch_size = config["training_config"]["batch_size"]
        n_epochs = config["training_config"]["n_epochs"]
        alpha = config["training_config"]["alpha"]
        max_norm = config["training_config"].get("max_norm", 0.5)
        lr_decay = config["training_config"]["lr_decay"]
        lr_decay_step_size = config["training_config"]["lr_decay_step_size"]

        # Model hyperparameters
        d_model = config["model_config"]["d_model"]
        num_heads = config["model_config"].get("num_heads")
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
        log_clamp_lower = config["model_config"].get("log_clamp_lower")
        log_clamp_upper = config["model_config"].get("log_clamp_upper")

        # Find the best model
        models_dir = os.path.join(run_dir, args.models_dir)
        best_model_path = find_best_model(models_dir)

        if not best_model_path:
            print(f"No model files found in {models_dir}")
            continue

        # Create output directory
        os.makedirs(args.output, exist_ok=True)
        run_name = os.path.basename(run_dir)
        eval_results_file = os.path.join(args.output, f"{run_name}_evaluation.txt")

        print(f"\nEvaluating best model: {os.path.basename(best_model_path)}")

        # Initialize the agent with explicit parameters
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

        # Set device for evaluation
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            agent.actor_critic.to(device)

        # Load the model
        agent = load_model(best_model_path, agent)

        # Evaluate the model
        results = evaluate_model(
            agent,
            env,
            height=height,
            lookback_frames=lookback_frames,
            n_episodes=n_episodes,
            reward_func=reward_func,
        )

        # Add run info to results
        results["run_name"] = "_".join(run_name.split("_")[:-1])
        results["model_path"] = best_model_path
        all_run_results.append(results)

        # Write results to file
        with open(eval_results_file, "w") as f:
            f.write(f"Run: {run_name}\n")
            f.write(f"Model: {os.path.basename(best_model_path)}\n")
            f.write(
                f"Mean Score: {results['mean_score']:.2f} ± {results['std_score']:.2f}\n"
            )
            f.write(
                f"Min/Max Score: {results['min_score']:.2f}/{results['max_score']:.2f}\n"
            )
            f.write(f"Mean Episode Length: {results['mean_episode_length']:.2f}\n")

        # Plot score distribution for this run
        plt.figure(figsize=(10, 6))
        plt.hist(results["scores"], bins=20, alpha=0.7)
        plt.axvline(results["mean_score"], color="r", linestyle="dashed", linewidth=2)
        plt.title(f"Score Distribution - {run_name}")
        plt.xlabel("Score")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)
        plt.savefig(
            os.path.join(args.output, f"{run_name}_score_dist.png"),
            dpi=300,
        )
        plt.close()

    # Compare all runs if we have multiple
    if len(all_run_results) > 1:
        # Plot comparison of mean scores across runs
        plt.figure(figsize=(12, 6))

        run_names = [r["run_name"] for r in all_run_results]
        mean_scores = [r["mean_score"] for r in all_run_results]
        std_devs = [r["std_score"] for r in all_run_results]

        # Sort by mean score
        sorted_indices = np.argsort(mean_scores)
        run_names = [run_names[i] for i in sorted_indices]
        mean_scores = [mean_scores[i] for i in sorted_indices]
        std_devs = [std_devs[i] for i in sorted_indices]

        plt.barh(range(len(run_names)), mean_scores, xerr=std_devs, alpha=0.7)
        plt.yticks(range(len(run_names)), run_names)
        plt.title(f"Architecture Evaluation - {game_name}")
        plt.xlabel("Mean Score")
        plt.ylabel("Architecture")
        plt.grid(True, alpha=0.3, axis="x")
        plt.tight_layout()
        plt.savefig(
            os.path.join(args.output, f"{game_name}_run_comparison.png"),
            dpi=300,
        )

        # Write comparison to file
        with open(os.path.join(args.output, f"{game_name}_comparison.txt"), "w") as f:
            f.write(f"Performance Comparison for {game_name}\n")
            f.write("=" * 50 + "\n\n")

            # Sort runs by performance
            sorted_results = sorted(
                all_run_results, key=lambda x: x["mean_score"], reverse=True
            )

            for i, result in enumerate(sorted_results):
                f.write(f"{i+1}. {result['run_name']}\n")
                f.write(
                    f"   Mean Score: {result['mean_score']:.2f} ± {result['std_score']:.2f}\n"
                )
                f.write(
                    f"   Mean Episode Length: {result['mean_episode_length']:.2f}\n\n"
                )

            f.write("\nBest Run: " + sorted_results[0]["run_name"] + "\n")

        print(f"\nRun comparison saved to {args.output} directory")

    env.close()


if __name__ == "__main__":
    main()
