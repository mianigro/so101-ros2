import argparse
import os
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from scipy.ndimage import gaussian_filter1d


def parse_log_file(file_path):
    """Parse a log file and extract episode, score, and other metrics."""
    results = []
    with open(file_path, "r") as f:
        for line in f:
            match = re.search(
                r"gpu_no: (\d+), gpu_episode: (\d+), score: ([-+]?\d*\.\d+|\d+), gpu_mean_score: ([-+]?\d*\.\d+|\d+), gpu_time_steps: (\d+), learning_steps: (\d+)",
                line,
            )
            if match:
                gpu_no = int(match.group(1))
                episode = int(match.group(2))
                score = float(match.group(3))
                mean_score = float(match.group(4))
                time_steps = int(match.group(5))
                learning_steps = int(match.group(6))

                results.append(
                    {
                        "gpu_no": gpu_no,
                        "episode": episode,
                        "score": score,
                        "mean_score": mean_score,
                        "time_steps": time_steps,
                        "learning_steps": learning_steps,
                    }
                )
    return results


def main():
    parser = argparse.ArgumentParser(description="Analyse RL game results")
    parser.add_argument("game_name", help="The name of game")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument(
        "--output", default="results", help="Output directory for plots"
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=5.0,
        help="Gaussian smoothing sigma (higher = smoother)",
    )
    args = parser.parse_args()

    game_name = args.game_name
    base_run_dir = "runs"
    sigma = args.sigma  # Gaussian smoothing parameter

    # Find all run directories that match the game name
    run_dirs = []
    for dir_name in os.listdir(base_run_dir):
        # Match directories with format: {timestamp}_{game_name}_{args}
        if f"_{game_name}_" in dir_name:
            run_dirs.append(dir_name)

    if not run_dirs:
        print(f"No runs found for game '{game_name}'")
        return

    print(f"Found {len(run_dirs)} runs for game '{game_name}'")

    # Process each run directory
    all_results = {}
    for run_dir in run_dirs:
        # Extract the arguments part from the directory name
        run_args = "_".join(run_dir.split("_")[1:-1])

        # Find all game log files
        log_dir = os.path.join(base_run_dir, run_dir, "logs")
        if not os.path.exists(log_dir):
            print(f"No logs directory found for run: {run_dir}")
            continue

        # Track GPU time steps and scores
        all_gpu_data = []

        for log_file in os.listdir(log_dir):
            if log_file.startswith("game_") and log_file.endswith(".txt"):
                log_path = os.path.join(log_dir, log_file)
                gpu_rank = int(log_file.split("_")[1].split(".")[0])

                # Parse the log file
                results = parse_log_file(log_path)

                # Store data for each GPU
                for result in results:
                    all_gpu_data.append(
                        {
                            "gpu_no": result["gpu_no"],
                            "time_steps": result["time_steps"],
                            "score": result["score"],
                        }
                    )

        # Sort data by time steps
        all_gpu_data.sort(key=lambda x: x["time_steps"])

        # Store in results
        all_results[run_args] = {"gpu_data": all_gpu_data}

    # Generate plots if requested
    if args.plot:
        # Create output directory if it doesn't exist
        os.makedirs(args.output, exist_ok=True)

        # Plot scores vs time steps with standard deviation
        plt.figure(figsize=(14, 8))

        # Define colors for different runs with higher contrast
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

        max_time_steps = 0
        run_stats = {}  # Store statistics for bar chart

        for i, (run_args, results) in enumerate(all_results.items()):
            gpu_data = results["gpu_data"]
            if not gpu_data:
                continue

            # Group by time steps (create bins)
            max_ts = max(item["time_steps"] for item in gpu_data)
            max_time_steps = max(max_time_steps, max_ts)

            # Create bins - we need to bin the data to calculate statistics
            bin_size = max(1, max_ts // 1000)  # Adapt bin size to data scale
            bins = {}

            for item in gpu_data:
                bin_idx = item["time_steps"] // bin_size
                if bin_idx not in bins:
                    bins[bin_idx] = []
                bins[bin_idx].append(item["score"])

            # Calculate mean and std for each bin
            x_steps = []
            y_mean = []
            y_std = []

            for bin_idx in sorted(bins.keys()):
                scores = bins[bin_idx]
                time_step = bin_idx * bin_size
                x_steps.append(time_step)
                y_mean.append(np.mean(scores))
                y_std.append(np.std(scores) if len(scores) > 1 else 0)

            # Store the average standard deviation for the bar chart
            run_stats[run_args] = {"avg_std": np.mean(y_std), "color": colors[i]}

            # Convert to numpy arrays
            x_steps = np.array(x_steps)
            y_mean = np.array(y_mean)
            y_std = np.array(y_std)

            # Apply Gaussian smoothing
            if len(x_steps) > 3:  # Need at least a few points for smoothing
                smooth_mean = gaussian_filter1d(y_mean, sigma=sigma)
                smooth_std = gaussian_filter1d(y_std, sigma=sigma)

                # Plot the smoothed line
                (line,) = plt.plot(
                    x_steps,
                    smooth_mean,
                    label=f"Args: {run_args}",
                    color=colors[i],
                    linewidth=2.5,
                )

                # Add the filled standard deviation area
                plt.fill_between(
                    x_steps,
                    smooth_mean - smooth_std,
                    smooth_mean + smooth_std,
                    color=colors[i],
                    alpha=0.2,
                )

        plt.title(
            f"Game: {game_name} - Scores vs Time Steps",
            fontsize=14,
        )
        plt.xlabel("Time Steps", fontsize=12)
        plt.ylabel("Mean Score", fontsize=12)
        plt.grid(True, alpha=0.3)

        # Only add legend if there are items
        if plt.gca().get_legend_handles_labels()[0]:
            plt.legend(fontsize=10)

        # Improve the figure appearance
        plt.tight_layout()

        # Save the figure with high DPI
        plt.savefig(
            os.path.join(args.output, f"{game_name}_timesteps_comparison.png"), dpi=300
        )
        print(f"\nPlot saved to {args.output} directory")

        # Also create a version without std deviation for clearer trend comparison
        plt.figure(figsize=(14, 8))

        for i, (run_args, results) in enumerate(all_results.items()):
            gpu_data = results["gpu_data"]
            if not gpu_data:
                continue

            # Group by time steps (create bins)
            bin_size = max(1, max_time_steps // 1000)  # Consistent bin size across runs
            bins = {}

            for item in gpu_data:
                bin_idx = item["time_steps"] // bin_size
                if bin_idx not in bins:
                    bins[bin_idx] = []
                bins[bin_idx].append(item["score"])

            # Calculate mean for each bin
            x_steps = []
            y_mean = []

            for bin_idx in sorted(bins.keys()):
                scores = bins[bin_idx]
                time_step = bin_idx * bin_size
                x_steps.append(time_step)
                y_mean.append(np.mean(scores))

            # Convert to numpy arrays
            x_steps = np.array(x_steps)
            y_mean = np.array(y_mean)

            # Apply Gaussian smoothing
            if len(x_steps) > 3:
                smooth_mean = gaussian_filter1d(y_mean, sigma=sigma)

                # Plot only the smoothed line
                plt.plot(
                    x_steps,
                    smooth_mean,
                    label=f"Args: {run_args}",
                    color=colors[i],
                    linewidth=2.5,
                )

        plt.title(f"Game: {game_name} - Scores vs Time Steps", fontsize=14)
        plt.xlabel("Time Steps", fontsize=12)
        plt.ylabel("Mean Score", fontsize=12)
        plt.grid(True, alpha=0.3)

        if plt.gca().get_legend_handles_labels()[0]:
            plt.legend(fontsize=10)

        plt.tight_layout()
        plt.savefig(
            os.path.join(args.output, f"{game_name}_timesteps_lines_only.png"), dpi=300
        )

        # Create a bar chart to compare standard deviations
        if run_stats:
            plt.figure(figsize=(12, 10))

            args_labels = list(run_stats.keys())
            std_values = [stats["avg_std"] for stats in run_stats.values()]
            bar_colors = [stats["color"] for stats in run_stats.values()]

            # Create the bar chart with matching colors
            bars = plt.bar(args_labels, std_values, color=bar_colors, alpha=0.7)

            # Add value labels on top of each bar
            for bar, value in zip(bars, std_values):
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01 * max(std_values),
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

            plt.title(
                f"Game: {game_name} - Average Standard Deviation by Configuration",
                fontsize=14,
            )
            plt.xlabel("Configuration Arguments", fontsize=12)
            plt.ylabel("Average Standard Deviation", fontsize=12)
            plt.grid(True, axis="y", alpha=0.3)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            plt.savefig(
                os.path.join(args.output, f"{game_name}_std_comparison.png"), dpi=300
            )
            print(
                f"Standard deviation comparison plot saved to {args.output} directory"
            )


if __name__ == "__main__":
    main()
