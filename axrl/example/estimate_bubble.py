from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def simulate_task_completion_k_equals_n(
    mu_task: float,
    sigma_task: float,
    num_nodes: int,
    num_simulations: int,
) -> tuple[np.ndarray, np.ndarray]:
    simulated_max_times = []
    simulated_busy_ratios = []

    for _ in range(num_simulations):
        # Each machine processes one task. Generate N task times.
        completion_times = np.maximum(0, np.random.normal(loc=mu_task, scale=sigma_task, size=num_nodes))

        max_completion_time = np.max(completion_times)
        total_busy_time = np.sum(completion_times)  # Sum of N individual task times
        total_time_spent = num_nodes * max_completion_time  # Total time slot for all machines

        # Avoid division by zero if total_time_spent happens to be 0 (unlikely for positive mu)
        busy_ratio = total_busy_time / total_time_spent if total_time_spent > 0 else 0

        simulated_max_times.append(max_completion_time)
        simulated_busy_ratios.append(busy_ratio)

    return np.array(simulated_max_times), np.array(simulated_busy_ratios)


def run_simulations_and_plot(
    mu_task: float,
    sigma_task: float,
    nodes_range: np.ndarray,
    num_simulations_per_point: int,
    save_path: Path,
) -> None:
    sns.set_theme(style="whitegrid")

    results_data = []
    for nodes in nodes_range:
        mt_sims, br_sims = simulate_task_completion_k_equals_n(mu_task, sigma_task, nodes, num_simulations_per_point)
        results_data.append({"# Nodes": nodes, "Max Completion Time": np.mean(mt_sims), "Busy Ratio": np.mean(br_sims)})

    df_results = pd.DataFrame(results_data)

    fig, axes = plt.subplots(2, 1, figsize=(7, 9))
    ax_mt, ax_br = axes

    fig.suptitle(f"Bubble Simualtion, Node Completion Time: Mean({int(mu_task)}s), Std({int(sigma_task)}s)")

    # Plot Max Completion Time
    sns.lineplot(data=df_results, x="# Nodes", y="Max Completion Time", marker="o", ax=ax_mt)
    ax_mt.set_xlabel("# Nodes")
    ax_mt.set_ylabel("Time")
    ax_mt.set_title("Max Completion Time")

    # Plot Bubble Ratio
    df_results["Bubble Ratio"] = 1 - df_results["Busy Ratio"]
    sns.lineplot(data=df_results, x="# Nodes", y="Bubble Ratio", marker="o", ax=ax_br)
    ax_br.set_xlabel("# Nodes")
    ax_br.set_ylabel("Ratio")
    ax_br.set_title("Bubble Ratio")

    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Simulation results plot saved to {save_path}")


if __name__ == "__main__":
    MU_SINGLE_TASK = 550.0
    STD_SINGLE_TASK = 180.0
    NUM_SIMULATIONS_PER_POINT = 8192
    NODES_RANGE = np.arange(1, 32 + 1)  # Range from 1 to 32 inclusive

    run_simulations_and_plot(
        MU_SINGLE_TASK,
        STD_SINGLE_TASK,
        NODES_RANGE,
        NUM_SIMULATIONS_PER_POINT,
        Path("tmp/task_completion_analysis_K_equals_N.png"),
    )
