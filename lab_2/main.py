import os
import numpy as np

from src.pso import PSO
from src.fitness_functions import (
    ackley,
    rosenbrock,
    cross_in_tray,
    holder_table,
    mccormick,
    styblinski_tang
)
from src.visualization import (
    plot_function,
    plot_history,
    plot_histories_comparison,
    animate  # only 2d animation used
    # animate_3d  # not used here
)

def run_and_visualize(
    func,
    func_name,
    bounds,
    pso_params_list,
    max_iter=50,
    do_animation=True
):
    # plot the 3d surface
    plot_function(
        func,
        bounds,
        title=f"{func_name} surface",
        save_path=f"results/{func_name.lower()}_surface.png"
    )

    histories = []
    labels = []

    # run pso with each config
    for idx, params in enumerate(pso_params_list):
        pso = PSO(
            func=func,
            swarm_size=params.get("swarm_size", 30),
            w=params.get("w", 0.7),
            c1=params.get("c1", 1.4),
            c2=params.get("c2", 1.4),
            bounds=bounds,
            max_iter=max_iter
        )
        best_pos, best_val, hist_best, pop_hist = pso.run()
        print(f"[{func_name}] run{idx+1} with {params}: best_pos = {best_pos}, best_val = {best_val}")

        histories.append(hist_best)
        label_str = f"run{idx+1}"
        labels.append(label_str)

        # plot the best-value history
        plot_history(
            hist_best,
            title=f"{func_name} {label_str} best",
            save_path=f"results/{func_name.lower()}_run{idx+1}_history.png"
        )

        # optionally create 2d animation
        if do_animation:
            animate(
                func,
                bounds,
                pop_hist,
                title=f"{func_name} animation (run{idx+1})",
                save_path=f"results/{func_name.lower()}_run{idx+1}_anim.gif"
            )

    # plot all runs on one comparison graph
    plot_histories_comparison(
        histories,
        labels,
        title=f"{func_name} comparison",
        save_path=f"results/{func_name.lower()}_compare.png"
    )

def main():
    np.random.seed(42)
    os.makedirs("results", exist_ok=True)

    # define tasks
    tasks = [
        {"func": ackley, "name": "Ackley", "bounds": [(-5, 5), (-5, 5)]},
        {"func": rosenbrock, "name": "Rosenbrock", "bounds": [(-2, 2), (-2, 2)]},
        {"func": cross_in_tray, "name": "cross_in_tray", "bounds": [(-10, 10), (-10, 10)]},
        {"func": holder_table, "name": "HolderTable", "bounds": [(-10, 10), (-10, 10)]},
        {"func": mccormick, "name": "McCormick", "bounds": [(-1.5, 4), (-3, 4)]},
        {"func": styblinski_tang, "name": "StyblinskiTang", "bounds": [(-5, 5), (-5, 5)]}
    ]

    # two sets of parameters
    pso_configs = [
        {"w": 0.9, "c1": 0.8, "c2": 0.8, "swarm_size": 20},
        {"w": 0.4, "c1": 1.8, "c2": 1.8, "swarm_size": 40}
    ]

    # run experiments for each function
    for task in tasks:
        print(f"\n=== {task['name']} ===")
        run_and_visualize(
            func=task["func"],
            func_name=task["name"],
            bounds=task["bounds"],
            pso_params_list=pso_configs,
            max_iter=50,
            do_animation=True
        )

    print("\nall experiments finished. check 'results/' folder.")

if __name__ == "__main__":
    main() 