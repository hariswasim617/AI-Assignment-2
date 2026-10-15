"""
Command-line entry point: runs the experiments and saves all results.

Examples
--------
    python main.py                          # main GA run (pop 100, 200 generations, seed 42)
    python main.py --experiment sensitivity # hyper-parameter sweeps
    python main.py --experiment baseline    # GA vs. random search
    python main.py --experiment all         # everything above
    python main.py --experiment all --quick # whole pipeline in ~2 minutes (for testing)
    python main.py --seed 7 --generations 50 --pop-size 60
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from uav_ga import config as cfg
from uav_ga import visualization as viz
from uav_ga.baseline import random_search
from uav_ga.environment import Environment
from uav_ga.ga import GeneticAlgorithm

# Settings used by --quick (small numbers, only meant to test the pipeline).
QUICK = {"pop_size": 30, "generations": 20, "sweep_generations": 10, "sweep_seeds": [1],
         "baseline_seeds": [1, 2], "sweep_pop_sizes": [20, 50], "sweep_mutation_rates": [0.05, 0.2],
         "sweep_tournament_k": [2, 5], "sweep_selections": ["roulette", "tournament"],
         "sweep_constraint_modes": ["penalty", "both"]}


def save_json(data: dict, path: str) -> None:
    """Write a dictionary to a JSON file (NumPy numbers are converted to plain numbers)."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def snapshot_generations(last: int) -> list[int]:
    """Generation 0, ~25 %, ~50 % and the final generation."""
    return [0, round(0.25 * last), round(0.5 * last), last]


# ---------------------------------------------------------------------------
# Experiment 1: the main run
# ---------------------------------------------------------------------------
def run_main(env: Environment, out: str, seed: int, pop_size: int, generations: int) -> None:
    os.makedirs(out, exist_ok=True)
    print(f"\n=== Main GA run: seed {seed}, population {pop_size}, {generations} generations ===")
    ga = GeneticAlgorithm(env, seed=seed, pop_size=pop_size, generations=generations, keep_populations=True)
    best = ga.run()
    history = pd.DataFrame(ga.history)
    history.to_csv(os.path.join(out, "history.csv"), index=False)

    last = int(history["generation"].iloc[-1])
    gens = snapshot_generations(last)
    viz.plot_environment(env, os.path.join(out, "environment.png"))
    viz.plot_convergence(history, os.path.join(out, "convergence.png"))
    viz.plot_diversity(history, os.path.join(out, "diversity.png"),
                       threshold=cfg.DIVERSITY_THRESHOLD * history["diversity"].iloc[0])
    viz.plot_objectives(history, os.path.join(out, "objectives_over_time.png"))
    viz.plot_routes_over_generations([(g, ga.best_per_generation[g]) for g in gens], env,
                                     os.path.join(out, "paths_generations.png"))
    viz.plot_coverage_heatmaps([(g, ga.best_per_generation[g], ga.populations[g]) for g in gens], env,
                               os.path.join(out, "coverage_heatmaps.png"))
    viz.plot_final_solution(best, env, os.path.join(out, "final_solution.png"),
                            f"Final evolved fleet plan (seed {seed}, generation {last})")

    save_json({
        "seed": seed, "pop_size": pop_size, "max_generations": generations,
        "generations_run": last, "stop_reason": ga.stop_reason,
        "fitness_evaluations": ga.evaluator.n_evaluations, "runtime_s": round(ga.runtime_s, 1),
        "initial_best_fitness": history["best_fitness"].iloc[0],
        "initial_mean_fitness": history["mean_fitness"].iloc[0],
        "final_best_fitness": best.fitness, "final_mean_fitness": history["mean_fitness"].iloc[-1],
        "immigrant_generations": int((history["immigrants"] > 0).sum()),
        "best_objectives": best.objectives, "snapshot_generations": gens,
        "best_plan": best.describe(), "config": cfg.as_dict(),
    }, os.path.join(out, "run_summary.json"))
    print(f"Saved main-run results to {out}/")


# ---------------------------------------------------------------------------
# Experiment 2: hyper-parameter sensitivity
# ---------------------------------------------------------------------------
def generations_to_converge(curve: list[float]) -> int:
    """First generation at which 99 % of the total improvement was reached."""
    curve = np.array(curve)
    goal = curve[0] + 0.99 * (curve[-1] - curve[0])        # start + 99 % of the total gain
    return int(np.argmax(curve >= goal))                   # argmax of True/False = first True


def run_sweep(env: Environment, out: str, name: str, parameter: str, values: list,
              seeds: list[int], generations: int, pop_size: int) -> pd.DataFrame:
    """Run the GA for every value of one parameter (all other settings fixed)."""
    rows, curves = [], {}
    for value in values:
        label = f"{parameter} = {value}"
        curves[label] = []
        for seed in seeds:
            # only ONE parameter changes, everything else keeps its default value;
            # stagnation is switched off so every run lasts the same number of generations
            settings = {"pop_size": pop_size, "mutation_rate": cfg.SWEEP_BASE_MUTATION_RATE,
                        "tournament_k": cfg.SWEEP_BASE_TOURNAMENT_K, parameter: value}
            ga = GeneticAlgorithm(env, seed=seed, generations=generations, verbose=False,
                                  stagnation_patience=10 ** 9, **settings)
            best = ga.run()
            curve = [h["best_fitness"] for h in ga.history]
            curves[label].append(curve)
            rows.append({"sweep": name, "parameter": parameter, "value": value, "seed": seed,
                         "final_best_fitness": best.fitness, "final_mean_fitness": ga.history[-1]["mean_fitness"],
                         "generations_to_converge": generations_to_converge(curve),
                         "final_diversity": ga.history[-1]["diversity"], "runtime_s": ga.runtime_s,
                         "evaluations": ga.evaluator.n_evaluations, "violation": best.violation,
                         "nofly_fraction": best.objectives["nofly_fraction"],
                         "feasible_share": ga.history[-1]["feasible_share"],
                         "coverage": best.objectives["coverage"], "detection": best.objectives["detection"]})
            print(f"  [{name}] {label:<28} seed {seed}: best {best.fitness:.4f} ({ga.runtime_s:.0f} s)")
    viz.plot_sweep(curves, f"Sensitivity: {name}", os.path.join(out, f"sweep_{parameter}.png"))
    return pd.DataFrame(rows)


def run_sensitivity(env: Environment, out: str, quick: bool) -> None:
    os.makedirs(out, exist_ok=True)
    q = QUICK if quick else {}
    seeds = q.get("sweep_seeds", cfg.SWEEP_SEEDS)
    gens = q.get("sweep_generations", cfg.SWEEP_GENERATIONS)
    pop = q.get("pop_size", cfg.POP_SIZE)
    sweeps = [
        ("population size", "pop_size", q.get("sweep_pop_sizes", cfg.SWEEP_POP_SIZES)),
        ("mutation rate", "mutation_rate", q.get("sweep_mutation_rates", cfg.SWEEP_MUTATION_RATES)),
        ("tournament size k (selection pressure)", "tournament_k", q.get("sweep_tournament_k", cfg.SWEEP_TOURNAMENT_K)),
        ("selection method", "selection", q.get("sweep_selections", cfg.SWEEP_SELECTIONS)),
        ("constraint handling", "constraint_mode", q.get("sweep_constraint_modes", cfg.SWEEP_CONSTRAINT_MODES)),
    ]
    print(f"\n=== Sensitivity analysis: seeds {seeds}, {gens} generations per run ===")
    all_runs = []
    for name, parameter, values in sweeps:
        all_runs.append(run_sweep(env, out, name, parameter, values, seeds, gens, pop))
        pd.concat(all_runs).to_csv(os.path.join(out, "sensitivity_runs.csv"), index=False)  # save progress
    runs = pd.concat(all_runs)
    summary = (runs.groupby(["sweep", "parameter", "value"], sort=False)
               .agg(best_mean=("final_best_fitness", "mean"), best_std=("final_best_fitness", "std"),
                    converge_gen_mean=("generations_to_converge", "mean"),
                    converge_gen_std=("generations_to_converge", "std"),
                    runtime_mean_s=("runtime_s", "mean"), evaluations=("evaluations", "mean"),
                    final_diversity=("final_diversity", "mean"), violation_mean=("violation", "mean"),
                    feasible_share=("feasible_share", "mean"))
               .reset_index())
    summary.to_csv(os.path.join(out, "sensitivity_summary.csv"), index=False)
    print(summary.to_string(index=False))
    run_combined_check(env, out, seeds, gens, pop)


def run_combined_check(env: Environment, out: str, seeds: list[int], generations: int, pop_size: int) -> None:
    """Run the best mutation rate and the best tournament k of the sweeps TOGETHER.

    The sweeps change one setting at a time, so we still have to check that the
    two best values also work well in combination before using them as defaults.
    """
    summary = pd.read_csv(os.path.join(out, "sensitivity_summary.csv"))
    best_of = lambda name: summary[summary["parameter"] == name].sort_values("best_mean").iloc[-1]["value"]
    best_rate, best_k = float(best_of("mutation_rate")), int(float(best_of("tournament_k")))
    rows = []
    for seed in seeds:
        ga = GeneticAlgorithm(env, seed=seed, generations=generations, pop_size=pop_size, verbose=False,
                              mutation_rate=best_rate, tournament_k=best_k, stagnation_patience=10 ** 9)
        best = ga.run()
        rows.append({"mutation_rate": best_rate, "tournament_k": best_k, "seed": seed,
                     "final_best_fitness": best.fitness, "runtime_s": ga.runtime_s,
                     "immigrant_generations": sum(h["immigrants"] > 0 for h in ga.history)})
        print(f"  [combined] mutation_rate = {best_rate}, k = {best_k}, seed {seed}: best {best.fitness:.4f}")
    pd.DataFrame(rows).to_csv(os.path.join(out, "combined_check.csv"), index=False)


# ---------------------------------------------------------------------------
# Experiment 3: GA vs. random search
# ---------------------------------------------------------------------------
METRICS = ["fitness", "coverage", "high_priority_coverage", "detection", "targets_detected",
           "energy_kj", "makespan_s", "hazard", "violation"]


def metric_row(method: str, seed: int, chrom, evaluations: int, runtime: float) -> dict:
    o = chrom.objectives
    return {"method": method, "seed": seed, "fitness": chrom.fitness, "coverage": o["coverage"],
            "high_priority_coverage": o["high_priority_coverage"], "detection": o["detection"],
            "targets_detected": o["targets_detected"], "energy_kj": o["energy_kj"],
            "makespan_s": o["makespan_s"], "hazard": o["hazard"], "violation": chrom.violation,
            "evaluations": evaluations, "runtime_s": runtime}


def run_baseline(env: Environment, out: str, seeds: list[int], pop_size: int, generations: int) -> None:
    os.makedirs(out, exist_ok=True)
    print(f"\n=== GA vs. random search: seeds {seeds} ===")
    rows, ga_curves, rs_curves, examples = [], [], [], None
    for seed in seeds:
        ga = GeneticAlgorithm(env, seed=seed, pop_size=pop_size, generations=generations, verbose=False)
        ga_best = ga.run()
        budget = ga.evaluator.n_evaluations                # random search gets exactly this many
        # random search is judged with the penalty of the GA's last generation
        rs_best, rs_curve, rs_time = random_search(env, budget, seed, generations=generations)
        ga_best.fitness = ga.evaluator.score(ga_best.objectives, generations)   # same penalty scale
        rows.append(metric_row("Genetic Algorithm", seed, ga_best, budget, ga.runtime_s))
        rows.append(metric_row("Random search", seed, rs_best, budget, rs_time))
        evals = [h["evaluations"] for h in ga.history]
        # best-so-far curve of the GA (running maximum), sampled once per generation
        ga_curve = np.maximum.accumulate([h["best_fitness_final_penalty"] for h in ga.history])
        ga_curves.append((evals, ga_curve))
        rs_curves.append((evals, [rs_curve[e - 1] for e in evals]))
        if examples is None:
            examples = (ga_best, rs_best, seed)
        print(f"  seed {seed}: GA {ga_best.fitness:.4f} vs random {rs_best.fitness:.4f} "
              f"({budget} evaluations each)")

    runs = pd.DataFrame(rows)
    runs.to_csv(os.path.join(out, "baseline_runs.csv"), index=False)
    table = runs.groupby("method")[METRICS + ["runtime_s"]].agg(["mean", "std"])
    table.columns = [f"{m}_{s}" for m, s in table.columns]
    table.to_csv(os.path.join(out, "baseline_summary.csv"))
    print(table.T.to_string())
    viz.plot_ga_vs_random(ga_curves, rs_curves, os.path.join(out, "ga_vs_random_convergence.png"))
    viz.plot_side_by_side(examples[0], examples[1], env, os.path.join(out, "ga_vs_random_solutions.png"))
    save_json({"seeds": seeds, "example_seed": examples[2], "ga_example_plan": examples[0].describe(),
               "random_example_plan": examples[1].describe()}, os.path.join(out, "baseline_examples.json"))


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="GA strategy engine for UAV fleet surveillance")
    parser.add_argument("--experiment", choices=["main", "sensitivity", "baseline", "all"], default="main")
    parser.add_argument("--seed", type=int, default=cfg.SEED, help="random seed of the main run")
    parser.add_argument("--generations", type=int, default=None, help="max generations")
    parser.add_argument("--pop-size", type=int, default=None, help="population size")
    parser.add_argument("--quick", action="store_true", help="tiny settings to test the pipeline")
    parser.add_argument("--output", default=None, help="output folder (default: results/)")
    args = parser.parse_args()

    out = args.output or ("results_quick" if args.quick else cfg.RESULTS_DIR)
    pop = args.pop_size or (QUICK["pop_size"] if args.quick else cfg.POP_SIZE)
    gens = args.generations or (QUICK["generations"] if args.quick else cfg.GENERATIONS)
    env = Environment()
    start = time.time()

    if args.experiment in ("main", "all"):
        run_main(env, out, args.seed, pop, gens)
    if args.experiment in ("sensitivity", "all"):
        run_sensitivity(env, os.path.join(out, "sensitivity"), args.quick)
    if args.experiment in ("baseline", "all"):
        seeds = QUICK["baseline_seeds"] if args.quick else cfg.BASELINE_SEEDS
        run_baseline(env, os.path.join(out, "baseline"), seeds, pop, gens)
    print(f"\nDone in {time.time() - start:.0f} s. Results are in '{out}/'.")


if __name__ == "__main__":
    main()
