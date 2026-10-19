"""
Runs everything and saves the results in the results/ folder.

    In Colab: Runtime -> Run all   (all experiments take about 10 minutes)
"""
import csv
import os

import numpy as np

import config as cfg
import plots
from environment import Environment
from ga import GeneticAlgorithm
from random_search import random_search


def save_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def experiment_main(env, out, generations):
    """One GA run with the default settings + all its figures."""
    print("1) Main GA run")
    ga = GeneticAlgorithm(env, generations=generations)
    best = ga.run()
    last = len(ga.history) - 1
    gens = [0, last // 4, last // 2, last]                 # 4 snapshots for the maps
    save_csv(ga.history, f"{out}/history.csv")
    plots.plot_convergence(ga.history, f"{out}/convergence.png")
    plots.plot_paths_over_generations(ga, env, gens, f"{out}/paths_generations.png")
    plots.plot_coverage_over_generations(ga, env, gens, f"{out}/coverage_heatmaps.png")
    summary = {"stop_reason": ga.stop_reason, "generations_run": last, "evaluations": ga.evaluations,
               "first_best": ga.history[0]["best"], "first_average": ga.history[0]["average"],
               "final_best": best.fitness, "final_average": ga.history[-1]["average"], **best.details}
    save_csv([summary], f"{out}/main_summary.csv")
    print(f"   best fitness {ga.history[0]['best']:.3f} -> {best.fitness:.3f} ({ga.stop_reason})")


def experiment_sensitivity(env, out, generations, seeds):
    """Change ONE setting at a time and see how the final best fitness changes."""
    print("2) Sensitivity analysis")
    sweeps = {"Population size": ("pop_size", [20, 50, 100]),
              "Mutation rate": ("mutation_rate", [0.02, 0.1, 0.3]),
              "Selection pressure (tournament k)": ("tournament_k", [2, 3, 5])}
    rows, curves = [], {}
    for name, (parameter, values) in sweeps.items():
        curves[name] = {}
        for value in values:
            finals, runs = [], []
            for seed in seeds:
                ga = GeneticAlgorithm(env, generations=generations, seed=seed,
                                      patience=generations, **{parameter: value})
                best = ga.run()
                finals.append(best.fitness)
                runs.append([h["best"] for h in ga.history])
            curves[name][f"{parameter} = {value}"] = runs
            rows.append({"parameter": parameter, "value": value, "mean_best": np.mean(finals),
                         "std_best": np.std(finals), "evaluations": ga.evaluations})
            print(f"   {parameter} = {value}: {np.mean(finals):.3f} +/- {np.std(finals):.3f}")
    save_csv(rows, f"{out}/sensitivity.csv")
    plots.plot_sensitivity(curves, f"{out}/sensitivity.png")


def experiment_baseline(env, out, generations, seeds):
    """GA vs. random search with exactly the same number of evaluations."""
    print("3) GA vs. random search")
    rows, ga_curves, rs_curves = [], [], []
    for seed in seeds:
        ga = GeneticAlgorithm(env, generations=generations, seed=seed)
        ga_best = ga.run()
        rs_best, rs_curve = random_search(env, ga.evaluations, seed)
        evals = [h["evaluations"] for h in ga.history]
        ga_curves.append((evals, [h["best"] for h in ga.history]))
        rs_curves.append((evals, [rs_curve[e - 1] for e in evals]))
        for method, plan in (("GA", ga_best), ("Random search", rs_best)):
            rows.append({"method": method, "seed": seed, "fitness": plan.fitness,
                         "evaluations": ga.evaluations, **plan.details})
        print(f"   seed {seed}: GA {ga_best.fitness:.3f}  random {rs_best.fitness:.3f}")
        if seed == seeds[0]:
            plots.plot_two_plans(ga_best, rs_best, env, f"{out}/ga_vs_random_maps.png")
    save_csv(rows, f"{out}/baseline.csv")
    plots.plot_ga_vs_random(ga_curves, rs_curves, f"{out}/ga_vs_random.png")
    for method in ("GA", "Random search"):
        values = [r["fitness"] for r in rows if r["method"] == method]
        print(f"   {method}: {np.mean(values):.3f} +/- {np.std(values):.3f}")


if __name__ == "__main__":
    out = "results"
    os.makedirs(out, exist_ok=True)
    env = Environment()
    experiment_main(env, out, cfg.GENERATIONS)
    experiment_sensitivity(env, out, cfg.GENERATIONS, seeds=[1, 2, 3])
    experiment_baseline(env, out, cfg.GENERATIONS, seeds=[1, 2, 3, 4, 5])
    print(f"Done. Results are in {out}/")
