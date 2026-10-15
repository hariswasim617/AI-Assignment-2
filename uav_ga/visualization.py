"""
All plots of the project (matplotlib only).

Map conventions used in every figure:
    * orange shading       hazard intensity (wind + radiation), 0..1
    * dark grey hatched    no-fly zones
    * black dashed line    border of high-priority disaster zones (priority >= 0.5)
    * circles              targets (filled = detected with probability >= 0.5)
    * black triangles      base stations
    * coloured lines       drone routes; SOLID = sensor ON, DASHED = sensor OFF
"""

import matplotlib
matplotlib.use("Agg")          # draw to files, no window needed (works on servers and Colab)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from uav_ga.chromosome import Chromosome
from uav_ga.environment import Environment
from uav_ga.fitness import coverage_map, detection_probabilities, build_legs

DRONE_COLOURS = ["#2a78d6", "#6250d6", "#008300", "#d55181", "#eb6834", "#1baf7a"]
SERIES_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#d55181", "#6250d6"]
NOFLY_COLOUR = "#3d3d3d"

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "axes.grid": False,
                     "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 8})


def _save(fig: plt.Figure, path: str) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Maps
# ---------------------------------------------------------------------------
def draw_environment(ax: plt.Axes, env: Environment, show_hazard: bool = True) -> None:
    """Draw hazard field, no-fly zones, priority zones and base stations."""
    extent = (0, env.size, 0, env.size)
    if show_hazard:
        ax.imshow(env.hazard, origin="lower", extent=extent, cmap="Oranges", vmin=0, vmax=1.3, alpha=0.9)
    nofly = np.ma.masked_where(~env.blocked, env.blocked.astype(float))
    ax.imshow(nofly, origin="lower", extent=extent, cmap=ListedColormap([NOFLY_COLOUR]), alpha=0.9)
    ax.contourf(env.cell_centres_x, env.cell_centres_x, env.blocked.astype(float), levels=[0.5, 1.5],
                colors="none", hatches=["////"])
    ax.contour(env.cell_centres_x, env.cell_centres_x, env.priority, levels=[0.5],
               colors="black", linestyles="--", linewidths=1.0)
    ax.scatter(env.base_stations[:, 0], env.base_stations[:, 1], marker="^", s=110,
               c="black", edgecolors="white", linewidths=1.2, zorder=6)
    ax.set_xlim(0, env.size); ax.set_ylim(0, env.size)
    ax.set_aspect("equal")
    ax.set_xlabel("x (cells, 10 m)"); ax.set_ylabel("y (cells, 10 m)")


def draw_routes(ax: plt.Axes, chrom: Chromosome, env: Environment, show_targets: bool = True) -> None:
    """Draw every drone route; solid legs = sensor on, dashed legs = sensor off."""
    for d, seg in enumerate(chrom.segments):
        colour = DRONE_COLOURS[d % len(DRONE_COLOURS)]
        route = seg.route(env)
        sensor = np.append(seg.sensor_on, False)          # return leg: sensor off
        for k in range(len(route) - 1):
            ax.plot(route[k:k + 2, 0], route[k:k + 2, 1], color=colour,
                    linestyle="-" if sensor[k] else "--",
                    linewidth=2.2 if sensor[k] else 1.1, zorder=4)
        ax.scatter(seg.waypoints[:, 0], seg.waypoints[:, 1], s=12, color=colour, zorder=5)
    if show_targets:
        p = detection_probabilities(build_legs(chrom, env), env)
        found = p >= 0.5
        ax.scatter(env.targets[found, 0], env.targets[found, 1], s=28, c="black",
                   edgecolors="white", linewidths=0.8, zorder=7)
        ax.scatter(env.targets[~found, 0], env.targets[~found, 1], s=28, facecolors="white",
                   edgecolors="black", linewidths=1.0, zorder=7)


def map_legend(ax: plt.Axes, n_drones: int, loc: str = "upper right") -> None:
    handles = [Line2D([], [], color=DRONE_COLOURS[d], lw=2.2, label=f"Drone {d + 1}") for d in range(n_drones)]
    handles += [
        Line2D([], [], color="grey", lw=2.2, label="sensor ON"),
        Line2D([], [], color="grey", lw=1.1, ls="--", label="sensor OFF"),
        Patch(facecolor=NOFLY_COLOUR, hatch="////", label="no-fly zone"),
        Patch(facecolor="#f4a261", label="hazard (darker = worse)"),
        Line2D([], [], color="black", ls="--", label="priority zone (P >= 0.5)"),
        Line2D([], [], marker="o", ls="", mfc="black", mec="black", label="target detected"),
        Line2D([], [], marker="o", ls="", mfc="white", mec="black", label="target missed"),
        Line2D([], [], marker="^", ls="", mfc="black", mec="black", ms=8, label="base station"),
    ]
    ax.legend(handles=handles, loc=loc, framealpha=0.92)


def short_stats(obj: dict, fitness: float) -> str:
    return (f"F={fitness:.3f} | cov={obj['coverage']:.2f} det={obj['detection']:.2f} "
            f"E={obj['energy']:.2f} T={obj['time']:.2f} H={obj['hazard']:.2f}")


def plot_environment(env: Environment, path: str) -> None:
    """The map on its own: hazard, priority map, no-fly zones, targets."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    draw_environment(axes[0], env)
    axes[0].scatter(env.targets[:, 0], env.targets[:, 1], s=22, facecolors="white", edgecolors="black", zorder=7)
    axes[0].set_title("Hazard field, no-fly zones, targets, bases")
    im = axes[1].imshow(env.priority, origin="lower", extent=(0, env.size, 0, env.size), cmap="Blues", vmin=0, vmax=1)
    nofly = np.ma.masked_where(~env.blocked, env.blocked.astype(float))
    axes[1].imshow(nofly, origin="lower", extent=(0, env.size, 0, env.size), cmap=ListedColormap([NOFLY_COLOUR]))
    axes[1].set_title("Priority map (disaster zones)")
    axes[1].set_xlabel("x (cells, 10 m)"); axes[1].set_ylabel("y (cells, 10 m)")
    fig.colorbar(im, ax=axes[1], label="priority weight")
    map_legend(axes[0], 0, loc="lower right")
    _save(fig, path)


def plot_routes_over_generations(snapshots: list[tuple[int, Chromosome]], env: Environment, path: str) -> None:
    """2 x 2 grid: best fleet plan at four generations."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 11))
    for ax, (gen, chrom) in zip(axes.ravel(), snapshots):
        draw_environment(ax, env)
        draw_routes(ax, chrom, env)
        ax.set_title(f"Generation {gen}\n{short_stats(chrom.objectives, chrom.fitness)}", fontsize=9)
    map_legend(axes[0, 0], len(snapshots[0][1].segments), loc="upper right")
    fig.suptitle("Best UAV flight plan across generations", fontsize=13)
    _save(fig, path)


def plot_coverage_heatmaps(snapshots: list[tuple[int, Chromosome, list[Chromosome]]], env: Environment, path: str) -> None:
    """Row 1: priority-weighted coverage of the best plan.
    Row 2: share of the whole population whose sensors cover each cell."""
    fig, axes = plt.subplots(2, len(snapshots), figsize=(4.2 * len(snapshots), 8.4))
    for col, (gen, best, population) in enumerate(snapshots):
        covered = coverage_map(best, env)
        shown = np.where(covered, env.priority, np.nan)
        ax = axes[0, col]
        ax.imshow(np.zeros_like(env.priority), origin="lower", extent=(0, env.size, 0, env.size), cmap="Greys", vmin=0, vmax=1)
        im1 = ax.imshow(shown, origin="lower", extent=(0, env.size, 0, env.size), cmap="viridis", vmin=0, vmax=1)
        draw_environment(ax, env, show_hazard=False)
        ax.set_title(f"Gen {gen}: best plan coverage\n(priority-weighted = {best.objectives['coverage']:.2f})", fontsize=9)
        freq = np.mean([coverage_map(c, env) for c in population], axis=0)
        ax = axes[1, col]
        im2 = ax.imshow(freq, origin="lower", extent=(0, env.size, 0, env.size), cmap="magma", vmin=0, vmax=1)
        draw_environment(ax, env, show_hazard=False)
        ax.set_title(f"Gen {gen}: population coverage frequency", fontsize=9)
        if col > 0:                       # only the first column needs y-axis labels
            axes[0, col].set_ylabel(""); axes[1, col].set_ylabel("")
    fig.colorbar(im1, ax=axes[0, :].tolist(), shrink=0.8, label="priority of covered cell (white = not covered)")
    fig.colorbar(im2, ax=axes[1, :].tolist(), shrink=0.8, label="share of population covering the cell")
    fig.suptitle("Coverage heatmaps across generations", fontsize=13)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_final_solution(chrom: Chromosome, env: Environment, path: str, title: str) -> None:
    """One large, annotated map of a single fleet plan."""
    fig, ax = plt.subplots(figsize=(10, 10))
    draw_environment(ax, env)
    draw_routes(ax, chrom, env)
    for d, seg in enumerate(chrom.segments):
        ax.annotate(f"D{d + 1}", seg.waypoints[0], xytext=(4, 4), textcoords="offset points",
                    color=DRONE_COLOURS[d], fontsize=10, fontweight="bold")
    o = chrom.objectives
    ax.set_title(f"{title}\nfitness {chrom.fitness:.3f} | coverage {o['coverage']:.2f} | detected "
                 f"{o['targets_detected']}/{len(env.targets)} targets | energy {o['energy_kj']:.0f} kJ | "
                 f"makespan {o['makespan_s']:.0f} s | hazard {o['hazard']:.2f}", fontsize=10)
    map_legend(ax, len(chrom.segments))
    _save(fig, path)


# ---------------------------------------------------------------------------
# Curves
# ---------------------------------------------------------------------------
def plot_convergence(history, path: str) -> None:
    """Best and average fitness per generation (average shown with +/- 1 std band)."""
    g = history["generation"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.fill_between(g, history["mean_fitness"] - history["std_fitness"],
                    history["mean_fitness"] + history["std_fitness"], color=SERIES_COLOURS[1], alpha=0.18,
                    label="average +/- 1 std")
    ax.plot(g, history["mean_fitness"], color=SERIES_COLOURS[1], lw=2, label="average fitness")
    ax.plot(g, history["best_fitness"], color=SERIES_COLOURS[0], lw=2, label="best fitness")
    ax.set_xlabel("generation"); ax.set_ylabel("fitness F")
    ax.set_title("GA convergence: best vs. average fitness")
    ax.grid(alpha=0.3); ax.legend(loc="lower right")
    _save(fig, path)


def plot_diversity(history, path: str) -> None:
    """Population diversity, with generations that received immigrants marked."""
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)
    axes[0].plot(history["generation"], history["diversity"], color=SERIES_COLOURS[0], lw=2)
    boosted = history[history["immigrants"] > 0]
    axes[0].scatter(boosted["generation"], boosted["diversity"], color=SERIES_COLOURS[1], s=14,
                    zorder=3, label="immigrants added + mutation boosted")
    axes[0].set_ylabel("mean pairwise distance"); axes[0].set_title("Population diversity")
    axes[0].grid(alpha=0.3)
    if len(boosted):
        axes[0].legend()
    axes[1].plot(history["generation"], history["sigma"], color=SERIES_COLOURS[2], lw=2)
    axes[1].set_ylabel("mutation step sigma (cells)"); axes[1].set_xlabel("generation")
    axes[1].set_title("Annealed waypoint mutation step"); axes[1].grid(alpha=0.3)
    _save(fig, path)


def plot_objectives(history, path: str) -> None:
    """Each raw objective of the best individual over the generations (small multiples)."""
    names = [("coverage", "Coverage (max)"), ("detection", "Detection rate (max)"),
             ("energy", "Energy (min)"), ("time", "Flight time (min)"),
             ("hazard", "Hazard exposure (min)"), ("violation", "Constraint violation (min)")]
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.5), sharex=True)
    for ax, (key, label) in zip(axes.ravel(), names):
        ax.plot(history["generation"], history[key], color=SERIES_COLOURS[0], lw=2)
        ax.set_title(label); ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("generation")
    for ax in axes[:, 0]:
        ax.set_ylabel("normalised value")
    fig.suptitle("Objectives of the best individual per generation", fontsize=12)
    _save(fig, path)


def plot_sweep(curves: dict, title: str, path: str, x_label: str = "generation") -> None:
    """Mean best-fitness curve (+/- std over seeds) for each setting of a sweep."""
    fig, ax = plt.subplots(figsize=(7, 4.3))
    for i, (label, runs) in enumerate(curves.items()):
        length = min(len(r) for r in runs)
        data = np.array([r[:length] for r in runs])
        mean, std = data.mean(axis=0), data.std(axis=0)
        x = np.arange(length)
        colour = SERIES_COLOURS[i % len(SERIES_COLOURS)]
        ax.plot(x, mean, color=colour, lw=2, label=label)
        ax.fill_between(x, mean - std, mean + std, color=colour, alpha=0.15)
    ax.set_xlabel(x_label); ax.set_ylabel("best fitness (mean of seeds)")
    ax.set_title(title); ax.grid(alpha=0.3); ax.legend(loc="lower right")
    _save(fig, path)


def plot_ga_vs_random(ga_curves: list, rs_curves: list, path: str) -> None:
    """Best-so-far fitness against the number of fitness evaluations."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, runs, colour in (("Genetic Algorithm", ga_curves, SERIES_COLOURS[0]),
                                ("Random search", rs_curves, SERIES_COLOURS[1])):
        length = min(len(r[0]) for r in runs)
        evals = runs[0][0][:length]
        data = np.array([r[1][:length] for r in runs])
        mean, std = data.mean(axis=0), data.std(axis=0)
        ax.plot(evals, mean, color=colour, lw=2, label=f"{label} (mean of {len(runs)} seeds)")
        ax.fill_between(evals, mean - std, mean + std, color=colour, alpha=0.18)
    ax.set_xlabel("fitness evaluations"); ax.set_ylabel("best fitness found so far")
    ax.set_title("GA vs. random search with the same evaluation budget")
    ax.grid(alpha=0.3); ax.legend(loc="lower right")
    _save(fig, path)


def plot_side_by_side(ga_best: Chromosome, rs_best: Chromosome, env: Environment, path: str) -> None:
    """Routes (top) and coverage (bottom) of the GA solution next to the random-search one."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 11))
    for col, (name, chrom) in enumerate((("Genetic Algorithm", ga_best), ("Random search", rs_best))):
        draw_environment(axes[0, col], env)
        draw_routes(axes[0, col], chrom, env)
        axes[0, col].set_title(f"{name}: routes\n{short_stats(chrom.objectives, chrom.fitness)}", fontsize=9)
        covered = np.where(coverage_map(chrom, env), env.priority, np.nan)
        im = axes[1, col].imshow(covered, origin="lower", extent=(0, env.size, 0, env.size),
                                 cmap="viridis", vmin=0, vmax=1)
        draw_environment(axes[1, col], env, show_hazard=False)
        axes[1, col].set_title(f"{name}: covered cells (colour = priority)\n"
                               f"high-priority cells covered: {100 * chrom.objectives['high_priority_coverage']:.0f}%",
                               fontsize=9)
    map_legend(axes[0, 0], len(ga_best.segments))
    fig.colorbar(im, ax=axes[1, :].tolist(), shrink=0.7, label="priority of covered cell")
    fig.suptitle("Best GA plan vs. best random-search plan (same seed, same budget)", fontsize=13)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
