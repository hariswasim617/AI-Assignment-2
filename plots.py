"""All figures of the project (matplotlib)."""
import matplotlib
matplotlib.use("Agg")                      # save figures to files (no window needed)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

import config as cfg
from fitness import coverage_map

DRONE_COLOURS = ["#2a78d6", "#d55181", "#008300"]


def draw_map(ax, env):
    """Hazard (orange), no-fly zones (black), priority zones (blue boxes) and targets."""
    n = env.size
    ax.imshow(env.hazard.T, origin="lower", extent=(0, n, 0, n), cmap="Oranges", vmin=0, vmax=1.5)
    blocked = np.ma.masked_where(~env.no_fly.T, env.no_fly.T)
    ax.imshow(blocked, origin="lower", extent=(0, n, 0, n), cmap="Greys", vmin=0, vmax=1)
    for x1, y1, x2, y2, w in cfg.PRIORITY_ZONES:
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="#1f3a8a", lw=1.5, ls="--"))
    ax.scatter(env.targets[:, 0], env.targets[:, 1], marker="x", c="black", s=25, zorder=4)
    ax.set_xlim(0, n); ax.set_ylim(0, n); ax.set_aspect("equal")
    ax.set_xlabel("x (cells)"); ax.set_ylabel("y (cells)")


def draw_routes(ax, plan):
    """Each drone's route: solid = sensor ON (thicker = stronger), dotted = sensor OFF.
    The triangle is the drone's deployment position."""
    for d in range(cfg.N_DRONES):
        route = plan.route(d)
        for k in range(len(route) - 1):
            if k < cfg.N_WAYPOINTS and plan.genes[d, k, 4] == 1:
                width, style = 1 + 2.5 * plan.genes[d, k, 3], "-"
            else:
                width, style = 1, ":"
            ax.plot(route[k:k + 2, 0], route[k:k + 2, 1], color=DRONE_COLOURS[d], lw=width, ls=style)
        ax.scatter(route[1:-1, 0], route[1:-1, 1], color=DRONE_COLOURS[d], s=14, zorder=3)
        ax.scatter(*plan.deploy[d], marker="^", color=DRONE_COLOURS[d], edgecolors="black", s=110, zorder=5)


def legend(ax):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    items = [Line2D([], [], color=c, lw=2, label=f"Drone {i + 1}") for i, c in enumerate(DRONE_COLOURS)]
    items += [Line2D([], [], color="grey", lw=2.5, label="sensor ON (thicker = stronger)"),
              Line2D([], [], color="grey", ls=":", label="sensor OFF"),
              Patch(color="#555555", label="no-fly zone"), Patch(color="#f4a261", label="hazard zone"),
              Patch(fill=False, ec="#1f3a8a", ls="--", label="priority zone"),
              Line2D([], [], marker="x", ls="", color="black", label="target"),
              Line2D([], [], marker="^", ls="", color="grey", mec="black", label="deployment position")]
    ax.legend(handles=items, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)


def title_of(plan):
    d = plan.details
    return (f"F={plan.fitness:.3f}  cov={d['coverage']:.2f}  det={d['detection']:.2f}\n"
            f"energy={d['energy']:.2f}  time={d['time']:.2f}  hazard={d['hazard']:.2f}  penalty={d['penalty']:.2f}")


def plot_convergence(history, path):
    gens = [h["generation"] for h in history]
    plt.figure(figsize=(7, 4))
    plt.plot(gens, [h["best"] for h in history], label="best fitness", lw=2)
    plt.plot(gens, [h["average"] for h in history], label="average fitness", lw=2)
    plt.xlabel("generation"); plt.ylabel("fitness"); plt.title("Convergence: best vs. average fitness")
    plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_paths_over_generations(ga, env, gens, path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for ax, g in zip(axes.ravel(), gens):
        plan = ga.best_per_generation[g]
        draw_map(ax, env); draw_routes(ax, plan)
        ax.set_title(f"Generation {g}\n{title_of(plan)}", fontsize=8)
    legend(axes[0, 1])
    fig.suptitle("Best flight paths across generations")
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_coverage_over_generations(ga, env, gens, path):
    fig, axes = plt.subplots(1, len(gens), figsize=(4 * len(gens), 4.2))
    for ax, g in zip(axes, gens):
        plan = ga.best_per_generation[g]
        seen = np.where(coverage_map(plan, env), env.priority, np.nan)   # NaN = not covered (white)
        im = ax.imshow(seen.T, origin="lower", extent=(0, env.size, 0, env.size), cmap="viridis", vmin=0, vmax=1)
        blocked = np.ma.masked_where(~env.no_fly.T, env.no_fly.T)
        ax.imshow(blocked, origin="lower", extent=(0, env.size, 0, env.size), cmap="Greys", vmin=0, vmax=1)
        ax.set_title(f"Generation {g}: coverage {plan.details['coverage']:.2f}", fontsize=9)
        ax.set_xlabel("x (cells)")
    axes[0].set_ylabel("y (cells)")
    fig.colorbar(im, ax=list(axes), label="priority of covered cell (white = not covered)", shrink=0.8)
    fig.suptitle("Coverage heatmaps across generations")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_sensitivity(curves, path):
    """curves = {sweep name: {setting label: [best-fitness curve of each seed]}}"""
    fig, axes = plt.subplots(1, len(curves), figsize=(5 * len(curves), 4))
    for ax, (name, settings) in zip(axes, curves.items()):
        for label, runs in settings.items():
            mean = np.mean(runs, axis=0)
            ax.plot(mean, lw=2, label=label)
        ax.set_title(name); ax.set_xlabel("generation"); ax.set_ylabel("best fitness (mean of seeds)")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_ga_vs_random(ga_curves, rs_curves, path):
    plt.figure(figsize=(7, 4))
    for label, curves in (("Genetic Algorithm", ga_curves), ("Random search", rs_curves)):
        x = curves[0][0]
        mean = np.mean([c[1] for c in curves], axis=0)
        plt.plot(x, mean, lw=2, label=f"{label} (mean of {len(curves)} seeds)")
    plt.xlabel("number of fitness evaluations"); plt.ylabel("best fitness found so far")
    plt.title("GA vs. random search (same number of evaluations)")
    plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_two_plans(ga_plan, rs_plan, env, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, plan, name in ((axes[0], ga_plan, "Genetic Algorithm"), (axes[1], rs_plan, "Random search")):
        draw_map(ax, env); draw_routes(ax, plan)
        ax.set_title(f"{name}\n{title_of(plan)}", fontsize=9)
    legend(axes[1])
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
