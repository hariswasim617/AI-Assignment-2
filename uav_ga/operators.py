"""
Genetic operators: selection, crossover and mutation.

All operators are simple functions. They receive a NumPy random generator
(``rng``) so that a run with the same seed always gives the same result.
"""

import numpy as np

from uav_ga import config as cfg
from uav_ga.chromosome import Chromosome, DroneSegment
from uav_ga.environment import Environment


# ---------------------------------------------------------------------------
# Selection: which individuals become parents?
# ---------------------------------------------------------------------------
def tournament_selection(fitness: np.ndarray, n_parents: int, k: int,
                         rng: np.random.Generator) -> np.ndarray:
    """Pick k random individuals and keep the best one; repeat n_parents times.

    k is the SELECTION PRESSURE: with k = 2 weak individuals still win quite
    often (more exploration), with k = 8 the best individuals win almost
    every tournament (more exploitation, risk of premature convergence).
    """
    contestants = rng.integers(0, len(fitness), size=(n_parents, k))
    winners = np.argmax(fitness[contestants], axis=1)
    return contestants[np.arange(n_parents), winners]


def rank_selection(fitness: np.ndarray, n_parents: int, pressure: float,
                   rng: np.random.Generator) -> np.ndarray:
    """Linear ranking: the chance depends on the RANK, not the raw fitness.

    Worst rank gets (2 - pressure) / N, best rank gets pressure / N.
    This keeps the selection pressure constant even when all fitness values
    are very close together (late in the run).
    """
    n = len(fitness)
    ranks = np.empty(n)
    ranks[np.argsort(fitness)] = np.arange(n)          # 0 = worst, n-1 = best
    probs = (2 - pressure + 2 * (pressure - 1) * ranks / max(1, n - 1)) / n
    return rng.choice(n, size=n_parents, p=probs / probs.sum())


def roulette_selection(fitness: np.ndarray, n_parents: int,
                       rng: np.random.Generator) -> np.ndarray:
    """Fitness-proportionate ("roulette wheel") selection.

    Our fitness can be negative, so we first shift it so the worst
    individual has a small positive weight.
    """
    weights = fitness - fitness.min() + 1e-6
    return rng.choice(len(fitness), size=n_parents, p=weights / weights.sum())


def select_parents(fitness: np.ndarray, n_parents: int, rng: np.random.Generator,
                   method: str = cfg.SELECTION, k: int = cfg.TOURNAMENT_K) -> np.ndarray:
    """Call the selection method chosen in the configuration."""
    if method == "tournament":
        return tournament_selection(fitness, n_parents, k, rng)
    if method == "rank":
        return rank_selection(fitness, n_parents, cfg.RANK_PRESSURE, rng)
    if method == "roulette":
        return roulette_selection(fitness, n_parents, rng)
    raise ValueError(f"unknown selection method: {method}")


# ---------------------------------------------------------------------------
# Crossover: mix two parents into two children
# ---------------------------------------------------------------------------
def blx_alpha(x: np.ndarray, y: np.ndarray, alpha: float, rng: np.random.Generator) -> np.ndarray:
    """BLX-alpha blend crossover for real numbers.

    The child value is drawn uniformly from the interval spanned by the two
    parents, widened by alpha * |x - y| on both sides. For waypoints this is
    a GEOMETRIC crossover: the child waypoint lands in the rectangle between
    the parents' waypoints (or slightly outside it, which keeps diversity).
    """
    low, high = np.minimum(x, y), np.maximum(x, y)
    spread = high - low
    return rng.uniform(low - alpha * spread, high + alpha * spread)


def blend_segments(a: DroneSegment, b: DroneSegment, rng: np.random.Generator) -> tuple[DroneSegment, DroneSegment]:
    """Blend two drone segments gene by gene (positions 0..min(Ka, Kb) - 1).

    * waypoints, speeds, powers -> BLX-alpha (real genes)
    * sensor bits               -> uniform crossover (each bit from either parent)
    Extra waypoints of the longer parent are kept unchanged.
    """
    child_a, child_b = a.copy(), b.copy()
    n = min(a.n_waypoints, b.n_waypoints)
    for child in (child_a, child_b):
        child.waypoints[:n] = blx_alpha(a.waypoints[:n], b.waypoints[:n], cfg.BLX_ALPHA, rng)
        child.speeds[:n] = blx_alpha(a.speeds[:n], b.speeds[:n], cfg.BLX_ALPHA, rng)
        child.powers[:n] = blx_alpha(a.powers[:n], b.powers[:n], cfg.BLX_ALPHA, rng)
    take_from_b = rng.random(n) < 0.5                     # uniform crossover mask
    child_a.sensor_on[:n] = np.where(take_from_b, b.sensor_on[:n], a.sensor_on[:n])
    child_b.sensor_on[:n] = np.where(take_from_b, a.sensor_on[:n], b.sensor_on[:n])
    return child_a, child_b


def crossover(parent1: Chromosome, parent2: Chromosome, rng: np.random.Generator) -> tuple[Chromosome, Chromosome]:
    """Two-level crossover designed for the multi-segment chromosome.

    Level 1 - drone-segment swap: each drone's whole route is taken from
              parent 1 or parent 2. Good routes are kept intact (a route is
              a "building block" that a cut in the middle could destroy).
    Level 2 - for each drone, with probability 0.5, the two versions are
              blended (BLX-alpha on coordinates, uniform on sensor bits).
    """
    segs1 = [s.copy() for s in parent1.segments]
    segs2 = [s.copy() for s in parent2.segments]
    for d in range(len(segs1)):
        if rng.random() < cfg.SEGMENT_SWAP_PROB:
            segs1[d], segs2[d] = segs2[d], segs1[d]
        if rng.random() < 0.5:
            segs1[d], segs2[d] = blend_segments(segs1[d], segs2[d], rng)
    return Chromosome(segs1), Chromosome(segs2)


# ---------------------------------------------------------------------------
# Mutation: small random changes that keep the population diverse
# ---------------------------------------------------------------------------
def annealed_sigma(generation: int, total_generations: int) -> float:
    """Step size of the waypoint mutation: large at the start, small at the end.

    sigma(g) = SIGMA_START * (SIGMA_END / SIGMA_START) ** (g / G)
    Big steps early let drones explore the whole map; small steps late
    fine-tune good routes (like simulated annealing).
    """
    progress = min(1.0, generation / max(1, total_generations))
    return cfg.SIGMA_START * (cfg.SIGMA_END / cfg.SIGMA_START) ** progress


def mutate_segment(seg: DroneSegment, rate: float, sigma: float, env: Environment,
                   rng: np.random.Generator) -> DroneSegment:
    """Apply every mutation type to one drone segment (each gene with probability ``rate``)."""
    if rng.random() < cfg.RESET_PROB:                       # rare: brand-new random route
        return DroneSegment.random(env, rng)
    k = seg.n_waypoints

    # 1) Gaussian perturbation of waypoint coordinates (spatial mutation)
    moved = rng.random(k) < rate
    seg.waypoints[moved] += rng.normal(0.0, sigma, size=(moved.sum(), 2))
    # 2) speed jitter and 3) sensor power jitter
    changed = rng.random(k) < rate
    seg.speeds[changed] += rng.normal(0.0, 0.1 * (cfg.V_MAX - cfg.V_MIN), size=changed.sum())
    changed = rng.random(k) < rate
    seg.powers[changed] += rng.normal(0.0, 0.15, size=changed.sum())
    # 4) sensor bit flip (changes the activation schedule)
    flipped = rng.random(k) < rate
    seg.sensor_on[flipped] = ~seg.sensor_on[flipped]
    # 5) launch from a different base station (integer gene)
    if rng.random() < rate / 2:
        seg.base = int(rng.integers(0, len(env.base_stations)))

    # 6) structural mutations: insert / delete / swap waypoints
    if rng.random() < rate and seg.n_waypoints < cfg.K_MAX:
        i = int(rng.integers(0, seg.n_waypoints))          # insert after waypoint i
        route = seg.route(env)
        new_point = (route[i + 1] + route[i + 2]) / 2 + rng.normal(0.0, sigma, 2)
        seg.waypoints = np.insert(seg.waypoints, i + 1, new_point, axis=0)
        seg.speeds = np.insert(seg.speeds, i + 1, seg.speeds[i])
        seg.powers = np.insert(seg.powers, i + 1, seg.powers[i])
        seg.sensor_on = np.insert(seg.sensor_on, i + 1, seg.sensor_on[i])
    if rng.random() < rate and seg.n_waypoints > cfg.K_MIN:
        i = int(rng.integers(0, seg.n_waypoints))
        seg.waypoints = np.delete(seg.waypoints, i, axis=0)
        seg.speeds, seg.powers = np.delete(seg.speeds, i), np.delete(seg.powers, i)
        seg.sensor_on = np.delete(seg.sensor_on, i)
    if rng.random() < rate and seg.n_waypoints >= 2:
        i, j = rng.choice(seg.n_waypoints, size=2, replace=False)
        seg.waypoints[[i, j]] = seg.waypoints[[j, i]]      # visit order changes
    return seg


def mutate(chrom: Chromosome, rate: float, sigma: float, env: Environment,
           rng: np.random.Generator) -> Chromosome:
    """Mutate every drone segment of a chromosome (in place)."""
    chrom.segments = [mutate_segment(s, rate, sigma, env, rng) for s in chrom.segments]
    chrom.forget_fitness()
    return chrom
