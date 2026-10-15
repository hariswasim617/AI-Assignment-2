"""
Baseline: pure random search.

Random search creates completely random fleet plans (with the SAME random
generator and the SAME repair as the GA's first generation) and simply keeps
the best one. It gets exactly the same number of fitness evaluations as the
GA, so any difference in quality comes from the GA's selection, crossover
and mutation, not from a bigger budget.
"""

import time

import numpy as np

from uav_ga import config as cfg
from uav_ga.chromosome import Chromosome
from uav_ga.constraints import repair
from uav_ga.environment import Environment
from uav_ga.fitness import FitnessEvaluator


def random_search(env: Environment, n_evaluations: int, seed: int,
                  generations: int = cfg.GENERATIONS) -> tuple[Chromosome, list[float], float]:
    """Evaluate ``n_evaluations`` random plans and return the best one.

    The fitness is computed with the penalty factor of the GA's LAST
    generation, which is the scale on which the final GA solution is judged.

    Returns (best chromosome, best-so-far fitness after every evaluation, runtime in s).
    """
    start = time.time()
    rng = np.random.default_rng(seed)
    evaluator = FitnessEvaluator(env, generations, cfg.CONSTRAINT_MODE)
    best, best_so_far = None, []
    for _ in range(n_evaluations):
        chrom = repair(Chromosome.random(env, rng), env, cfg.CONSTRAINT_MODE)
        evaluator.evaluate(chrom, generations)
        if best is None or chrom.fitness > best.fitness:
            best = chrom
        best_so_far.append(best.fitness)
    return best, best_so_far, time.time() - start
