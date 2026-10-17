"""
Baseline: random search.
Make random plans (same generator and repair as the GA) and keep the best one.
It gets exactly as many evaluations as the GA, so the comparison is fair.
"""
import numpy as np

from chromosome import Chromosome
from fitness import evaluate, repair


def random_search(env, n_evaluations, seed):
    rng = np.random.default_rng(seed)
    best, best_so_far = None, []
    for _ in range(n_evaluations):
        plan = repair(Chromosome.random(rng), env)
        evaluate(plan, env)
        if best is None or plan.fitness > best.fitness:
            best = plan
        best_so_far.append(best.fitness)
    return best, best_so_far
