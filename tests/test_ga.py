"""Simple tests. Run with:  pytest"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from chromosome import Chromosome
from environment import Environment
from fitness import evaluate, repair
from ga import GeneticAlgorithm

env = Environment()
rng = np.random.default_rng(0)


def test_random_chromosome_has_right_shape():
    plan = Chromosome.random(rng)
    assert plan.genes.shape == (cfg.N_DRONES, cfg.N_WAYPOINTS, 4)


def test_repair_moves_waypoints_out_of_no_fly_zones():
    plan = Chromosome.random(rng)
    plan.genes[0, 0, 0:2] = (14.5, 15.5)             # inside the first no-fly zone
    repair(plan, env)
    for d in range(cfg.N_DRONES):
        for k in range(cfg.N_WAYPOINTS):
            assert not env.is_blocked(*plan.genes[d, k, 0:2])


def test_mutation_and_crossover_keep_genes_in_range():
    ga = GeneticAlgorithm(env, mutation_rate=1.0)
    p1, p2 = Chromosome.random(rng), Chromosome.random(rng)
    child = repair(ga.mutate(ga.crossover(p1, p2)), env)
    g = child.genes
    assert g[:, :, 0:2].min() >= 0 and g[:, :, 0:2].max() < cfg.GRID_SIZE
    assert g[:, :, 2].min() >= cfg.SPEED_MIN and g[:, :, 2].max() <= cfg.SPEED_MAX
    assert g[:, :, 3].min() >= 0 and g[:, :, 3].max() <= 1


def test_objectives_are_between_0_and_1():
    plan = repair(Chromosome.random(rng), env)
    evaluate(plan, env)
    for name in ("coverage", "detection", "energy", "time", "hazard"):
        assert 0 <= plan.details[name] <= 1


def test_ga_improves_the_best_fitness():
    ga = GeneticAlgorithm(env, pop_size=20, generations=10, seed=1)
    ga.run()
    assert ga.history[-1]["best"] >= ga.history[0]["best"]
