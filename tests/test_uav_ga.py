"""
Unit tests (run with:  pytest -q).

They check the most important promises of the code:
    * random chromosomes are valid (all genes inside their ranges)
    * repair removes every no-fly violation and respects range / battery
    * crossover and mutation (followed by clipping) keep genes in bounds
    * the fitness function behaves sensibly
"""

import numpy as np
import pytest

from uav_ga import config as cfg
from uav_ga.chromosome import Chromosome, DroneSegment
from uav_ga.constraints import clip_genes, repair
from uav_ga.environment import Environment
from uav_ga.fitness import FitnessEvaluator, flight_power
from uav_ga.ga import GeneticAlgorithm
from uav_ga.operators import (blx_alpha, crossover, mutate, rank_selection,
                              roulette_selection, tournament_selection)


@pytest.fixture(scope="module")
def env():
    return Environment()


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_environment_layers_are_valid(env):
    assert env.blocked.shape == (cfg.GRID_SIZE, cfg.GRID_SIZE)
    assert 0 <= env.hazard.min() and env.hazard.max() <= 1
    assert 0 <= env.priority.min() and env.priority.max() <= 1
    assert all(env.is_free(b) for b in env.base_stations)          # drones can take off
    assert not any(env.blocked[int(y), int(x)] for x, y in env.targets)


def test_random_chromosome_is_valid(env, rng):
    for _ in range(20):
        chrom = Chromosome.random(env, rng)
        assert len(chrom.segments) == cfg.N_DRONES
        assert chrom.is_within_bounds(env)
        assert all(cfg.K_INIT_MIN <= s.n_waypoints <= cfg.K_INIT_MAX for s in chrom.segments)


def test_repair_removes_nofly_violations(env, rng):
    evaluator = FitnessEvaluator(env)
    for _ in range(30):
        chrom = repair(Chromosome.random(env, rng), env, "repair")
        evaluator.evaluate(chrom, 0)
        assert chrom.objectives["nofly_fraction"] == 0.0
        for seg in chrom.segments:
            route = seg.route(env)
            for a, b in zip(route[:-1], route[1:]):
                assert env.segment_is_clear(a, b, with_margin=False)


def test_repair_moves_waypoint_out_of_nofly_zone(env):
    blocked_y, blocked_x = np.argwhere(env.blocked)[0]
    seg = DroneSegment(0, [[blocked_x + 0.5, blocked_y + 0.5], [50.5, 50.5]],
                       [10.0, 10.0], [0.5, 0.5], [True, True])
    chrom = repair(Chromosome([seg] * 1), env, "repair")
    assert all(env.is_free(p, with_margin=False) for p in chrom.segments[0].waypoints)


def test_repair_respects_range_and_battery(env, rng):
    evaluator = FitnessEvaluator(env)
    for _ in range(30):
        chrom = repair(Chromosome.random(env, rng), env, "repair")
        evaluator.evaluate(chrom, 0)
        assert chrom.objectives["range_overrun"] == 0.0
        assert chrom.objectives["battery_overrun"] < 0.01   # only a 1-waypoint drone may be left over


def test_crossover_and_mutation_respect_bounds(env, rng):
    parents = [repair(Chromosome.random(env, rng), env) for _ in range(10)]
    for i in range(0, 10, 2):
        for child in crossover(parents[i], parents[i + 1], rng):
            mutate(child, rate=0.5, sigma=20.0, env=env, rng=rng)
            clip_genes(child, env)
            assert child.is_within_bounds(env)
            assert all(cfg.K_MIN <= s.n_waypoints for s in child.segments)


def test_crossover_does_not_change_parents(env, rng):
    p1, p2 = Chromosome.random(env, rng), Chromosome.random(env, rng)
    before = p1.segments[0].waypoints.copy()
    crossover(p1, p2, rng)
    assert np.array_equal(before, p1.segments[0].waypoints)


def test_blx_alpha_stays_in_widened_interval(rng):
    x, y = np.zeros(1000), np.ones(1000)
    child = blx_alpha(x, y, 0.3, rng)
    assert child.min() >= -0.3 and child.max() <= 1.3


def test_selection_prefers_fitter_individuals(rng):
    fitness = np.arange(50, dtype=float)
    for picks in (tournament_selection(fitness, 2000, 5, rng),
                  rank_selection(fitness, 2000, 1.8, rng),
                  roulette_selection(fitness, 2000, rng)):
        assert picks.mean() > 24.5            # better than a uniform random choice


def test_fitness_sanity(env, rng):
    evaluator = FitnessEvaluator(env)
    chrom = repair(Chromosome.random(env, rng), env)
    evaluator.evaluate(chrom, 0)
    for key in ("coverage", "detection", "energy", "time", "hazard"):
        assert 0.0 <= chrom.objectives[key] <= 1.0
    # switching every sensor off must give zero coverage and zero detection
    silent = chrom.copy()
    for seg in silent.segments:
        seg.sensor_on[:] = False
    silent.forget_fitness()
    evaluator.evaluate(silent, 0)
    assert silent.objectives["coverage"] == 0.0 and silent.objectives["detection"] == 0.0
    assert silent.objectives["energy"] < chrom.objectives["energy"]   # sensors cost energy


def test_dynamic_penalty_grows(env):
    evaluator = FitnessEvaluator(env, total_generations=100, constraint_mode="both")
    assert evaluator.penalty_factor(0) < evaluator.penalty_factor(50) < evaluator.penalty_factor(100)
    assert FitnessEvaluator(env, 100, "repair").penalty_factor(50) == 0.0


def test_flight_power_is_u_shaped():
    speeds = np.linspace(0.5, 30, 60)
    power = flight_power(speeds)
    lowest = int(np.argmin(power))
    assert 0 < lowest < len(speeds) - 1      # minimum is in the middle, not at an end


def test_ga_improves_and_is_reproducible(env):
    runs = [GeneticAlgorithm(env, seed=3, pop_size=16, generations=6, verbose=False) for _ in range(2)]
    best = [ga.run().fitness for ga in runs]
    assert best[0] == best[1]                                   # same seed -> same result
    history = runs[0].history
    assert history[-1]["best_fitness"] >= history[0]["best_fitness"] - 0.05
    assert runs[0].stop_reason.startswith("max generations")
