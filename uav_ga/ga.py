"""
The Genetic Algorithm engine.

One generation:
    1. score every individual (fitness uses the penalty of THIS generation)
    2. record statistics (best, mean, std, diversity, objectives, ...)
    3. stop if a termination criterion fires
    4. copy the elite (top ELITISM %) unchanged
    5. fill the rest: select parents -> crossover -> mutate -> repair -> evaluate
    6. if diversity is too low: add random immigrants and boost mutation
"""

import time

import numpy as np

from uav_ga import config as cfg
from uav_ga.chromosome import Chromosome, path_features
from uav_ga.constraints import repair
from uav_ga.environment import Environment
from uav_ga.fitness import FitnessEvaluator
from uav_ga.operators import annealed_sigma, crossover, mutate, select_parents


def population_diversity(population: list[Chromosome], env: Environment) -> float:
    """Mean distance between all pairs of individuals (0 = everyone identical).

    Each chromosome is turned into a fixed-length feature vector (its routes
    resampled to 10 points each, see chromosome.path_features); the mean of
    all pairwise Euclidean distances is the diversity.
    """
    features = np.array([path_features(c, env) for c in population])
    diff = features[:, None, :] - features[None, :, :]
    distances = np.sqrt((diff ** 2).sum(axis=2))
    n = len(population)
    return float(distances.sum() / (n * (n - 1)))


class GeneticAlgorithm:
    """Evolves fleet plans. Every setting defaults to the value in config.py."""

    def __init__(self, env: Environment, seed: int = cfg.SEED, pop_size: int = cfg.POP_SIZE,
                 generations: int = cfg.GENERATIONS, mutation_rate: float = cfg.MUTATION_RATE,
                 crossover_rate: float = cfg.CROSSOVER_RATE, selection: str = cfg.SELECTION,
                 tournament_k: int = cfg.TOURNAMENT_K, constraint_mode: str = cfg.CONSTRAINT_MODE,
                 stagnation_patience: int = cfg.STAGNATION_PATIENCE,
                 keep_populations: bool = False, verbose: bool = True) -> None:
        self.env = env
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.selection = selection
        self.tournament_k = tournament_k
        self.constraint_mode = constraint_mode
        self.stagnation_patience = stagnation_patience
        self.n_elite = max(1, int(round(cfg.ELITISM * pop_size)))
        self.keep_populations = keep_populations
        self.verbose = verbose
        self.evaluator = FitnessEvaluator(env, generations, constraint_mode)

        self.history: list[dict] = []            # one row of statistics per generation
        self.best_per_generation: list[Chromosome] = []
        self.populations: list[list[Chromosome]] = []   # only if keep_populations=True
        self.best: Chromosome | None = None
        self.stop_reason = ""
        self.runtime_s = 0.0

    # ------------------------------------------------------------------
    def random_individual(self, generation: int) -> Chromosome:
        """Create, repair and evaluate one random chromosome."""
        chrom = repair(Chromosome.random(self.env, self.rng), self.env, self.constraint_mode)
        self.evaluator.evaluate(chrom, generation)
        return chrom

    def make_child_pairs(self, population: list[Chromosome], fitness: np.ndarray,
                         n_children: int, rate: float, sigma: float, generation: int) -> list[Chromosome]:
        """Selection -> crossover -> mutation -> repair -> evaluation."""
        parents = select_parents(fitness, n_children + 1, self.rng, self.selection, self.tournament_k)
        children = []
        for i in range(0, n_children, 2):
            p1, p2 = population[parents[i]], population[parents[i + 1]]
            if self.rng.random() < self.crossover_rate:
                c1, c2 = crossover(p1, p2, self.rng)
            else:
                c1, c2 = p1.copy(), p2.copy()
            for child in (c1, c2):
                mutate(child, rate, sigma, self.env, self.rng)
                repair(child, self.env, self.constraint_mode)
                self.evaluator.evaluate(child, generation)
                children.append(child)
        return children[:n_children]

    def record(self, generation: int, population: list[Chromosome], diversity: float,
               rate: float, sigma: float, immigrants: int, start_time: float) -> None:
        """Save one row of statistics for this generation."""
        fitness = np.array([c.fitness for c in population])
        best = population[int(np.argmax(fitness))]
        final_scale = [self.evaluator.score(c.objectives, self.generations) for c in population]
        obj = best.objectives
        self.history.append({
            "generation": generation,
            "evaluations": self.evaluator.n_evaluations,
            "best_fitness": float(fitness.max()),
            "mean_fitness": float(fitness.mean()),
            "std_fitness": float(fitness.std()),
            "worst_fitness": float(fitness.min()),
            "best_fitness_final_penalty": float(max(final_scale)),
            "diversity": diversity,
            "mutation_rate": rate,
            "sigma": sigma,
            "penalty_lambda": self.evaluator.penalty_factor(generation),
            "immigrants": immigrants,
            "feasible_share": float(np.mean([c.violation < 1e-9 for c in population])),
            "coverage": obj["coverage"], "detection": obj["detection"], "energy": obj["energy"],
            "time": obj["time"], "hazard": obj["hazard"], "violation": best.violation,
            "elapsed_s": time.time() - start_time,
        })
        self.best_per_generation.append(best.copy())
        if self.keep_populations:
            self.populations.append([c.copy() for c in population])

    # ------------------------------------------------------------------
    def run(self) -> Chromosome:
        """Run the GA until a termination criterion fires; return the best plan."""
        start = time.time()
        population = [self.random_individual(0) for _ in range(self.pop_size)]
        diversity_start = population_diversity(population, self.env)
        best_so_far, last_improvement = -np.inf, 0
        rate, immigrants = self.mutation_rate, 0

        for generation in range(self.generations + 1):
            # 1) re-score: the penalty factor depends on the generation number
            for chrom in population:
                self.evaluator.evaluate(chrom, generation)
            fitness = np.array([c.fitness for c in population])
            diversity = population_diversity(population, self.env)
            sigma = annealed_sigma(generation, self.generations)
            self.record(generation, population, diversity, rate, sigma, immigrants, start)
            if self.verbose and (generation % 20 == 0 or generation == self.generations):
                h = self.history[-1]
                print(f"  gen {generation:4d} | best {h['best_fitness']:.4f} | mean {h['mean_fitness']:.4f}"
                      f" | diversity {diversity:.3f} | evals {h['evaluations']}")

            # 2) termination criteria
            if fitness.max() > best_so_far + cfg.STAGNATION_EPS:
                best_so_far, last_improvement = fitness.max(), generation
            if generation >= self.generations:
                self.stop_reason = f"max generations ({self.generations}) reached"
                break
            if cfg.TARGET_FITNESS is not None and fitness.max() >= cfg.TARGET_FITNESS:
                self.stop_reason = f"target fitness {cfg.TARGET_FITNESS} reached"
                break
            if generation - last_improvement >= self.stagnation_patience:
                self.stop_reason = (f"stagnation: no improvement > {cfg.STAGNATION_EPS} "
                                    f"for {self.stagnation_patience} generations")
                break

            # 3) diversity control: low diversity -> immigrants + stronger mutation
            low_diversity = diversity < cfg.DIVERSITY_THRESHOLD * diversity_start
            rate = self.mutation_rate * (cfg.MUTATION_BOOST if low_diversity else 1.0)
            immigrants = int(round(cfg.IMMIGRANT_FRACTION * self.pop_size)) if low_diversity else 0

            # 4) elitism + offspring
            order = np.argsort(fitness)[::-1]
            elite = [population[i].copy() for i in order[:self.n_elite]]
            n_children = self.pop_size - self.n_elite - immigrants
            children = self.make_child_pairs(population, fitness, n_children, rate, sigma, generation + 1)
            newcomers = [self.random_individual(generation + 1) for _ in range(immigrants)]
            population = elite + children + newcomers

        final_fitness = np.array([c.fitness for c in population])
        self.best = population[int(np.argmax(final_fitness))].copy()
        self.final_population = population
        self.runtime_s = time.time() - start
        if self.verbose:
            print(f"  stopped: {self.stop_reason} | runtime {self.runtime_s:.1f} s")
        return self.best
