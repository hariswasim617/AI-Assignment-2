"""
The Genetic Algorithm.

    1. make a random population
    2. evaluate every plan (fitness)
    3. keep the ELITE best plans
    4. fill the rest: tournament selection -> crossover -> mutation -> repair
    5. add a few random "immigrants" so the population stays diverse
    6. repeat until GENERATIONS is reached or nothing improved for PATIENCE generations
"""
import numpy as np

import config as cfg
from chromosome import Chromosome
from fitness import evaluate, repair


class GeneticAlgorithm:
    def __init__(self, env, pop_size=cfg.POP_SIZE, generations=cfg.GENERATIONS,
                 mutation_rate=cfg.MUTATION_RATE, tournament_k=cfg.TOURNAMENT_K, seed=cfg.SEED,
                 patience=cfg.PATIENCE):
        self.env = env
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.tournament_k = tournament_k
        self.patience = patience
        self.rng = np.random.default_rng(seed)     # all randomness comes from this seed
        self.evaluations = 0                       # how many plans were evaluated
        self.history = []                          # best / average fitness per generation
        self.best_per_generation = []              # best plan of every generation (for the maps)
        self.stop_reason = ""

    # ---------- population initialisation ----------
    def new_random_plan(self):
        plan = repair(Chromosome.random(self.rng), self.env)
        evaluate(plan, self.env)
        self.evaluations += 1
        return plan

    def initial_population(self):
        return [self.new_random_plan() for _ in range(self.pop_size)]

    # ---------- selection ----------
    def tournament_selection(self, population):
        """Pick k random plans and return the best of them."""
        contestants = self.rng.choice(len(population), self.tournament_k, replace=False)
        best = max(contestants, key=lambda i: population[i].fitness)
        return population[best]

    # ---------- crossover ----------
    def crossover(self, parent1, parent2):
        """Custom two-step crossover for drone routes.

        Step 1 - segment swap: each drone's whole route is copied from parent 1
                 or parent 2 (good routes stay together).
        Step 2 - geometric blend: for one random drone the child's waypoints are
                 a random mix of both parents' waypoints (a point between them).
        """
        child = parent1.genes.copy()
        for d in range(cfg.N_DRONES):
            if self.rng.random() < 0.5:
                child[d] = parent2.genes[d]
        d = self.rng.integers(cfg.N_DRONES)
        alpha = self.rng.random()
        child[d] = alpha * parent1.genes[d] + (1 - alpha) * parent2.genes[d]
        return Chromosome(child)

    # ---------- mutation ----------
    def mutate(self, plan):
        """Mutation designed for spatial routes.

        * move a waypoint a little (Gaussian step of about MUTATION_STEP cells)
        * change its speed and sensor power a little
        * sometimes swap the order of two waypoints of a drone
        """
        g = plan.genes
        for d in range(cfg.N_DRONES):
            for k in range(cfg.N_WAYPOINTS):
                if self.rng.random() < self.mutation_rate:
                    g[d, k, 0:2] += self.rng.normal(0, cfg.MUTATION_STEP, 2)
                    g[d, k, 2] += self.rng.normal(0, 0.3)
                    g[d, k, 3] += self.rng.normal(0, 0.2)
            if self.rng.random() < self.mutation_rate:
                i, j = self.rng.choice(cfg.N_WAYPOINTS, 2, replace=False)
                g[d, [i, j]] = g[d, [j, i]]
        return plan

    # ---------- main loop ----------
    def run(self):
        population = self.initial_population()
        best_ever, no_improvement = -np.inf, 0

        for generation in range(self.generations + 1):
            population.sort(key=lambda p: p.fitness, reverse=True)   # best first
            fitness = [p.fitness for p in population]
            self.history.append({"generation": generation, "best": fitness[0],
                                 "average": float(np.mean(fitness)), "evaluations": self.evaluations})
            self.best_per_generation.append(population[0].copy())

            # termination criteria
            if fitness[0] > best_ever + 1e-6:
                best_ever, no_improvement = fitness[0], 0
            else:
                no_improvement += 1
            if generation == self.generations:
                self.stop_reason = f"reached {self.generations} generations"
                break
            if no_improvement >= self.patience:
                self.stop_reason = f"no improvement for {self.patience} generations"
                break

            # next generation = elite + children + a few random immigrants
            next_population = [p.copy() for p in population[:cfg.ELITE]]
            while len(next_population) < self.pop_size - cfg.IMMIGRANTS:
                parent1 = self.tournament_selection(population)
                parent2 = self.tournament_selection(population)
                if self.rng.random() < cfg.CROSSOVER_RATE:
                    child = self.crossover(parent1, parent2)
                else:
                    child = parent1.copy()
                child = repair(self.mutate(child), self.env)
                evaluate(child, self.env)
                self.evaluations += 1
                next_population.append(child)
            while len(next_population) < self.pop_size:
                next_population.append(self.new_random_plan())
            population = next_population

        return population[0]
