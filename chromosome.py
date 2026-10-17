"""
The chromosome = the complete plan for the whole fleet.

It is a real-valued array of shape (N_DRONES, N_WAYPOINTS, 4).
Each drone has its own SEGMENT, and every waypoint has 4 genes:

    [ x , y , speed , sensor_power ]

    x, y          where the drone flies to (cells)
    speed         how fast it flies to this waypoint (SPEED_MIN..SPEED_MAX)
    sensor_power  0..1, a stronger sensor sees a bigger circle but uses more energy

Every drone flies: BASE -> waypoint 1 -> ... -> waypoint 6 -> BASE
"""
import numpy as np

import config as cfg


class Chromosome:
    def __init__(self, genes):
        self.genes = genes          # numpy array (N_DRONES, N_WAYPOINTS, 4)
        self.fitness = None         # filled in by the fitness function
        self.details = None         # the separate objective values

    @staticmethod
    def random(rng):
        """A random plan: random waypoints, speeds and sensor powers."""
        shape = (cfg.N_DRONES, cfg.N_WAYPOINTS)
        genes = np.stack([
            rng.uniform(0, cfg.GRID_SIZE, shape),            # x
            rng.uniform(0, cfg.GRID_SIZE, shape),            # y
            rng.uniform(cfg.SPEED_MIN, cfg.SPEED_MAX, shape),  # speed
            rng.uniform(0, 1, shape),                        # sensor power
        ], axis=2)
        return Chromosome(genes)

    def copy(self):
        new = Chromosome(self.genes.copy())
        new.fitness, new.details = self.fitness, self.details
        return new

    def route(self, drone):
        """All points one drone visits, from the base and back to the base."""
        return np.vstack([cfg.BASE, self.genes[drone, :, :2], cfg.BASE])

    def clip(self):
        """Keep every gene inside its allowed range."""
        g = self.genes
        g[:, :, 0:2] = np.clip(g[:, :, 0:2], 0, cfg.GRID_SIZE - 0.01)
        g[:, :, 2] = np.clip(g[:, :, 2], cfg.SPEED_MIN, cfg.SPEED_MAX)
        g[:, :, 3] = np.clip(g[:, :, 3], 0, 1)
