"""
The chromosome = the complete plan for the whole fleet (a hybrid real + binary encoding).

Each drone has its own SEGMENT:

    [ deploy_x, deploy_y | (x, y, speed, power, on) x N_WAYPOINTS ]

    deploy_x, deploy_y   deployment position: where the drone is launched and lands again
    x, y                 waypoint position (cells)
    speed                speed on the way to this waypoint (SPEED_MIN..SPEED_MAX)
    power                sensor power 0..1 (stronger sensor = bigger circle, more energy)
    on                   1 = sensor ON on this leg, 0 = OFF  (the sensor activation schedule)

Every drone flies: deployment position -> waypoint 1 -> ... -> waypoint 6 -> deployment position
"""
import numpy as np

import config as cfg


class Chromosome:
    def __init__(self, deploy, genes):
        self.deploy = deploy        # array (N_DRONES, 2): deployment position of each drone
        self.genes = genes          # array (N_DRONES, N_WAYPOINTS, 5): x, y, speed, power, on
        self.fitness = None         # filled in by the fitness function
        self.details = None         # the separate objective values

    @staticmethod
    def random(rng):
        """A random plan: random deployment positions, waypoints, speeds, powers and on/off bits."""
        shape = (cfg.N_DRONES, cfg.N_WAYPOINTS)
        deploy = rng.uniform(0, cfg.GRID_SIZE, (cfg.N_DRONES, 2))
        genes = np.stack([
            rng.uniform(0, cfg.GRID_SIZE, shape),              # x
            rng.uniform(0, cfg.GRID_SIZE, shape),              # y
            rng.uniform(cfg.SPEED_MIN, cfg.SPEED_MAX, shape),  # speed
            rng.uniform(0, 1, shape),                          # sensor power
            rng.integers(0, 2, shape),                         # sensor on (1) / off (0)
        ], axis=2).astype(float)
        return Chromosome(deploy, genes)

    def copy(self):
        new = Chromosome(self.deploy.copy(), self.genes.copy())
        new.fitness, new.details = self.fitness, self.details
        return new

    def route(self, drone):
        """All points one drone visits: deployment position -> waypoints -> deployment position."""
        start = self.deploy[drone]
        return np.vstack([start, self.genes[drone, :, :2], start])

    def clip(self):
        """Keep every gene inside its allowed range."""
        g = self.genes
        self.deploy[:] = np.clip(self.deploy, 0, cfg.GRID_SIZE - 0.01)
        g[:, :, 0:2] = np.clip(g[:, :, 0:2], 0, cfg.GRID_SIZE - 0.01)
        g[:, :, 2] = np.clip(g[:, :, 2], cfg.SPEED_MIN, cfg.SPEED_MAX)
        g[:, :, 3] = np.clip(g[:, :, 3], 0, 1)
        g[:, :, 4] = np.round(np.clip(g[:, :, 4], 0, 1))      # the on/off gene is always 0 or 1
