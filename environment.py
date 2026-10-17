"""
The grid world: no-fly zones, hazard zones, priority (disaster) zones and targets.
Arrays are indexed as grid[x, y].
"""
import numpy as np

import config as cfg


class Environment:
    def __init__(self):
        n = cfg.GRID_SIZE
        self.size = n

        # 1 = no-fly (blocked), 0 = open
        self.no_fly = np.zeros((n, n), dtype=bool)
        for x1, y1, x2, y2 in cfg.NO_FLY_ZONES:
            self.no_fly[x1:x2, y1:y2] = True

        # centre coordinates of every cell (used to check which cells a sensor sees)
        xs, ys = np.meshgrid(np.arange(n) + 0.5, np.arange(n) + 0.5, indexing="ij")
        self.cell_x, self.cell_y = xs.ravel().astype(np.float32), ys.ravel().astype(np.float32)

        # hazard: 1 in the middle of a hazard zone, falling to 0 at its edge
        self.hazard = np.zeros((n, n))
        for cx, cy, r in cfg.HAZARD_ZONES:
            dist = np.hypot(xs - cx, ys - cy)
            self.hazard = np.maximum(self.hazard, np.clip(1 - dist / r, 0, 1))

        # priority: how important it is to survey each cell
        self.priority = np.full((n, n), cfg.BACKGROUND_PRIORITY)
        for x1, y1, x2, y2, weight in cfg.PRIORITY_ZONES:
            self.priority[x1:x2, y1:y2] = weight
        self.priority[self.no_fly] = 0.0

        # targets: random open cells inside the priority zones (same every run)
        rng = np.random.default_rng(cfg.MAP_SEED)
        cells = np.argwhere((self.priority >= 0.5) & ~self.no_fly)
        chosen = rng.choice(len(cells), cfg.N_TARGETS, replace=False)
        self.targets = cells[chosen] + 0.5

    def is_blocked(self, x, y):
        """True if the point (x, y) is outside the map or inside a no-fly zone."""
        if not (0 <= x < self.size and 0 <= y < self.size):
            return True
        return bool(self.no_fly[int(x), int(y)])

    def nearest_free(self, x, y):
        """Closest open cell centre to (x, y): search in growing squares around it."""
        cx, cy = int(np.clip(x, 0, self.size - 1)), int(np.clip(y, 0, self.size - 1))
        for r in range(1, self.size):
            for i in range(cx - r, cx + r + 1):
                for j in range(cy - r, cy + r + 1):
                    if not self.is_blocked(i, j):
                        return i + 0.5, j + 0.5
        return x, y
