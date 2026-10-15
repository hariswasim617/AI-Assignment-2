"""
The 2-D grid world (the "hazard grid") that the drones survey.

The map is a GRID_SIZE x GRID_SIZE matrix. Each cell stores:
    * blocked  : True if the cell is a no-fly zone (value 1), else open (0)
    * hazard   : 0..1, how dangerous the cell is (wind + radiation pockets)
    * wind     : 0..1, the wind part of the hazard (it also costs energy)
    * priority : 0..1, how important it is to survey the cell (disaster zones)

Coordinate convention (used in the whole project):
    a point is (x, y) with 0 <= x, y < GRID_SIZE. The cell that contains the
    point is (int(x), int(y)). NumPy arrays are indexed as array[y, x]
    (row = y, column = x) so that ``plt.imshow(array, origin="lower")``
    draws the map the right way round.
"""

from collections import deque
import heapq

import numpy as np

from uav_ga import config as cfg


def gaussian_blob(size: int, cx: float, cy: float, sigma: float) -> np.ndarray:
    """Return a size x size array with a 2-D Gaussian bump centred at (cx, cy).

    The peak value is 1 and it falls smoothly to 0 far away. We use this to
    model hazard pockets and disaster zones, which are strongest in the
    middle and weaker at the edges.
    """
    ys, xs = np.mgrid[0:size, 0:size] + 0.5          # cell centres
    dist_sq = (xs - cx) ** 2 + (ys - cy) ** 2
    return np.exp(-dist_sq / (2.0 * sigma ** 2))


class Environment:
    """The survey area: no-fly zones, hazard field, priority map and targets.

    The map is generated from a fixed random seed (cfg.ENV_SEED), so every
    experiment and every GA run sees exactly the same world. Only the GA
    itself uses different seeds.
    """

    def __init__(self, seed: int = cfg.ENV_SEED, size: int = cfg.GRID_SIZE) -> None:
        self.size = size
        self.cell_size_m = cfg.CELL_SIZE_M
        self.base_stations = np.array(cfg.BASE_STATIONS, dtype=float)
        rng = np.random.default_rng(seed)

        self.blocked = self._make_nofly_zones(rng)
        self.priority_zones = []       # list of (cx, cy, sigma, weight) for plotting
        self.priority = self._make_priority_map(rng)
        self.wind, self.radiation = self._make_hazard_fields(rng)
        self.hazard = np.clip(self.wind + self.radiation, 0.0, 1.0)
        self.targets = self._place_targets(rng)

        # Repair uses a slightly "inflated" no-fly mask (1 extra cell around
        # every zone) so repaired paths keep a safety margin from the border.
        self.blocked_margin = self._inflate(self.blocked, 1)
        self.nearest_free = self._nearest_free_cells()
        self.priority_total = float(self.priority.sum())
        self.cell_centres_x = np.arange(size) + 0.5   # used by the coverage code
        self._astar_cache = {}   # remembers A* answers: the same detour is asked for often

    # ------------------------------------------------------------------
    # Map generation
    # ------------------------------------------------------------------
    def _make_nofly_zones(self, rng: np.random.Generator) -> np.ndarray:
        """Place rectangles and circles of restricted airspace.

        Zones are not allowed near a base station (drones must be able to
        take off) and must leave one big connected open area so every open
        cell can be reached.
        """
        n = self.size
        blocked = np.zeros((n, n), dtype=bool)
        ys, xs = np.mgrid[0:n, 0:n] + 0.5
        placed = 0
        while placed < cfg.N_NOFLY_ZONES:
            cx, cy = rng.uniform(12, n - 12, size=2)
            if rng.random() < 0.5:     # rectangle
                half_w, half_h = rng.uniform(3, 9, size=2)
                shape = (np.abs(xs - cx) <= half_w) & (np.abs(ys - cy) <= half_h)
            else:                      # circle
                radius = rng.uniform(4, 8)
                shape = (xs - cx) ** 2 + (ys - cy) ** 2 <= radius ** 2
            # keep a clear 12-cell circle around every base station
            too_close = any(np.hypot(cx - bx, cy - by) < 12 + 9
                            for bx, by in self.base_stations)
            if too_close:
                continue
            blocked |= shape
            placed += 1
        return blocked

    def _make_priority_map(self, rng: np.random.Generator) -> np.ndarray:
        """Disaster zones = Gaussian blobs with an importance weight 0.5..1."""
        n = self.size
        priority = np.full((n, n), cfg.BACKGROUND_PRIORITY)
        for _ in range(cfg.N_PRIORITY_ZONES):
            # pick a centre that is not inside a no-fly zone
            while True:
                cx, cy = rng.uniform(10, n - 10, size=2)
                if not self.blocked[int(cy), int(cx)]:
                    break
            sigma = rng.uniform(5, 10)
            weight = rng.uniform(0.5, 1.0)
            self.priority_zones.append((cx, cy, sigma, weight))
            priority = np.maximum(priority, weight * gaussian_blob(n, cx, cy, sigma))
        priority[self.blocked] = 0.0      # we never need to survey restricted airspace
        return priority

    def _make_hazard_fields(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Wind and radiation pockets, each a sum of Gaussian bumps in 0..1.

        Two of the radiation pockets are placed close to disaster zones on
        purpose: in a real disaster the important area is often also the
        dangerous one, which creates a true trade-off for the GA.
        """
        n = self.size
        wind = np.zeros((n, n))
        for _ in range(cfg.N_WIND_POCKETS):
            cx, cy = rng.uniform(0, n, size=2)
            wind += rng.uniform(0.6, 1.0) * gaussian_blob(n, cx, cy, rng.uniform(6, 12))
        radiation = np.zeros((n, n))
        for i in range(cfg.N_RADIATION_POCKETS):
            if i < 2 and i < len(self.priority_zones):
                zx, zy, _, _ = self.priority_zones[i]
                cx, cy = zx + rng.uniform(-6, 6), zy + rng.uniform(-6, 6)
            else:
                cx, cy = rng.uniform(0, n, size=2)
            radiation += rng.uniform(0.6, 1.0) * gaussian_blob(n, cx, cy, rng.uniform(4, 8))
        return np.clip(wind, 0, 1), np.clip(radiation, 0, 1)

    def _place_targets(self, rng: np.random.Generator) -> np.ndarray:
        """Put targets mostly inside disaster zones (75 %) and some anywhere."""
        open_cells = np.argwhere(~self.blocked)            # rows of (y, x)
        weights = self.priority[open_cells[:, 0], open_cells[:, 1]] ** 2
        n_zone = int(round(0.75 * cfg.N_TARGETS))
        pick_zone = rng.choice(len(open_cells), size=n_zone, replace=False,
                               p=weights / weights.sum())
        pick_any = rng.choice(len(open_cells), size=cfg.N_TARGETS - n_zone, replace=False)
        cells = open_cells[np.concatenate([pick_zone, pick_any])]
        # convert (y, x) rows to (x, y) points in the middle of the cell
        return np.column_stack([cells[:, 1] + 0.5, cells[:, 0] + 0.5])

    @staticmethod
    def _inflate(mask: np.ndarray, margin: int) -> np.ndarray:
        """Grow every True region of ``mask`` by ``margin`` cells (8 directions)."""
        grown = mask.copy()
        for dy in range(-margin, margin + 1):
            for dx in range(-margin, margin + 1):
                shifted = np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
                # np.roll wraps around the border; undo that wrap-around
                if dy > 0: shifted[:dy, :] = False
                if dy < 0: shifted[dy:, :] = False
                if dx > 0: shifted[:, :dx] = False
                if dx < 0: shifted[:, dx:] = False
                grown |= shifted
        return grown

    # ------------------------------------------------------------------
    # Search helpers used by the repair operator (BFS and A*)
    # ------------------------------------------------------------------
    def _nearest_free_cells(self) -> np.ndarray:
        """Multi-source Breadth-First Search from all free cells.

        Result: array[y, x] = (fx, fy), the closest free cell to (x, y).
        It is computed once, so moving a waypoint out of a no-fly zone later
        is just a table lookup.
        """
        n = self.size
        nearest = np.full((n, n, 2), -1, dtype=int)
        queue = deque()
        for y, x in np.argwhere(~self.blocked_margin):
            nearest[y, x] = (x, y)
            queue.append((x, y))
        while queue:
            x, y = queue.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < n and 0 <= ny < n and nearest[ny, nx, 0] < 0:
                    nearest[ny, nx] = nearest[y, x]   # inherit the source cell
                    queue.append((nx, ny))
        return nearest

    def astar(self, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]] | None:
        """A* search on the grid (8 neighbours) avoiding the inflated no-fly mask.

        Returns the list of cells from start to goal, or None if no path
        exists. The heuristic is the straight-line (Euclidean) distance,
        which never over-estimates, so the path found is the shortest one.
        """
        if (start, goal) in self._astar_cache:
            return self._astar_cache[(start, goal)]
        path = self._astar_search(start, goal)
        self._astar_cache[(start, goal)] = path
        return path

    def _astar_search(self, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]] | None:
        """The actual A* loop (called by astar(), which adds caching)."""
        n = self.size
        free = ~self.blocked_margin
        moves = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                 (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
        open_heap = [(0.0, start)]
        came_from = {start: None}
        cost = {start: 0.0}
        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current == goal:
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]
            cx, cy = current
            for dx, dy, step in moves:
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < n and 0 <= ny < n) or not free[ny, nx]:
                    continue
                # do not cut corners of a blocked cell when moving diagonally
                if dx and dy and (not free[cy, nx] or not free[ny, cx]):
                    continue
                new_cost = cost[current] + step
                if new_cost < cost.get((nx, ny), np.inf):
                    cost[(nx, ny)] = new_cost
                    came_from[(nx, ny)] = current
                    h = np.hypot(goal[0] - nx, goal[1] - ny)
                    heapq.heappush(open_heap, (new_cost + h, (nx, ny)))
        return None

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def in_bounds(self, point: np.ndarray) -> bool:
        return 0 <= point[0] < self.size and 0 <= point[1] < self.size

    def is_free(self, point: np.ndarray, with_margin: bool = True) -> bool:
        """True if the cell containing ``point`` is open airspace."""
        mask = self.blocked_margin if with_margin else self.blocked
        return self.in_bounds(point) and not mask[int(point[1]), int(point[0])]

    def segment_is_clear(self, a: np.ndarray, b: np.ndarray, with_margin: bool = True) -> bool:
        """True if the straight line a -> b never enters a no-fly cell.

        We sample points every 0.5 cells along the line (a simple and robust
        alternative to Bresenham's algorithm for real-valued end points).
        Because the margin mask makes every zone at least 3 cells wide, a
        0.5-cell step can never jump over a no-fly cell.
        """
        mask = self.blocked_margin if with_margin else self.blocked
        length = float(np.hypot(*(b - a)))
        n_samples = max(2, int(np.ceil(length / 0.5)) + 1)
        t = np.linspace(0.0, 1.0, n_samples)
        xs = np.clip((a[0] + t * (b[0] - a[0])).astype(int), 0, self.size - 1)
        ys = np.clip((a[1] + t * (b[1] - a[1])).astype(int), 0, self.size - 1)
        return not mask[ys, xs].any()

    def sample_legs(self, starts: np.ndarray, ends: np.ndarray, step: float = 0.5) -> dict:
        """Walk along many straight legs at once and read the map under them.

        Each leg i goes from starts[i] to ends[i]. Points are sampled every
        ``step`` cells. For every leg we return its length (cells), the mean
        wind and hazard under it, and how many samples were in no-fly cells.
        This is done with NumPy for all legs together because it is the most
        frequently called code in the whole GA.
        """
        vectors = ends - starts
        lengths = np.hypot(vectors[:, 0], vectors[:, 1])
        counts = np.maximum(1, np.ceil(lengths / step).astype(int))
        leg_id = np.repeat(np.arange(len(starts)), counts)
        # position of each sample inside its leg: 0.5/count, 1.5/count, ...
        first_index = np.cumsum(counts) - counts
        local = np.arange(counts.sum()) - np.repeat(first_index, counts)
        t = (local + 0.5) / counts[leg_id]
        px = starts[leg_id, 0] + t * vectors[leg_id, 0]
        py = starts[leg_id, 1] + t * vectors[leg_id, 1]
        xi = np.clip(px.astype(int), 0, self.size - 1)
        yi = np.clip(py.astype(int), 0, self.size - 1)
        return {
            "length": lengths,
            "mean_wind": np.bincount(leg_id, self.wind[yi, xi], len(starts)) / counts,
            "mean_hazard": np.bincount(leg_id, self.hazard[yi, xi], len(starts)) / counts,
            "blocked_samples": np.bincount(leg_id, self.blocked[yi, xi], len(starts)),
            "samples": counts,
        }
