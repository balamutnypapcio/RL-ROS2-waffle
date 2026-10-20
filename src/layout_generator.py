"""Generate grid layouts (maps) for the navigation environment.

Pure Python + NumPy, no MuJoCo here. A layout is:
    - a grid of shape (grid_size, grid_size): FREE (0) or WALL (1),
    - a start cell (row, col) and start heading (yaw),
    - a goal cell (row, col).

Convention (same as make_scene.cell_center):
    row 0 = bottom (smallest y), col 0 = left (smallest x).
    grid[row, col] -> cell at row `row`, column `col`.

Later, at every env reset:
    layout = generate_layout(...)
    for each WALL cell -> move one block from the pool to cell_center(row, col)
    park the remaining blocks, place the robot at start and the goal marker at goal.

Usage (when finished), to look at maps in the terminal:
    python layout_generator.py --grid 8 --mode random --obstacles 12 --count 3 --seed 0
"""
import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FREE = 0
WALL = 1
MODES = ("random", "maze", "rooms")

Cell = tuple[int, int]   # (row, col)


@dataclass
class Layout:
    grid: np.ndarray      # shape (grid_size, grid_size), values FREE / WALL
    start: Cell
    start_yaw: float      # radians, 0 = facing +x
    goal: Cell


# --------------------------------------------------------------------------- #
# Basic grid helpers
# --------------------------------------------------------------------------- #
def empty_grid(grid_size: int) -> np.ndarray:
    """Return a grid with all cells FREE (dtype int8 is enough)."""
    return np.full((grid_size, grid_size), FREE, dtype=np.int8)


def in_bounds(cell: Cell, grid_size: int) -> bool:
    """True if (row, col) is inside the grid."""
    row, col = cell
    return 0 <= row < grid_size and 0 <= col < grid_size
    



def neighbors(cell: Cell, grid: np.ndarray) -> list[Cell]:
    """FREE neighbors of `cell`: up, down, left, right (4-connectivity).

    Why no diagonals: two blocks touching only at a corner leave a zero-width
    gap - the robot cannot drive through it, so BFS must not either.
    """
    row, col = cell
    cells_around = [(1,0),(-1,0),(0,1),(0,-1)]
    neighbors_list = []
    for c in cells_around:
        candidate = (row + c[0], col + c[1])
        if in_bounds(candidate, grid.shape[0]) and grid[candidate] == FREE:
            neighbors_list.append(candidate)
    
    return neighbors_list




def free_cells(grid: np.ndarray) -> list[Cell]:
    """All FREE cells as a list of (row, col). Hint: np.argwhere."""
    return [tuple(p) for p in np.argwhere(grid == FREE)]


# --------------------------------------------------------------------------- #
# Map types (each returns a grid of FREE / WALL)
# --------------------------------------------------------------------------- #
def random_obstacles(grid_size: int, n_obstacles: int, rng: np.random.Generator) -> np.ndarray:
    """Scatter `n_obstacles` WALL cells at random free positions.

    Hint: rng.choice over flat indices without replacement, then np.unravel_index.
    Simplest map type - start with this one.
    """
    if n_obstacles >= grid_size*grid_size:
        raise ValueError(f"n_obstacles={n_obstacles} does not fit in a {grid_size}x{grid_size} grid")

    n_cells = grid_size*grid_size # 0-(0,0), 1-(0,1) and more
    selected = rng.choice(n_cells, size=n_obstacles, replace=False)
    rows, cols = np.unravel_index(selected,(grid_size, grid_size))
    grid = empty_grid(grid_size)
    grid[rows,cols] = WALL
    return grid

def maze(grid_size: int, rng: np.random.Generator) -> np.ndarray:
    """Maze with 1-cell-wide corridors (recursive backtracker / randomized DFS).

    Rooms are cells with EVEN (row, col): 0, 2, 4, ... The cells between two
    rooms are walls that can be knocked down. The outer arena walls (from the
    XML) already form the border, so the maze has no border of its own.
    For an even grid_size the last row and column don't fit the room pattern;
    they are left FREE as a corridor along the arena wall.
    """
    grid = np.full((grid_size, grid_size), WALL, dtype=np.int8)
    jumps = [(2, 0), (-2, 0), (0, 2), (0, -2)]

    # Start in a random room (even row, even col)
    n_rooms = (grid_size + 1) // 2
    start = (2 * int(rng.integers(n_rooms)), 2 * int(rng.integers(n_rooms)))
    grid[start] = FREE
    stack = [start]

    while stack:
        row, col = stack[-1]
        # Neighbor rooms 2 cells away that are inside the grid and not visited yet
        unvisited = []
        for dr, dc in jumps:
            nxt = (row + dr, col + dc)
            if in_bounds(nxt, grid_size) and grid[nxt] == WALL:
                unvisited.append((dr, dc))

        if unvisited:
            dr, dc = unvisited[rng.integers(len(unvisited))]
            grid[row + dr // 2, col + dc // 2] = FREE   # knock down the wall in between
            grid[row + dr, col + dc] = FREE              # enter the neighbor room
            stack.append((row + dr, col + dc))
        else:
            stack.pop()                                  # dead end -> go back

    if grid_size % 2 == 0:
        grid[-1, :] = FREE
        grid[:, -1] = FREE
    return grid


def rooms(grid_size: int, rng: np.random.Generator) -> np.ndarray:
    """Four rooms: one full horizontal and one full vertical wall line with doorways.

    The crossing splits each line into two segments and every segment gets one
    door, so all four rooms are connected (in a loop). Each line stays at least
    2 cells from the arena wall, so no room is narrower than 2 cells.
    Grids smaller than 5 have no room for this and stay empty.
    """
    grid = empty_grid(grid_size)
    if grid_size < 5:
        return grid

    wall_row = int(rng.integers(2, grid_size - 2))
    wall_col = int(rng.integers(2, grid_size - 2))
    grid[wall_row, :] = WALL
    grid[:, wall_col] = WALL

    # One door in each segment, on both sides of the crossing
    for lo, hi in [(0, wall_col), (wall_col + 1, grid_size)]:
        grid[wall_row, int(rng.integers(lo, hi))] = FREE
    for lo, hi in [(0, wall_row), (wall_row + 1, grid_size)]:
        grid[int(rng.integers(lo, hi)), wall_col] = FREE
    return grid


# --------------------------------------------------------------------------- #
# Reachability
# --------------------------------------------------------------------------- #
def bfs_distances(grid: np.ndarray, start: Cell) -> np.ndarray:
    """Breadth-first search from `start` over FREE cells.

    Return an int array of shape grid.shape with the number of steps from
    start to every cell, and -1 for walls and unreachable cells.
    Hint: collections.deque as the queue.
    Bonus: the distance to the goal is a nice measure of map difficulty.
    """
    if not in_bounds(start, grid.shape[0]):
        raise ValueError(f"start {start} is outside the {grid.shape[0]}x{grid.shape[0]} grid")

    if grid[start] == WALL:
        raise ValueError(f"start {start} is in the wall")
    
    distance_grid = np.full((grid.shape), -1, dtype=np.int32)
    distance_grid[start] = 0
    queue = deque([start])
    while queue:
        current = queue.popleft()
        for nb in neighbors(current,grid):
            if distance_grid[nb] == -1:
                distance_grid[nb] = distance_grid[current] + 1
                queue.append(nb)

    return distance_grid


def is_reachable(grid: np.ndarray, start: Cell, goal: Cell) -> bool:
    """True if the goal can be reached from start (use bfs_distances)."""
    return bfs_distances(grid, start)[goal] != -1


# --------------------------------------------------------------------------- #
# Start / goal
# --------------------------------------------------------------------------- #
def sample_start_goal(grid: np.ndarray, rng: np.random.Generator,
                      min_dist_cells: int) -> tuple[Cell, Cell] | None:
    """Pick a random FREE start and a FREE goal at least `min_dist_cells` BFS steps away.

    Return None if no such pair exists (e.g. map too small or too blocked),
    so the caller can generate a new grid.
    Hint: run BFS from start once, then choose the goal among cells with
    distance >= min_dist_cells.
    """
    cells = free_cells(grid)
    if not cells:
        return None
    
    start = cells[rng.integers(len(cells))]
    distances_grid = bfs_distances(grid, start)
    goal_candidates = [tuple(p) for p in np.argwhere(distances_grid >= min_dist_cells)]
    if not goal_candidates:
        return None
    
    goal = goal_candidates[rng.integers(len(goal_candidates))]
    return start, goal 


def sample_start_yaw(rng: np.random.Generator) -> float:
    """Random heading in [-pi, pi). (Later idea: face a free direction.)"""
    return float(rng.uniform(-np.pi, np.pi))


# --------------------------------------------------------------------------- #
# Main entry point used by the environment
# --------------------------------------------------------------------------- #
def generate_layout(grid_size: int, rng: np.random.Generator, mode: str = "random",
                    n_obstacles: int = 10, min_dist_cells: int = 4,
                    max_tries: int = 100) -> Layout:
    """Generate one valid layout.

    Steps:
        1. build a grid with the chosen `mode` ("random", "maze", "rooms"),
        2. sample start and goal (reachable, far enough apart),
        3. if that fails, try again (up to max_tries), then raise an error.
    `n_obstacles` (or a density) is the knob for curriculum learning.
    """
    if mode not in MODES:
        raise ValueError(f"mode={mode!r} must be one of {MODES}")

    for _ in range(max_tries):
        if mode == "random":
            grid = random_obstacles(grid_size, n_obstacles, rng)
        elif mode == "maze":
            grid = maze(grid_size, rng)
        else:  # "rooms" - the only option left, checked above
            grid = rooms(grid_size, rng)

        result = sample_start_goal(grid, rng, min_dist_cells)
        if result is None:
            continue

        start, goal = result
        start_yaw = sample_start_yaw(rng)

        return Layout(grid=grid, start=start, start_yaw=start_yaw ,goal=goal)

    raise RuntimeError(f"No valid layout after {max_tries} tries (mode={mode}, grid={grid_size}, n_obstacles={n_obstacles})")
    


# --------------------------------------------------------------------------- #
# Debugging and fixed test sets
# --------------------------------------------------------------------------- #
def layout_to_str(layout: Layout) -> str:
    """ASCII picture of a layout: '#' wall, '.' free, 'S' start, 'G' goal.

    Row grid_size-1 is printed FIRST, so the picture matches the top view
    (row 0 = bottom of the arena). Characters are separated by spaces so the
    cells look roughly square in a terminal.
    """
    chars = np.where(layout.grid == WALL, "#", ".")   # array of single-character strings
    chars[layout.start] = "S"
    chars[layout.goal] = "G"
    rows = [" ".join(row) for row in chars[::-1]]     # [::-1] reverses rows: top row first
    return "\n".join(rows)


def save_layouts(layouts: list[Layout], path: Path) -> None:
    """Save a list of layouts to one .npz file (a fixed test set for evaluation).

    Arrays in the file:
        grids  (N, n, n)  int8
        starts (N, 2)     int   - (row, col)
        goals  (N, 2)     int   - (row, col)
        yaws   (N,)       float - start heading in radians
    """
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")   # np.savez would add it anyway; keep the printed name honest
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        grids=np.stack([l.grid for l in layouts]),
        starts=np.array([l.start for l in layouts], dtype=int),
        goals=np.array([l.goal for l in layouts], dtype=int),
        yaws=np.array([l.start_yaw for l in layouts], dtype=float),
    )
    print(f"Saved {len(layouts)} layouts to {path}")


def load_layouts(path: Path) -> list[Layout]:
    """Inverse of save_layouts."""
    with np.load(path) as data:
        grids, starts, goals, yaws = data["grids"], data["starts"], data["goals"], data["yaws"]
    return [
        Layout(
            grid=grids[i],
            start=(int(starts[i][0]), int(starts[i][1])),
            start_yaw=float(yaws[i]),
            goal=(int(goals[i][0]), int(goals[i][1])),
        )
        for i in range(len(grids))
    ]


def parse_args() -> argparse.Namespace:
    """Arguments for quick testing: --grid, --mode, --obstacles, --count, --seed, --save."""
    parser = argparse.ArgumentParser(description="Generate grid layouts and print them as ASCII.")
    parser.add_argument("--grid", type=int, default=8, help="cells per side (default: 8)")
    parser.add_argument("--mode", choices=MODES, default="random", help="map type (default: random)")
    parser.add_argument("--obstacles", type=int, default=10,
                        help="number of obstacles for mode 'random' (default: 10)")
    parser.add_argument("--count", type=int, default=3, help="how many layouts to generate (default: 3)")
    parser.add_argument("--seed", type=int, default=None, help="random seed (default: different every run)")
    parser.add_argument("--save", type=str, default=None,
                        help="optional .npz path to save the layouts, e.g. data/test_layouts.npz")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    layouts = [generate_layout(args.grid, rng, mode=args.mode, n_obstacles=args.obstacles)
               for _ in range(args.count)]
    for i, layout in enumerate(layouts):
        print(f"--- layout {i} ---")
        print(layout_to_str(layout))
    if args.save:
        save_layouts(layouts, Path(args.save))


if __name__ == "__main__":
    main()
