"""Generate a MuJoCo scene (MJCF/XML) for grid-based navigation maps.

The scene contains:
    - a floor,
    - outer walls around a square arena of grid_size x grid_size cells,
    - a POOL of movable blocks (mocap bodies), one per grid cell,
    - a goal marker (mocap body, no collision),
    - the robot, included from a separate XML file.

This script only builds the "warehouse of blocks". Arranging the blocks into an
actual map happens later, in Python, at every reset (by moving mocap bodies).

Usage (when finished):
    python make_scene.py --grid 8 --cell 0.5 --out models/scene_grid.xml
    python -m mujoco.viewer --mjcf models/scene_grid.xml
"""
import argparse
import math
from pathlib import Path

# Robot model to include in the scene (path relative to the generated XML file).
ROBOT_XML = "turtlebot3_waffle_pi.xml"

# Where unused blocks are "parked" so they don't affect physics or the LiDAR.
PARKING_POS = (100.0, 100.0, -1.0)

WALL_THICKNESS = 0.1    # full thickness of the outer walls [m]
GOAL_RADIUS = 0.15      # radius of the goal marker (visual only) [m]

ROOT = Path(__file__).resolve().parent.parent   # project/
MODEL_DIR = ROOT / "models"

# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate a grid-based MuJoCo navigation scene.")
    parser.add_argument("--grid", type=int, default=8,
                        help="number of cells per side (default: 8)")
    parser.add_argument("--cell", type=float, default=0.5,
                        help="cell size in meters (default: 0.5)")
    parser.add_argument("--wall-height", type=float, default=0.3,
                        help="height of walls and blocks in meters (default: 0.3)")
    parser.add_argument("--out", type=Path, default=MODEL_DIR / "scene_grid.xml",
                        help="output XML path (default: models/scene_grid.xml)")
    args = parser.parse_args()

    if args.grid < 1:
        parser.error("--grid must be >= 1")
    if args.cell <= 0 or args.wall_height <= 0:
        parser.error("--cell and --wall-height must be > 0")
    return args


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
"""Return half of the arena side length in meters (arena is centered at 0, 0)."""
def arena_half_size(grid_size: int, cell_size: float) -> float:
    return grid_size*cell_size/2


"""Return the (x, y) world coordinates of the center of cell (row, col).
    row 0 = bottom (smallest y), col 0 = left (smallest x).
"""

def cell_center(row: int, col: int, grid_size: int, cell_size: float) -> tuple[float, float]:
    """Return the (x, y) world coordinates of the center of cell (row, col).

    Decide (and write down here!) your convention, e.g.:
        
    Example: grid 8, cell 0.5 -> cell (0, 0) center is (-1.75, -1.75).
    You will reuse this function later when placing blocks at reset.
    """
    x = -arena_half_size(grid_size, cell_size) + cell_size/2 + col*cell_size
    y = -arena_half_size(grid_size, cell_size) + cell_size/2 + row*cell_size

    return (x,y)


# --------------------------------------------------------------------------- #
# XML sections (each returns a string with a piece of MJCF)
# --------------------------------------------------------------------------- #
def make_header(model_name: str) -> str:
    """Opening <mujoco> tag, <include> of the robot, <option>, <visual>, <statistic>.

    The robot file already sets <compiler meshdir="assets/"> and
    <option cone="elliptic" impratio="10">, so only the timestep is added here.
    """
    return f"""<mujoco model="{model_name}">
  <include file="{ROBOT_XML}"/>

  <option timestep="0.002"/>

  <statistic center="0 0 0" extent="3"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1" rangefinder="1 0.3 0.1 0.5"/>
    <global azimuth="135" elevation="-45" offwidth="1280" offheight="960"/>
    <scale framelength="0.3" framewidth="0.02"/>
  </visual>
"""


def make_assets(cell_size: float) -> str:
    """<asset> section: skybox, floor texture, materials for walls, blocks and goal.

    With texuniform="true" one checker square has side 1 / texrepeat meters
    (checked by rendering), so texrepeat = 1 / cell_size makes one floor square
    exactly one grid cell. The white edge lines are drawn every 2 squares.
    Squares match cells when the grid size is even (the plane is centered at 0, 0).
    """
    repeat = 1.0 / cell_size
    return f"""
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
             markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="{repeat:g} {repeat:g}" reflectance="0.1"/>
    <material name="wall" rgba="0.75 0.75 0.7 1"/>
    <material name="block" rgba="0.85 0.45 0.15 1"/>
    <material name="goal" rgba="0.1 0.9 0.2 0.6"/>
  </asset>
"""


def make_defaults(cell_size: float, wall_height: float) -> str:
    """<default> section with classes, so each block line stays short.

    Classes:
        - "wall":  box geom, wall material (size set per wall)
        - "block": box geom filling one cell, as tall as the walls
    Remember: box 'size' in MuJoCo is HALF of each dimension.
    """
    half_cell = cell_size / 2
    half_h = wall_height / 2
    return f"""
  <default>
    <default class="wall">
      <geom type="box" material="wall"/>
    </default>
    <default class="block">
      <geom type="box" size="{half_cell:g} {half_cell:g} {half_h:g}" material="block"/>
    </default>
  </default>
"""


def make_floor() -> str:
    """Floor plane geom and a light (inside <worldbody>)."""
    return """
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
"""


def _look_at_xyaxes(pos: tuple[float, float, float],
                    target: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> str:
    """xyaxes of a camera at `pos` looking at `target` (camera looks along its -z)."""
    f = [t - p for p, t in zip(pos, target)]
    norm = math.sqrt(sum(c * c for c in f))
    f = [c / norm for c in f]
    # right = f x up, with up = +z
    right = [f[1], -f[0], 0.0]
    norm = math.hypot(right[0], right[1])
    right = [c / norm for c in right]
    # up = right x f
    up = [right[1] * f[2] - right[2] * f[1],
          right[2] * f[0] - right[0] * f[2],
          right[0] * f[1] - right[1] * f[0]]
    return " ".join(f"{c:.4g}" for c in right + up)


def make_cameras(grid_size: int, cell_size: float) -> str:
    """Named cameras: "top" sees the whole arena from above, "persp" from a corner.

    The top camera height is chosen so the arena (plus walls and a 10% margin)
    fits in the default 45 deg vertical field of view.
    """
    half = arena_half_size(grid_size, cell_size) + WALL_THICKNESS
    top_z = 1.1 * half / math.tan(math.radians(45 / 2))
    persp = (1.3 * half, -1.3 * half, 1.2 * half)
    return f"""
    <camera name="top" pos="0 0 {top_z:.3g}" xyaxes="1 0 0 0 1 0"/>
    <camera name="persp" pos="{persp[0]:.3g} {persp[1]:.3g} {persp[2]:.3g}" xyaxes="{_look_at_xyaxes(persp)}"/>
"""


def make_outer_walls(grid_size: int, cell_size: float, wall_height: float) -> str:
    """Four static wall geoms around the arena (inside <worldbody>).

    The inner faces of the walls are exactly at +/- arena_half_size(), so the
    grid cells fit inside perfectly. North/south walls are longer by one
    thickness on each side to close the corners.
    """
    L = arena_half_size(grid_size, cell_size)
    t = WALL_THICKNESS / 2
    h = wall_height / 2
    c = L + t           # wall center line
    long = L + 2 * t    # half length of N/S walls (covers corners)
    walls = {
        "wall_n": ((0, c), (long, t)),
        "wall_s": ((0, -c), (long, t)),
        "wall_e": ((c, 0), (t, L)),
        "wall_w": ((-c, 0), (t, L)),
    }
    lines = [
        f'    <geom name="{name}" class="wall" pos="{x:g} {y:g} {h:g}" size="{sx:g} {sy:g} {h:g}"/>'
        for name, ((x, y), (sx, sy)) in walls.items()
    ]
    return "\n" + "\n".join(lines) + "\n"


def make_block_pool(n_blocks: int, wall_height: float) -> str:
    """The pool: n_blocks mocap bodies, each with one block geom.

    All blocks start at PARKING_POS. Mocap bodies are static for the physics,
    so parked blocks overlapping each other do not generate contacts.
    The env moves them into cells at reset via data.mocap_pos
    (z = wall_height / 2 puts a block on the floor).
    """
    x, y, z = PARKING_POS
    lines = [
        f'    <body name="block_{i:03d}" mocap="true" pos="{x:g} {y:g} {z:g}">\n'
        f'      <geom name="block_{i:03d}" class="block"/>\n'
        f'    </body>'
        for i in range(n_blocks)
    ]
    return "\n" + "\n".join(lines) + "\n"


def make_goal() -> str:
    """Goal marker: a mocap body with a flat, colored cylinder.

    It does not collide with anything (contype="0" conaffinity="0") and is
    1 cm high, far below the LiDAR plane, so the rays pass above it.
    """
    return f"""
    <body name="goal" mocap="true" pos="0 0 0">
      <geom name="goal" type="cylinder" size="{GOAL_RADIUS:g} 0.005" pos="0 0 0.005"
            material="goal" contype="0" conaffinity="0"/>
    </body>
"""


def make_footer() -> str:
    """Closing tags: </worldbody> and </mujoco> (build_scene opens <worldbody>)."""
    return """  </worldbody>
</mujoco>
"""


# --------------------------------------------------------------------------- #
# Putting it together
# --------------------------------------------------------------------------- #
def build_scene(args: argparse.Namespace) -> str:
    """Assemble all sections into one XML string.

    Order: header, assets, defaults, then <worldbody> with floor, cameras,
    walls, block pool and goal, then footer (which closes <worldbody>).
    The pool has one block per cell, so any map on this grid fits.
    """
    n_blocks = args.grid * args.grid
    parts = [
        make_header(f"grid_{args.grid}x{args.grid}"),
        make_assets(args.cell),
        make_defaults(args.cell, args.wall_height),
        "\n  <worldbody>",
        make_floor(),
        make_cameras(args.grid, args.cell),
        make_outer_walls(args.grid, args.cell, args.wall_height),
        make_block_pool(n_blocks, args.wall_height),
        make_goal(),
        make_footer(),
    ]
    header_comment = (
        f"<!-- GENERATED by make_scene.py: grid {args.grid}x{args.grid}, "
        f"cell {args.cell:g} m, wall height {args.wall_height:g} m. Do not edit by hand. -->\n"
    )
    return header_comment + "".join(parts)


def write_scene(xml: str, path: Path) -> None:
    """Write the XML string to a file (create parent folders if needed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(xml, encoding="utf-8")
    print(f"Saved {path}")


def validate_scene(path: Path) -> None:
    """Load the generated file with MuJoCo and print a short summary.

    Raises SystemExit with a readable message if the XML does not compile
    or the number of mocap bodies is not blocks + goal.
    """
    import mujoco  # imported here so the XML can be generated even without MuJoCo

    try:
        model = mujoco.MjModel.from_xml_path(str(path))
    except ValueError as err:
        raise SystemExit(f"MuJoCo could not load {path}:\n{err}")

    def names(obj_type, count):
        return [mujoco.mj_id2name(model, obj_type, i) or "" for i in range(count)]

    body_names = names(mujoco.mjtObj.mjOBJ_BODY, model.nbody)
    n_blocks = sum(n.startswith("block_") for n in body_names)
    has_goal = "goal" in body_names
    has_robot = "base" in body_names

    print(f"Loaded OK: {model.nbody} bodies, {model.ngeom} geoms, "
          f"{model.nmocap} mocap bodies ({n_blocks} blocks + {int(has_goal)} goal), "
          f"robot included: {has_robot}")

    if model.nmocap != n_blocks + 1 or not has_goal:
        raise SystemExit("Expected nmocap == number of blocks + 1 (goal)")
    if not has_robot:
        raise SystemExit(f"Robot body 'base' not found - check the include of {ROBOT_XML}")


def main() -> None:
    args = parse_args()
    xml = build_scene(args)
    out = Path(args.out)
    write_scene(xml, out)
    validate_scene(out)


if __name__ == "__main__":
    main()