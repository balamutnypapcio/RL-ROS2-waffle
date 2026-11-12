"""Standard benchmark: every model drives on the SAME fixed maps, so results are comparable.

Usage:
    tb3-bench runs/stall_seed0/model_3M.zip --name stall_s0_3M
    tb3-bench --compare-only          (only redraw the comparison of all saved results)

Results of one model go to tests/benchmark/<name>/:
    episodes.csv      one row per episode (map type, seed, outcome, steps, ...)
    summary.json      rates per map type + which model / commit / date
    outcomes.png      outcome per map type, and success with the goal visible vs hidden
    trajectories.png  robot paths on the first maps of every type
Comparison of ALL saved results: tests/benchmark/comparison.png and comparison.csv.
"""

import argparse
import csv
import json
import os
import subprocess
import time
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # draw to files only, no window
import matplotlib.pyplot as plt
import numpy as np

from tb3_nav.layout_generator import WALL, bfs_distances
from tb3_nav.make_scene import ROOT
from tb3_nav.nav_env import NavEnv

BENCH_DIR = ROOT / "tests" / "benchmark"
SEED_OFFSET = 20_000    # map seeds 20000, 20001, ... (the same maps as all earlier tests)
# Map type -> NavEnv settings. Don't change them, or old results stop being comparable.
TESTS = {
    "random": dict(n_obstacles=(8, 15), min_dist_cells=5, max_steps=1000),
    "rooms":  dict(n_obstacles=(8, 15), min_dist_cells=5, max_steps=1000),
    "maze":   dict(n_obstacles=(8, 15), min_dist_cells=5, max_steps=1500),
}
OUTCOMES = ("goal", "collision", "timeout")
COLORS = {"goal": "#2a9d8f", "collision": "#e76f51", "timeout": "#e9c46a"}
N_PATHS = 6             # paths drawn per map type in trajectories.png
ROBOT_HALF_WIDTH = 0.17  # [m] used to decide if the straight line to the goal is blocked


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the standard benchmark for one model.")
    parser.add_argument("model", type=Path, nargs="?", help="model .zip file")
    parser.add_argument("--name", type=str, help="result folder name (default: model file name)")
    parser.add_argument("--episodes", type=int, default=200, help="maps per map type (default: 200)")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="parallel processes")
    parser.add_argument("--out", type=Path, default=BENCH_DIR, help="results folder (default: tests/benchmark)")
    parser.add_argument("--compare-only", action="store_true", help="only redraw comparison.png")
    return parser.parse_args()


# ----------------------------- driving (runs in worker processes) -------------------------
_model = None   # every worker process loads its own copy of the model ...
_envs = {}      # ... and keeps one environment per map type


def _init_worker(model_path: str) -> None:
    global _model
    import torch
    torch.set_num_threads(1)    # many processes, so one thread each
    from stable_baselines3 import PPO
    _model = PPO.load(model_path, device="cpu")


def goal_hidden(start: np.ndarray, goal: np.ndarray, grid: np.ndarray, cell: float) -> bool:
    """True if a robot-wide strip from start to goal crosses a wall cell (no line of sight)."""
    half = grid.shape[0] * cell / 2
    d = goal - start
    length = np.linalg.norm(d)
    normal = np.array([-d[1], d[0]]) / length
    for t in np.linspace(0, 1, int(length / 0.02) + 2):
        for offset in (-ROBOT_HALF_WIDTH, 0.0, ROBOT_HALF_WIDTH):
            x, y = start + t * d + offset * normal
            col, row = int((x + half) // cell), int((y + half) // cell)
            if 0 <= row < grid.shape[0] and 0 <= col < grid.shape[1] and grid[row, col] == WALL:
                return True
    return False


def run_episode(task: tuple[str, int]) -> dict:
    mode, episode = task
    if mode not in _envs:
        _envs[mode] = NavEnv(mode=mode, **TESTS[mode])
    env = _envs[mode]
    n_inputs = _model.observation_space.shape[0]   # older models have fewer inputs (cut the rest)

    obs, _ = env.reset(seed=SEED_OFFSET + episode)
    start, goal = env._robot_pose()[0].copy(), env.goal_xy.copy()
    path = [start]
    while True:
        action, _ = _model.predict(obs[:n_inputs], deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        path.append(env._robot_pose()[0].copy())
        if terminated or truncated:
            break

    path = np.array(path)
    grid = env.layout.grid
    return dict(
        mode=mode,
        episode=episode,
        outcome=info["outcome"],
        steps=info["steps"],
        goal_hidden=int(goal_hidden(start, goal, grid, env.placer.cell_size)),
        path_cells=int(bfs_distances(grid, env.layout.start)[env.layout.goal]),  # shortest path in cells
        straight_m=round(float(np.linalg.norm(goal - start)), 3),
        driven_m=round(float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()), 3),
        end_goal_dist_m=round(float(np.linalg.norm(goal - path[-1])), 3),
        min_obstacle_dist_m=round(float(info["min_obstacle_dist"]), 3),
        # only for drawing, not saved to the csv
        _path=path if episode < N_PATHS else None,
        _grid=grid,
        _goal=goal,
        _cell=env.placer.cell_size,
    )


# ----------------------------- results ----------------------------------------------------
def summarize(rows: list[dict]) -> dict:
    summary = {}
    for mode in TESTS:
        r = [x for x in rows if x["mode"] == mode]
        outcome = np.array([x["outcome"] for x in r])
        hidden = np.array([x["goal_hidden"] for x in r], dtype=bool)
        goal = outcome == "goal"
        summary[mode] = dict(
            episodes=len(r),
            **{f"{o}_rate": round(float(np.mean(outcome == o)), 3) for o in OUTCOMES},
            goal_rate_visible=round(float(goal[~hidden].mean()), 3) if (~hidden).any() else None,
            goal_rate_hidden=round(float(goal[hidden].mean()), 3) if hidden.any() else None,
            n_visible=int((~hidden).sum()),
            n_hidden=int(hidden.sum()),
            median_steps_to_goal=float(np.median([x["steps"] for x in r if x["outcome"] == "goal"])) if goal.any() else None,
            median_min_obstacle_dist_m=round(float(np.median([x["min_obstacle_dist_m"] for x in r])), 3),
        )
    return summary


def plot_outcomes(summary: dict, name: str, path: Path) -> None:
    modes = list(TESTS)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    # left: how the episodes ended, per map type
    left = np.zeros(len(modes))
    for o in OUTCOMES:
        vals = np.array([summary[m][f"{o}_rate"] for m in modes]) * 100
        ax1.barh(modes, vals, left=left, color=COLORS[o], label=o)
        for i, v in enumerate(vals):
            if v >= 6:
                ax1.text(left[i] + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=9)
        left += vals
    ax1.set_xlim(0, 100)
    ax1.invert_yaxis()
    ax1.set_xlabel("% of episodes")
    ax1.set_title("How the episodes ended")
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, -0.35), ncol=3, frameon=False)

    # right: success when the goal is visible from the start vs hidden behind walls
    x = np.arange(len(modes))
    for k, (key, label, color) in enumerate([("visible", "goal visible", "#457b9d"),
                                             ("hidden", "goal hidden", "#1d3557")]):
        vals = [100 * (summary[m][f"goal_rate_{key}"] or 0) for m in modes]
        bars = ax2.bar(x + (k - 0.5) * 0.38, vals, 0.38, color=color, label=label)
        for b, m in zip(bars, modes):
            ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                     f"n={summary[m][f'n_{key}']}", ha="center", fontsize=8)
    ax2.set_xticks(x, modes)
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("goal reached [%]")
    ax2.set_title("Success: goal visible vs hidden at start")
    ax2.legend(loc="lower center", bbox_to_anchor=(0.5, -0.35), ncol=2, frameon=False)

    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_paths(rows: list[dict], name: str, path: Path) -> None:
    modes = list(TESTS)
    fig, axes = plt.subplots(len(modes), N_PATHS, figsize=(2.2 * N_PATHS, 2.3 * len(modes)))
    for i, mode in enumerate(modes):
        for j in range(N_PATHS):
            ax = axes[i, j]
            ax.set_xticks([]), ax.set_yticks([])
            r = next((x for x in rows if x["mode"] == mode and x["episode"] == j), None)
            if r is None:
                continue
            half = r["_grid"].shape[0] * r["_cell"] / 2
            ax.imshow(r["_grid"], origin="lower", cmap="Greys", vmin=0, vmax=1.4,
                      extent=(-half, half, -half, half))
            p = r["_path"]
            ax.plot(p[:, 0], p[:, 1], color=COLORS[r["outcome"]], lw=1.5)
            ax.plot(*p[0], "o", color="#264653", ms=5)
            ax.plot(*r["_goal"], "*", color="#2a9d8f", ms=11, mec="black", mew=0.5)
            ax.set_title(f"{mode} #{j}: {r['outcome']}", fontsize=8)
    fig.suptitle(f"{name}  (dot = start, star = goal, line colour = outcome)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def compare_all(out: Path) -> None:
    """Draw every saved result next to each other: comparison.png + comparison.csv."""
    results = []
    for f in sorted(out.glob("*/summary.json")):
        results.append(json.loads(f.read_text()))
    if not results:
        return
    results.sort(key=lambda s: s["info"]["date"])
    names = [s["info"]["name"] for s in results]

    with open(out / "comparison.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "date", "mode", *[f"{o}_rate" for o in OUTCOMES], "goal_rate_visible", "goal_rate_hidden"])
        for s in results:
            for m in TESTS:
                r = s["results"][m]
                w.writerow([s["info"]["name"], s["info"]["date"], m, *[r[f"{o}_rate"] for o in OUTCOMES],
                            r["goal_rate_visible"], r["goal_rate_hidden"]])

    fig, axes = plt.subplots(1, len(TESTS), figsize=(4 + 1.6 * len(names), 4.5), sharey=True)
    y = np.arange(len(names))
    for ax, mode in zip(axes, TESTS):
        left = np.zeros(len(names))
        for o in OUTCOMES:
            vals = np.array([100 * s["results"][mode][f"{o}_rate"] for s in results])
            ax.barh(y, vals, left=left, color=COLORS[o], label=o)
            for i, v in enumerate(vals):
                if v >= 8:
                    ax.text(left[i] + v / 2, i, f"{v:.0f}", ha="center", va="center", fontsize=8)
            left += vals
        ax.set_title(mode)
        ax.set_xlim(0, 100)
        ax.set_xlabel("% of episodes")
    axes[0].set_yticks(y, names)
    axes[0].invert_yaxis()
    axes[-1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Benchmark: all models on the same maps")
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=120)
    plt.close(fig)
    print(f"comparison of {len(names)} results -> {out / 'comparison.png'}")


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True).stdout.strip()
    except OSError:
        return "unknown"


def main() -> None:
    args = parse_args()
    if args.compare_only:
        compare_all(args.out)
        return
    if args.model is None:
        raise SystemExit("give a model .zip (or --compare-only)")

    name = args.name or args.model.stem
    run_dir = args.out / name
    run_dir.mkdir(parents=True, exist_ok=True)

    # long episodes (rooms, maze) first, so all workers finish at about the same time
    tasks = [(m, ep) for m in reversed(list(TESTS)) for ep in range(args.episodes)]
    t0 = time.time()
    rows = []
    with Pool(args.workers, initializer=_init_worker, initargs=(str(args.model),)) as pool:
        for i, row in enumerate(pool.imap_unordered(run_episode, tasks), start=1):
            rows.append(row)
            if i % 50 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} episodes, {time.time() - t0:.0f} s", flush=True)
    rows.sort(key=lambda r: (list(TESTS).index(r["mode"]), r["episode"]))

    with open(run_dir / "episodes.csv", "w", newline="") as f:
        fields = [k for k in rows[0] if not k.startswith("_")]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    summary = summarize(rows)
    info = dict(name=name, model=str(args.model), date=datetime.now().isoformat(timespec="seconds"),
                git_commit=git_commit(), episodes_per_mode=args.episodes, seed_offset=SEED_OFFSET,
                tests=TESTS, duration_s=round(time.time() - t0))
    (run_dir / "summary.json").write_text(json.dumps(dict(info=info, results=summary), indent=2))
    plot_outcomes(summary, name, run_dir / "outcomes.png")
    plot_paths(rows, name, run_dir / "trajectories.png")

    for m, r in summary.items():
        print(f"{m:7s} goal {r['goal_rate']:.0%}  collision {r['collision_rate']:.0%}  "
              f"timeout {r['timeout_rate']:.0%}  | goal hidden: {r['goal_rate_hidden']:.0%} of {r['n_hidden']}")
    print(f"results -> {run_dir}")
    compare_all(args.out)


if __name__ == "__main__":
    main()
