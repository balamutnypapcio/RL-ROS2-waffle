"""Plot the training curves of a run from its TensorBoard logs.

Writes into the run folder:
    plots/training.png   success rate, reward, episode length, curriculum, exploration
    metrics.csv          the same data as a table (metric, step, value)

Called automatically at the end of train_ppo.py. To (re)plot a run by hand:
    tb3-plot runs/<run_name>     (same as: python -m tb3_nav.plot_run ...)
    tb3-plot                     (the newest run in runs/)
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # files only, no window (works on machines without a display)
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from tb3_nav.make_scene import ROOT

RUNS_DIR = ROOT / "runs"

# Colors (light surface): two categorical slots + text / grid tokens
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
TRAIN_COLOR = "#2a78d6"   # slot 1: training episodes (current curriculum stage)
EVAL_COLOR = "#eb6834"    # slot 2: evaluation (always the last, hardest stage)
STAGE_LINE = "#a8a79f"    # recessive marker of curriculum stage changes

TRAIN_LABEL = "training (current stage)"
EVAL_LABEL = "evaluation (hardest stage)"

# One panel per measure (one y-axis each): (title, [(tag, label, color)], y-format)
PANELS = [
    ("Success rate", [("rollout/success_rate", TRAIN_LABEL, TRAIN_COLOR),
                      ("eval/success_rate", EVAL_LABEL, EVAL_COLOR)], "percent"),
    ("Episode reward", [("rollout/ep_rew_mean", TRAIN_LABEL, TRAIN_COLOR),
                        ("eval/mean_reward", EVAL_LABEL, EVAL_COLOR)], "number"),
    ("Episode length [steps]", [("rollout/ep_len_mean", TRAIN_LABEL, TRAIN_COLOR),
                                ("eval/mean_ep_length", EVAL_LABEL, EVAL_COLOR)], "number"),
    ("Curriculum stage", [("curriculum/stage", None, TRAIN_COLOR)], "stage"),
    # How well the critic predicts returns: close to 1 is good, near 0 or negative = it learned nothing
    ("Value function fit (explained variance)", [("train/explained_variance", None, TRAIN_COLOR)], "number"),
    ("Policy std (exploration)", [("train/std", None, TRAIN_COLOR)], "number"),
]


def load_scalars(run_dir: Path) -> dict[str, list[tuple[int, float]]]:
    """All scalars from the run's TensorBoard logs: {tag: [(step, value), ...]} sorted by step."""
    scalars: dict[str, list[tuple[int, float]]] = {}
    for events in sorted((run_dir / "tb").glob("*/events.out.tfevents.*")):
        acc = EventAccumulator(str(events), size_guidance={"scalars": 0})   # 0 = keep all points
        acc.Reload()
        for tag in acc.Tags()["scalars"]:
            scalars.setdefault(tag, []).extend((e.step, e.value) for e in acc.Scalars(tag))
    if not scalars:
        raise SystemExit(f"No TensorBoard logs in {run_dir / 'tb'}")
    return {tag: sorted(points) for tag, points in scalars.items()}


def save_csv(scalars: dict[str, list[tuple[int, float]]], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "step", "value"])
        for tag in sorted(scalars):
            writer.writerows((tag, step, value) for step, value in scalars[tag])


def format_steps(x: float, _pos=None) -> str:
    """Env steps on the x axis: 500k, 1.5M."""
    if abs(x) >= 1e6:
        return f"{x / 1e6:g}M"
    if abs(x) >= 1e3:
        return f"{x / 1e3:g}k"
    return f"{x:g}"


def stage_changes(scalars: dict[str, list[tuple[int, float]]]) -> list[int]:
    """Steps at which the curriculum moved to the next stage."""
    points = scalars.get("curriculum/stage", [])
    return [step for (_, prev), (step, value) in zip(points, points[1:]) if value > prev]


def load_config(run_dir: Path) -> dict:
    """config.json saved by train_ppo.py (empty dict for runs without it)."""
    path = run_dir / "config.json"
    return json.loads(path.read_text()) if path.exists() else {}


def style_axis(ax, title: str, y_format: str, n_stages: int) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=TEXT_PRIMARY, pad=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)
    ax.xaxis.set_major_formatter(FuncFormatter(format_steps))
    if y_format == "percent":
        ax.set_ylim(-0.03, 1.03)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    elif y_format == "stage":
        ax.set_ylim(-0.3, n_stages - 0.7)
        ax.set_yticks(range(n_stages))


def end_label(value: float, y_format: str) -> str:
    if y_format == "percent":
        return f"{value:.0%}"
    if y_format == "stage":
        return f"{value:.0f}"
    return f"{value:.3g}"


def plot_run(run_dir: Path) -> Path:
    """Save plots/training.png and metrics.csv for one run, return the png path."""
    scalars = load_scalars(run_dir)
    save_csv(scalars, run_dir / "metrics.csv")
    changes = stage_changes(scalars)
    config = load_config(run_dir)
    n_stages = len(config.get("stages", [])) or 1
    threshold = config.get("curriculum_threshold")
    last_step = max(points[-1][0] for points in scalars.values())

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), facecolor=SURFACE, sharex=True)
    for ax, (title, series, y_format) in zip(axes.flat, PANELS):
        style_axis(ax, title, y_format, n_stages)
        ax.tick_params(labelbottom=True)   # sharex hides the top row's labels, keep them
        for step in changes:   # same marker in every panel: reward drops right after a stage change
            ax.axvline(step, color=STAGE_LINE, linewidth=1, linestyle=(0, (3, 3)), zorder=1)
        if title == "Success rate" and threshold is not None:
            ax.axhline(threshold, color=STAGE_LINE, linewidth=1, zorder=1)
            ax.annotate(f"next stage at {threshold:.0%}", (0, threshold), xytext=(2, 4),
                        textcoords="offset points", fontsize=8, color=TEXT_SECONDARY)

        for tag, label, color in series:
            points = scalars.get(tag)
            if not points:
                continue
            steps, values = zip(*points)
            is_eval = tag.startswith("eval/")
            drawstyle = "steps-post" if y_format == "stage" else "default"
            ax.plot(steps, values, color=color, linewidth=2, label=label, drawstyle=drawstyle, zorder=3,
                    marker="o" if is_eval else None, markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
            # Selective direct label: only the last value of each series
            ax.annotate(end_label(values[-1], y_format), (steps[-1], values[-1]),
                        xytext=(6, 0), textcoords="offset points", va="center",
                        fontsize=9, color=TEXT_SECONDARY)
    axes.flat[0].set_xlim(0, last_step * 1.08)   # shared by all panels, room for the end labels

    # One legend for the whole figure (two series, same colors in every panel)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if changes:
        handles.append(plt.Line2D([], [], color=STAGE_LINE, linewidth=1, linestyle=(0, (3, 3))))
        labels.append("curriculum stage change")
    fig.legend(handles, labels, loc="upper right", ncols=len(labels), frameon=False,
               fontsize=10, labelcolor=TEXT_SECONDARY, bbox_to_anchor=(0.99, 0.985))
    fig.suptitle(f"Training: {run_dir.name}", x=0.01, y=0.985, ha="left",
                 fontsize=13, color=TEXT_PRIMARY, fontweight="bold")
    fig.supxlabel("environment steps", fontsize=10, color=TEXT_SECONDARY)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.5, w_pad=2.0)

    out = run_dir / "plots" / "training.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training curves of a run.")
    parser.add_argument("run_dir", type=Path, nargs="?", default=None,
                        help="run folder (default: the newest folder in runs/)")
    args = parser.parse_args()

    run_dir = args.run_dir
    if run_dir is None:
        runs = sorted((p for p in RUNS_DIR.glob("*") if p.is_dir()), key=lambda p: p.stat().st_mtime)
        if not runs:
            raise SystemExit(f"No runs in {RUNS_DIR}")
        run_dir = runs[-1]
    print(f"Saved: {plot_run(run_dir)}")


if __name__ == "__main__":
    main()
