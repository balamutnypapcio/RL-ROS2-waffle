"""Watch a trained PPO model drive in the MuJoCo viewer.

Runs NavEnv episodes one after another with the model's actions, draws the LiDAR rays
and prints how every episode ended (goal / collision / timeout).

Usage (same as: python -m tb3_nav.play_model ...):
    tb3-play                                      best_model.zip of the newest run, last curriculum stage
    tb3-play runs/<run_name>                      best_model.zip of that run
    tb3-play runs/<run_name>/final_model.zip      any saved model (also checkpoints/*.zip)
    tb3-play --stage 1                            maps of curriculum stage 1 (0 = easiest)
    tb3-play --mode maze                          only one map type
    tb3-play --speed 3                            3x faster than real time

Keys (click the viewer window first):
    N       - skip to the next episode
    Space   - pause / resume
"""
import argparse
import time
from collections import Counter
from pathlib import Path

import mujoco
import mujoco.viewer
from stable_baselines3 import PPO

from tb3_nav.layout_generator import MODES
from tb3_nav.nav_env import FRAME_SKIP, NavEnv
from tb3_nav.show_layout import draw_rays
from tb3_nav.train_ppo import RUNS_DIR, STAGES

KEY_N = 78       # GLFW key codes
KEY_SPACE = 32

END_PAUSE = 1.0  # seconds to show the final state of an episode before the next one


def find_model(path: Path | None) -> Path:
    """Model file from a .zip path, a run folder (its best_model.zip) or None (newest run)."""
    if path is None:
        runs = sorted((p for p in RUNS_DIR.glob("*") if (p / "best_model.zip").exists()),
                      key=lambda p: p.stat().st_mtime)
        if not runs:
            raise SystemExit(f"No run with best_model.zip in {RUNS_DIR}")
        path = runs[-1]
    if path.is_dir():
        path = path / "best_model.zip"
    if not path.exists():
        raise SystemExit(f"Model not found: {path}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch a trained model in the MuJoCo viewer.")
    parser.add_argument("model", type=Path, nargs="?", default=None,
                        help="run folder or model .zip (default: best_model.zip of the newest run)")
    parser.add_argument("--stage", type=int, default=len(STAGES) - 1,
                        help=f"curriculum stage of the maps, 0..{len(STAGES) - 1} (default: last)")
    parser.add_argument("--mode", choices=MODES, default=None,
                        help="only this map type (default: the stage's map types)")
    parser.add_argument("--episodes", type=int, default=0, help="stop after N episodes (default: 0 = never)")
    parser.add_argument("--seed", type=int, default=None, help="random seed of the maps (default: different every run)")
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed, 1 = real time (default: 1)")
    parser.add_argument("--stochastic", action="store_true",
                        help="sample actions like in training (default: deterministic, like the evaluation)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.stage < len(STAGES):
        raise SystemExit(f"--stage must be 0..{len(STAGES) - 1}")

    model_path = find_model(args.model)
    model = PPO.load(model_path, device="cpu")
    print(f"Model: {model_path}")

    stage = dict(STAGES[args.stage])
    if args.mode is not None:
        stage["mode"] = args.mode
    env = NavEnv(**stage)
    print(f"Stage {args.stage}: {stage}")

    # The key callback runs in the viewer's thread, so it only sets flags for the main loop
    keys = {"skip": False, "paused": False}

    def on_key(key: int) -> None:
        if key == KEY_N:
            keys["skip"] = True
        elif key == KEY_SPACE:
            keys["paused"] = not keys["paused"]

    step_time = FRAME_SKIP * env.model.opt.timestep / args.speed   # wall time of one env step
    outcomes = Counter()
    episode = 0
    obs, _ = env.reset(seed=args.seed)

    # The viewer shows env.model / env.data directly; reset() and step() change them in place
    with mujoco.viewer.launch_passive(env.model, env.data, key_callback=on_key) as viewer:
        print("\nPress N in the viewer to skip an episode, Space to pause.")
        episode_reward = 0.0
        while viewer.is_running():
            t0 = time.time()
            if keys["paused"]:
                viewer.sync()
                time.sleep(0.05)
                continue

            action, _ = model.predict(obs, deterministic=not args.stochastic)
            with viewer.lock():
                obs, reward, terminated, truncated, info = env.step(action)
                draw_rays(viewer, env.lidar.scan(env.data, return_points=True)[1])
            episode_reward += reward
            viewer.sync()

            if terminated or truncated or keys["skip"]:
                if info["is_success"]:
                    outcome = "goal"
                elif terminated:
                    outcome = "collision"
                elif truncated:
                    outcome = "timeout"
                else:
                    outcome = "skipped"
                outcomes[outcome] += 1
                episode += 1
                finished = episode - outcomes["skipped"]
                rate = outcomes["goal"] / finished if finished else 0.0
                print(f"episode {episode:3d}: {outcome:9s} {info['steps']:4d} steps, reward {episode_reward:6.1f}"
                      f"   | goal {outcomes['goal']}/{finished} ({rate:.0%}), "
                      f"collision {outcomes['collision']}, timeout {outcomes['timeout']}")

                if args.episodes and episode >= args.episodes:
                    break
                if not keys["skip"]:
                    time.sleep(END_PAUSE)   # show where the episode ended
                keys["skip"] = False
                obs, _ = env.reset()
                episode_reward = 0.0
                continue

            time.sleep(max(0.0, step_time - (time.time() - t0)))

    # Closing the viewer from code (--episodes) and exiting right away segfaults in MuJoCo 3.x:
    # the viewer thread is still shutting down. A short wait lets it finish.
    time.sleep(0.5)


if __name__ == "__main__":
    main()
