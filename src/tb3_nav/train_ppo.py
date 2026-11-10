"""Train PPO (Stable-Baselines3) on NavEnv.

Every run gets its own folder:
    runs/<run_name>/
        tb/              TensorBoard logs
        checkpoints/     model saved every --save-freq steps
        best_model.zip   best model according to the evaluation
        final_model.zip  model after the last step
        config.json      all settings of the run (args, curriculum, PPO hyperparameters)
        plots/training.png, metrics.csv   training curves (plot_run.py, made at the end)

Curriculum: training starts on STAGES[0] and moves to the next stage when the success rate
of the last CURRICULUM_WINDOW training episodes reaches CURRICULUM_THRESHOLD.
Evaluation always runs on the last stage (the target task), so eval/* is comparable over time.

Usage (same as: python -m tb3_nav.train_ppo ...):
    tb3-train --timesteps 1000000 --n-envs 4
    tensorboard --logdir runs
"""
import argparse
import json
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from tb3_nav.make_scene import ROOT
from tb3_nav.nav_env import NavEnv
from tb3_nav.plot_run import plot_run

RUNS_DIR = ROOT / "runs"

# Curriculum stages: keyword arguments of NavEnv.set_difficulty(), from easy to the target task.
# n_obstacles is a (min, max) range drawn per episode; maze/rooms ignore it.
STAGES = [
    dict(mode="random", n_obstacles=(0, 3), min_dist_cells=2),
    dict(mode="random", n_obstacles=(3, 8), min_dist_cells=3),
    dict(mode="random", n_obstacles=(8, 15), min_dist_cells=5),
    # maze paths reach ~18 m (~70 s at full speed), so episodes need more steps
    dict(mode=("random", "rooms", "maze"), n_obstacles=(8, 15), min_dist_cells=5, max_steps=1500),
]
CURRICULUM_THRESHOLD = 0.7   # success rate needed to move to the next stage
CURRICULUM_WINDOW = 100      # number of recent training episodes the success rate is computed from

# PPO hyperparameters. Change one at a time and compare runs in TensorBoard
# (every run saves its settings to runs/<run_name>/config.json).
PPO_KWARGS = dict(
    learning_rate=3e-4,   # SB3 default; lower (1e-4) if the reward curve jumps around a lot
    n_steps=2048,         # steps collected PER ENV before each update (4 envs -> 8192 steps per update)
    batch_size=256,       # minibatch size, should divide n_steps * n_envs
    n_epochs=10,          # passes over the collected data per update
    # One env step = 0.05 s. The goal is often 200-400 steps away; with 0.99 a reward 400 steps
    # ahead is worth 0.99^400 = 2%, with 0.995 it is 13%, so the goal bonus still matters.
    gamma=0.995,
    gae_lambda=0.95,      # bias/variance trade-off of the advantage estimate (SB3 default)
    clip_range=0.2,       # max policy change per update (SB3 default)
    ent_coef=0.005,       # small bonus for random actions, so the policy doesn't settle on "stand still" too early
    vf_coef=0.5,
    max_grad_norm=0.5,
    # 41 inputs (36 LiDAR rays + 5) - bigger than the default 64x64 network
    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on NavEnv.")
    parser.add_argument("--timesteps", type=int, default=200_000, help="total env steps (default: 200000)")
    parser.add_argument("--n-envs", type=int, default=4, help="parallel environments (default: 4)")
    parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    parser.add_argument("--run-name", type=str, default=None, help="run folder name (default: date and time)")
    parser.add_argument("--out", type=Path, default=RUNS_DIR, help="folder for all runs (default: runs/)")
    parser.add_argument("--save-freq", type=int, default=50_000, help="checkpoint every N env steps (default: 50000)")
    # Evaluation runs on the hardest stage (up to 1500 steps per episode), so not too often
    parser.add_argument("--eval-freq", type=int, default=50_000, help="evaluate every N env steps (default: 50000)")
    parser.add_argument("--eval-episodes", type=int, default=10, help="episodes per evaluation (default: 10)")
    return parser.parse_args()


class CurriculumCallback(BaseCallback):
    """Moves all training envs to the next stage when the recent success rate is high enough.

    Logged to TensorBoard: curriculum/stage, curriculum/success_rate.
    """

    def __init__(self, stages: list[dict], threshold: float, window: int, verbose: int = 1):
        super().__init__(verbose)
        self.stages = stages
        self.threshold = threshold
        self.stage = 0
        self.successes = deque(maxlen=window)   # 1.0 / 0.0 per finished episode, oldest drop out

    def _on_step(self) -> bool:
        # A VecEnv resets finished envs by itself; info of the done step is still the last step's info
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                self.successes.append(float(info["is_success"]))

        if self.successes:
            rate = float(np.mean(self.successes))
            self.logger.record("curriculum/success_rate", rate)
            # Wait for a full window, so a few lucky episodes don't move the stage
            if (len(self.successes) == self.successes.maxlen and rate >= self.threshold
                    and self.stage < len(self.stages) - 1):
                self.stage += 1
                self.training_env.env_method("set_difficulty", **self.stages[self.stage])
                self.successes.clear()   # the new stage starts counting from zero
                if self.verbose:
                    print(f"\n[curriculum] success rate {rate:.0%} -> stage {self.stage}: {self.stages[self.stage]}")
        self.logger.record("curriculum/stage", self.stage)
        return True


def main() -> None:
    args = parse_args()
    run_name = args.run_name or datetime.now().strftime("ppo_%Y%m%d_%H%M%S")
    run_dir = args.out / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run folder: {run_dir}")

    # Save all settings of this run, so runs can be compared and repeated later
    config = dict(args=vars(args), stages=STAGES, curriculum_threshold=CURRICULUM_THRESHOLD,
                  curriculum_window=CURRICULUM_WINDOW, ppo=PPO_KWARGS)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))

    # Training envs: each in its own process (MuJoCo + Python LiDAR are CPU-bound).
    # make_vec_env wraps every env in a Monitor, so episode reward / length show up in the logs.
    vec_env_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    train_env = make_vec_env(NavEnv, n_envs=args.n_envs, seed=args.seed,
                             env_kwargs=STAGES[0], vec_env_cls=vec_env_cls)

    # Separate env for evaluation on the target task (last stage),
    # different seed so it sees other maps than training
    eval_env = make_vec_env(NavEnv, n_envs=1, seed=args.seed + 1000, env_kwargs=STAGES[-1])

    # Callbacks count calls of env.step() on the VecEnv, one call = n_envs env steps,
    # so the frequencies are divided by n_envs to stay in env steps.
    callbacks = [
        CheckpointCallback(save_freq=max(args.save_freq // args.n_envs, 1),
                           save_path=str(run_dir / "checkpoints"), name_prefix="ppo"),
        EvalCallback(eval_env, n_eval_episodes=args.eval_episodes,
                     eval_freq=max(args.eval_freq // args.n_envs, 1),
                     best_model_save_path=str(run_dir), log_path=str(run_dir / "eval"),
                     deterministic=True,
                     warn=False),   # train env is Subproc, eval env Dummy - that's fine here
        CurriculumCallback(STAGES, threshold=CURRICULUM_THRESHOLD, window=CURRICULUM_WINDOW),
    ]

    model = PPO(
        "MlpPolicy",
        train_env,
        seed=args.seed,
        verbose=1,
        tensorboard_log=str(run_dir / "tb"),
        **PPO_KWARGS,
    )

    model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=False)
    model.save(run_dir / "final_model")
    print(f"Saved: {run_dir / 'final_model.zip'}")

    train_env.close()
    eval_env.close()

    print(f"Saved: {plot_run(run_dir)}")


if __name__ == "__main__":   # required by SubprocVecEnv (new processes import this file)
    main()
