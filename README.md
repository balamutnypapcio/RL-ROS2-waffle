# RL-ROS2-waffle

Goal navigation for the **TurtleBot3 Waffle Pi** on randomly generated maps, learned with RL (PPO, Stable-Baselines3) in the **MuJoCo** simulator.

The robot has a 2D LiDAR (36 rays, 1.5 m range) and gets a new map in every episode: a grid with obstacles, a start and a goal.

## Installation

Requires Python ≥ 3.10.

```bash
git clone git@github.com:balamutnypapcio/RL-ROS2-waffle.git
cd RL-ROS2-waffle
python3 -m venv .venv
source .venv/bin/activate

# optional: CPU-only PyTorch (~200 MB instead of several GB with CUDA).
# Training runs on the CPU anyway, the simulation is the bottleneck.
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -e ".[dev]"
```

`pip install -e .` installs the dependencies and the `tb3_nav` package in editable mode, so changes in `src/tb3_nav/` take effect without reinstalling. It also adds the `tb3-*` commands below. `[dev]` adds `pytest`.

## Usage

All commands work from any directory (with `.venv` active). Each one is also available as `python -m tb3_nav.<module>`.

| command | what it does |
|---|---|
| `tb3-make-scene --grid 10 --cell 0.5` | generates the scene `models/scene_grid.xml` (10×10 grid, 0.5 m cells) |
| `tb3-layouts --grid 10 --mode maze --count 3` | prints example maps in the terminal |
| `tb3-show --mode random --obstacles 12` | MuJoCo viewer: map, robot, LiDAR rays |
| `tb3-train --timesteps 1000000 --n-envs 4` | PPO training |
| `tb3-plot [runs/<name>]` | (re)draws the training plots of a run (default: the newest one) |
| `tb3-play [runs/<name>]` | MuJoCo viewer: a trained model drives (default: the newest run) |
| `pytest` | tests |

The scene is already in the repo (10×10, 0.5 m). Run `tb3-make-scene` only after changing the grid or cell size.

### Viewer (`tb3-show`)

Click the viewer window first. Keys:
- **N** — new map
- **arrows** — drive (up/down: forward/back, left/right: turn)
- **Space** — stop

Red rays hit an obstacle, green rays found nothing within range. LiDAR readings can be logged with `--log-mode terminal` or `--log-mode file`.

### Training (`tb3-train`)

Every run is saved in its own folder `runs/<date_time>/`:
- `config.json` — all settings (arguments, curriculum, PPO hyperparameters)
- `best_model.zip` — best model according to the evaluation
- `final_model.zip` — model after the last step
- `plots/training.png` — training curves, drawn at the end of training (see below)
- `metrics.csv` — the same data as a table (`metric, step, value`)
- `checkpoints/` — model every `--save-freq` steps
- `tb/` — TensorBoard logs

`plots/training.png` shows the success rate, episode reward and length (training vs. evaluation), the curriculum stage, the value-function fit and the policy's exploration. Dashed vertical lines mark curriculum stage changes. For interactive plots use `tensorboard --logdir runs`.

**Curriculum:** training starts on easy maps (0–3 obstacles, goal close by). When the robot reaches the goal in 70% of the last 100 episodes, it moves to a harder stage. The last stage mixes the `random`, `rooms` and `maze` map types. Evaluation always runs on the last stage. After every stage change `rollout/ep_rew_mean` drops because the task got harder, not because the model got worse. Stages and hyperparameters are at the top of `src/tb3_nav/train_ppo.py` (`STAGES` and `PPO_KWARGS`).

### Watching a trained model (`tb3-play`)

```bash
tb3-play                                    # best_model.zip of the newest run, hardest maps
tb3-play runs/<name>                        # best_model.zip of that run
tb3-play runs/<name>/final_model.zip        # any saved model, also checkpoints/ppo_*.zip
tb3-play runs/<name> --stage 1              # maps of curriculum stage 1 (0 = easiest)
tb3-play runs/<name> --mode maze            # only one map type: random, rooms or maze
tb3-play runs/<name> --speed 3              # 3x faster than real time
```

Episodes run one after another on new maps. The terminal shows how each one ended (goal, collision or timeout) and the running totals. Keys in the viewer: **N** skips to the next episode, **Space** pauses. By default the model picks its most likely action, the same as in the evaluation. `--stochastic` samples actions as in training. Other options: `--episodes N` (stop after N episodes) and `--seed` (repeat the same maps).

Note: `best_model.zip` is chosen from only `--eval-episodes` evaluation episodes, so `final_model.zip` can be just as good. Compare both.

### Results in git

Run folders are committed to git, except `checkpoints/` (dozens of ~2 MB files per run). One run is about 4 MB: two models, plots, logs and config. Git keeps every committed file in its history forever, so **commit only the runs worth keeping** and delete failed experiments first:

```bash
rm -r runs/<failed_run>
git add runs/<good_run>
git commit -m "Training run: 2M steps, curriculum reached stage 3"
```

## Structure

```
models/                     MuJoCo models (TurtleBot3 robot, generated scene)
src/tb3_nav/
    make_scene.py           scene generator: floor, walls, block pool, goal, robot
    layout_generator.py     map generator: wall grid, start, goal (pure NumPy)
    show_layout.py          MuJoCo viewer + LayoutPlacer (puts a map into the simulation)
    lidar.py                simulated 2D LiDAR (mujoco.mj_ray)
    controller.py           control: [v, omega] -> wheel velocities, keyboard
    logger.py               LiDAR reading logger
    geometry.py             distance and angle to the goal in the robot frame
    nav_env.py              Gymnasium environment (observation, action, reward)
    train_ppo.py            PPO training with curriculum
    plot_run.py             training plots from the TensorBoard logs
    play_model.py           viewer: a trained model drives
runs/                       training results (models, plots, logs), one folder per run
tests/                      tests (pytest)
```

**Environment (`NavEnv`):**
- **action:** `[v, omega]` in [-1, 1], scaled to 0.26 m/s and 1.82 rad/s;
- **observation (41 values):** 36 LiDAR readings / range, robot v and ω (from the wheels), distance to goal / arena diagonal, sin and cos of the angle to the goal;
- **reward:** 10 × progress towards the goal [m], −0.01 per step, +15 for the goal, −15 for a collision;
- **episode end:** goal reached (< 0.2 m), collision (LiDAR < 0.2 m) or step limit (500, 1500 in the last stage).

## Docker (ROS 2 Humble)

```bash
xhost +local:            # lets the container open a window (MuJoCo viewer)
docker compose up -d --build
docker compose exec ros_dev bash
# inside the container (the repo is mounted at /ros2_ws):
pip install -e ".[dev]"
```

The image has Python 3.10. Python packages are installed inside the container by hand (last line above).

## Known issues

- **`LLVM ERROR: inconsistency in registered CommandLine options`** when training starts: TensorFlow is installed in the environment and clashes with PyTorch. It isn't needed, so uninstall it: `pip uninstall tensorflow keras`.
- **Import error for `tb3_nav`:** the package isn't installed in the active environment. Run `pip install -e .` in the repo root.
