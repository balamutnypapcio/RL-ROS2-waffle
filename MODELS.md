# Trained models

**Best model: [`model/world2_s0-37M`](https://github.com/balamutnypapcio/RL-ROS2-waffle/releases/tag/model/world2_s0-37M)**

```bash
gh release download model/world2_s0-37M -D runs/world2_s0 -p model.zip
tb3-play runs/world2_s0/model.zip
```

Model files are not in git. They are attached to [GitHub Releases](https://github.com/balamutnypapcio/RL-ROS2-waffle/releases) (tag `model/<run>-<steps>`, pointing at the commit the model was trained on), together with the training `config.json` and the benchmark result.

## Benchmark (current scene)

Scene 9×9 grid, 0.7 m cells, physics 0.005 s × 10. 200 fixed maps per type (map seeds 20000–20199), so every model drives on exactly the same maps.
Results: [tests/benchmark/](tests/benchmark/) (`comparison.png`, `comparison.csv`). Goal / collision / timeout in %, sorted by the mean goal rate.

| Model | Release | random | rooms | maze | mean goal |
|---|---|---|---|---|---|
| **world2_s0 37M** | [model/world2_s0-37M](https://github.com/balamutnypapcio/RL-ROS2-waffle/releases/tag/model/world2_s0-37M) | **90** / 3 / 7 | **77** / 3.5 / 19.5 | **33** / 28 / 39 | **67** |
| pathreward_s0_cont 17M | – (1M-step smoke test of continued training) | 88 / 10 / 2 | 73.5 / 9.5 / 17 | 12.5 / 83.5 / 4 | 58 |
| jakub_ppo 2M | [model/jakub_ppo-2M](https://github.com/balamutnypapcio/RL-ROS2-waffle/releases/tag/model/jakub_ppo-2M) | 86.5 / 10 / 3.5 | 49.5 / 30 / 20.5 | 11.5 / 72.5 / 16 | 49 |
| stall_s0 3M | [model/stall_s0-3M](https://github.com/balamutnypapcio/RL-ROS2-waffle/releases/tag/model/stall_s0-3M) | 80.5 / 12.5 / 7 | 45 / 12.5 / 42.5 | 4 / 96 / 0 | 43 |

Notes:
- Only world2_s0 was trained in the current world. The others were trained on the old 10×10 / 0.5 m scene, so their rows show how well they transfer.
- Mean goal is the plain mean of the three map types.
- With 200 maps per type a single rate has a standard error of up to ±3.5 percentage points, so differences below ~10 points between two models can be noise.
- Older results (mixed scenes, not comparable) are in [tests/benchmark/_archive/](tests/benchmark/_archive/).

## All models

| Model | Run folder | Commit | Steps | What is new | Status |
|---|---|---|---|---|---|
| jakub_ppo 2M | `runs/ppo_20261103_225610` | `6b1a565` | 2M | First PPO: 41 inputs, straight-line progress reward, LiDAR collision, success-rate curriculum, old world | released (baseline) |
| stall_s0 3M | `runs/stall_seed0` | `534ad1e` | 3M | Contact collisions, proximity penalty, record progress, `gap_from_best` + `stall` inputs (43) | released |
| fixed_s0 8M | `runs/fixed_s0` | `8c327cc` | 8M | Curriculum on a fixed step schedule | **model file missing (only on Maks's machine)** |
| pathreward_s0 16M | `runs/pathreward_s0` | `7cad22a` | 16M | Reward from BFS path distance instead of straight line | **missing**: parent of world2_s0, should be released |
| pathreward_maze_s0 30M | `runs/pathreward_maze_s0` | `d90d68e` | 30M | + wide maze stage (old world) | **missing** |
| world2_s0 37M | `runs/world2_s0` | `5214072` | 37M | Continued from pathreward_s0 16M on the 9×9 / 0.7 m world | **released, BEST** |
| world2_s0 21M / 26M / 36M / 46M | `runs/world2_s0` | `5214072` | – | Checkpoints at the end of each curriculum stage, final model | **missing** |

Runs that didn't produce a kept model are in `runs/_archive/` (logs and plots only).

## Adding a model

1. Benchmark it: `tb3-bench runs/<run>/<model>.zip --name <run>_<steps>`
2. Publish a release on the commit it was trained on:
   ```bash
   mkdir -p /tmp/rel && cp runs/<run>/<model>.zip /tmp/rel/model.zip   # asset is always called model.zip
   cp tests/benchmark/<name>/summary.json /tmp/rel/benchmark_summary.json
   gh release create model/<run>-<steps> --target <full commit sha> --title "<run> <steps>" \
       --notes "..." --latest=false \
       /tmp/rel/model.zip runs/<run>/config.json /tmp/rel/benchmark_summary.json
   ```
   Use `--latest` (instead of `--latest=false`) only for a new best model.
3. Add a row to both tables above and update **Best model** if needed.
