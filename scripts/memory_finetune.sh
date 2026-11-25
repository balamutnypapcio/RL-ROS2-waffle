#!/usr/bin/env bash
# Memory (visit map) experiment: the best model gets the visit-map inputs and is trained further,
# next to a control run with exactly the same settings but WITHOUT the visit map.
# Every checkpoint (every SAVE steps) of both runs is benchmarked, so we see WHEN memory starts to help.
#
#   bash scripts/memory_finetune.sh              both runs one after another: visit5, then control (~4 h),
#                                                each benchmarked at the end
#   bash scripts/memory_finetune.sh visit5       only one run (~2 h)
#   bash scripts/memory_finetune.sh list         list the runs and what is already done
#
# How the memory works: src/tb3_nav/nav_env.py (NavEnv docstring, visit_map).
# How the model gets the new inputs without losing what it learned: src/tb3_nav/expand_model.py.
# The script can be stopped and started again: finished trainings and benchmarks are skipped
# (a run stopped DURING training must be deleted first: rm -r runs/<PREFIX>_<name>).
set -u
cd "$(dirname "$0")/.."

BASE=runs/world2_s0/best_model.zip   # or: gh release download model/world2_s0-37M -D runs/world2_s0 -p model.zip
PREFIX=mem37
STEPS=6000000                        # memory needs time: the new weights start at 0
SAVE=1500000                         # checkpoint every SAVE steps
BENCH_CHECKPOINTS=0                  # 0 = benchmark only the final model (4 min), 1 = every checkpoint (4 x 4 min)
SEED=0
# 8 envs = like the base model was trained (16384 steps per PPO update); NOT the same as the
# ft37 sweep (4 envs), so compare only visit5 vs control of THIS script
COMMON=(--init-model "$BASE" --timesteps "$STEPS" --n-envs 8 --seed "$SEED" --stage -1
        --save-freq "$SAVE" --eval-freq 1000000 --eval-episodes 30)

# name | extra tb3-train arguments | description
RUNS=(
  "visit5|--visit-map 5|memory: 5x5 visit map around the robot (43 -> 68 inputs)"
  "control|--visit-map 0|same training without memory (reference)"
)

list_runs() {
  echo "Runs (base: $BASE, $STEPS steps each, benchmark every $SAVE steps, ~2 h per run):"
  for r in "${RUNS[@]}"; do
    IFS='|' read -r name _ desc <<< "$r"
    done_mark="   "
    [ -f "runs/${PREFIX}_$name/final_model.zip" ] && done_mark="[x]"
    printf "  %s %-8s %s\n" "$done_mark" "$name" "$desc"
  done
  echo "[x] = trained.  Usage: bash $0 [all | list | <name> ...]   (no argument = all)"
}

if [ "${1:-}" = "list" ]; then
  list_runs
  exit 0
fi
if [ $# -eq 0 ] || [ "$1" = "all" ]; then
  selected=()
  for r in "${RUNS[@]}"; do selected+=("${r%%|*}"); done
else
  selected=("$@")
fi
for name in "${selected[@]}"; do
  found=0
  for r in "${RUNS[@]}"; do [ "${r%%|*}" = "$name" ] && found=1; done
  if [ $found -eq 0 ]; then
    echo "Unknown run: $name"
    list_runs
    exit 1
  fi
done

PY=.venv/bin/python
mkdir -p runs/_logs

# bench <model.zip> <benchmark name>: skipped if already done
bench() {
  if [ -f "tests/benchmark/$2/summary.json" ]; then
    echo "    $2: already benchmarked"
    return
  fi
  echo "=== $2: benchmark $(date +%T)"
  $PY -m tb3_nav.benchmark "$1" --name "$2" 2>&1 | grep -E "goal|WARN|Error"
}

for name in "${selected[@]}"; do
  for r in "${RUNS[@]}"; do [ "${r%%|*}" = "$name" ] && IFS='|' read -r _ extra _ <<< "$r"; done
  run="${PREFIX}_${name}"

  if [ ! -f "runs/$run/final_model.zip" ]; then
    echo "=== $run: train ($extra) $(date +%T), log: runs/_logs/$run.log"
    # shellcheck disable=SC2086
    if ! $PY -m tb3_nav.train_ppo "${COMMON[@]}" --run-name "$run" $extra > "runs/_logs/$run.log" 2>&1; then
      echo "!!! $run: training failed, see runs/_logs/$run.log"
      continue
    fi
  fi

  # benchmark, named by the steps trained in THIS run: <run>_1.5M, ..., <run>_6.0M (= final model)
  if [ "$BENCH_CHECKPOINTS" = "1" ]; then
    start=$($PY -c "import json; print(json.load(open('runs/$run/config.json'))['start_timesteps'])")
    for ckpt in $(ls runs/$run/checkpoints/ppo_*_steps.zip | sort -t_ -k2 -n); do
      steps=$(basename "$ckpt" | sed -E 's/ppo_([0-9]+)_steps.zip/\1/')
      label=$($PY -c "print(f'{($steps - $start) / 1e6:.1f}M')")
      bench "$ckpt" "${run}_$label"
    done
  else
    bench "runs/$run/final_model.zip" "${run}_$($PY -c "print(f'{$STEPS / 1e6:.1f}M')")"
  fi
done
echo "=== finished $(date +%T)"
echo "Compare: tests/benchmark/comparison.png, always ${PREFIX}_visit5_<steps> against ${PREFIX}_control_<steps>"
