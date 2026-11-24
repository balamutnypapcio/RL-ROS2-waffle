#!/usr/bin/env bash
# Short fine-tuning runs from one base model, ONE PPO change per run, each followed by the benchmark.
#
#   bash scripts/finetune_sweep.sh                     list the variants
#   bash scripts/finetune_sweep.sh control ent0.1      run only these, one after another (in this order)
#   bash scripts/finetune_sweep.sh all                 run every variant, one after another
#
# Every variant gets its own run folder runs/<PREFIX>_<name>/ and benchmark tests/benchmark/<PREFIX>_<name>/.
# Variants whose benchmark already exists are skipped, so the script can be stopped and started again
# (a variant stopped DURING training must be deleted first: rm -r runs/<PREFIX>_<name>).
# Compare in tests/benchmark/comparison.png / comparison.csv, always against "<PREFIX>_control"
# (the same fine-tuning without any change), not against the base model.
set -u
cd "$(dirname "$0")/.."

BASE=runs/world2_s0/best_model.zip   # or: gh release download model/world2_s0-37M -D runs/world2_s0 -p model.zip
PREFIX=ft37
STEPS=1500000
SEED=0                               # same seed for all variants -> differences come from the parameters
COMMON=(--init-model "$BASE" --timesteps "$STEPS" --n-envs 8 --seed "$SEED" --stage -1
        --save-freq 500000 --eval-freq 500000 --eval-episodes 30)

# name | extra tb3-train arguments | description          (one change per variant, add your own lines)
VARIANTS=(
  "control||no change (reference for all others)"
  "lr1e-4|--learning-rate 1e-4|learning rate 3e-4 -> 1e-4 (smaller, more stable steps)"
  "ent0|--ent-coef 0.0|entropy bonus 0.005 -> 0 (no extra exploration)"
  "ent0.01|--ent-coef 0.01|entropy bonus 0.005 -> 0.01 (more exploration)"
  "clip0.1|--clip-range 0.1|clip range 0.2 -> 0.1 (smaller policy change per update)"
  "epochs5|--n-epochs 5|epochs per update 10 -> 5"
)

list_variants() {
  echo "Variants (base: $BASE, $STEPS steps each, ~35 min incl. benchmark):"
  for v in "${VARIANTS[@]}"; do
    IFS='|' read -r name _ desc <<< "$v"
    done_mark="   "
    [ -f "tests/benchmark/${PREFIX}_$name/summary.json" ] && done_mark="[x]"
    printf "  %s %-10s %s\n" "$done_mark" "$name" "$desc"
  done
  echo "[x] = already done.  Usage: bash $0 <name> [<name> ...] | all"
}

if [ $# -eq 0 ]; then
  list_variants
  exit 0
fi

# which variants to run
if [ "$1" = "all" ]; then
  selected=()
  for v in "${VARIANTS[@]}"; do selected+=("${v%%|*}"); done
else
  selected=("$@")
fi

# check all names first, so a typo doesn't stop the queue in the middle of the night
for name in "${selected[@]}"; do
  found=0
  for v in "${VARIANTS[@]}"; do [ "${v%%|*}" = "$name" ] && found=1; done
  if [ $found -eq 0 ]; then
    echo "Unknown variant: $name"
    list_variants
    exit 1
  fi
done

PY=.venv/bin/python
mkdir -p runs/_logs
for name in "${selected[@]}"; do
  for v in "${VARIANTS[@]}"; do [ "${v%%|*}" = "$name" ] && IFS='|' read -r _ extra _ <<< "$v"; done
  run="${PREFIX}_${name}"
  if [ -f "tests/benchmark/$run/summary.json" ]; then
    echo "=== $run: already done, skipped"
    continue
  fi
  echo "=== $run: train (${extra:-no change}) $(date +%T), log: runs/_logs/$run.log"
  # shellcheck disable=SC2086
  if ! $PY -m tb3_nav.train_ppo "${COMMON[@]}" --run-name "$run" $extra > "runs/_logs/$run.log" 2>&1; then
    echo "!!! $run: training failed, see runs/_logs/$run.log"
    continue
  fi
  echo "=== $run: benchmark $(date +%T)"
  $PY -m tb3_nav.benchmark "runs/$run/final_model.zip" --name "$run" 2>&1 | grep -E "goal|WARN|Error"
done
echo "=== finished $(date +%T)"
