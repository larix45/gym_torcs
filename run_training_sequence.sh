#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <episode_batch_size>"
  echo "Example: $0 50"
  exit 1
fi

BATCH_SIZE="$1"
SCRIPT="torcs_jm_par_new.py"
STEER_MODEL="steer.npz"
THROTTLE_MODEL="throttle.npz"
GEAR_MODEL="gear.npz"
REWARD_FILE="current_reward.txt"
REPORT_FILE="training_sequence_report.txt"

rm -f "$REPORT_FILE" "$REWARD_FILE"
touch "$REWARD_FILE"

open_reward_monitor() {
  local monitor_cmd="cd \"$PWD\" && python3 -u reward_monitor.py \"$REPORT_FILE\""
  if command -v xfce4-terminal >/dev/null 2>&1; then
    xfce4-terminal --hold --title "TORCS Reward Monitor" --geometry=100x30 \
      --command "bash -lc '$monitor_cmd; echo "--- Reward monitor exited ---"; read -r'" >/dev/null 2>&1 &
    disown
  elif command -v xterm >/dev/null 2>&1; then
    xterm -hold -T "TORCS Reward Monitor" -geometry 100x30 \
      -e "bash -lc '$monitor_cmd; echo \"--- Reward monitor exited ---\"; read -r'" >/dev/null 2>&1 &
    disown
  else
    echo "Warning: no terminal emulator found for reward monitor. Starting plot monitor in the background."
    bash -lc "$monitor_cmd" >/dev/null 2>&1 &
    disown
  fi
}

open_reward_monitor

exec > >(tee -a "$REPORT_FILE") 2>&1

echo "Training sequence started: $(date)"
echo "Episode batch size: $BATCH_SIZE"
echo "Steer model: $STEER_MODEL"
echo "Throttle model: $THROTTLE_MODEL"
echo "Gear model: $GEAR_MODEL"
echo ""

run_stage() {
  local stage_name="$1"
  local extra_args="$2"
  local model_args="$3"
  local batch=0
  local total_episodes=0
  local lap_done=false

  echo "=== $stage_name ==="

  while [ "$lap_done" = false ]; do
    batch=$((batch + 1))
    total_episodes=$((total_episodes + BATCH_SIZE))
    echo "--- $stage_name: batch $batch ($BATCH_SIZE episodes) ---"
    : > "$REWARD_FILE"
    batch_output=$(mktemp)

    python3 -u "$SCRIPT" --train --train-fast --mode qlearn --episodes "$BATCH_SIZE" $extra_args $model_args 2>&1 \
      | tee -a "$REPORT_FILE" \
      | tee "$batch_output" \
      | while IFS= read -r line; do
          if [[ "$line" == REWARD:* ]]; then
            echo "$line" > "$REWARD_FILE"
          fi
        done

    if grep -q "LAP_COMPLETED" "$batch_output"; then
      lap_done=true
      echo "$stage_name: lap completion detected in batch $batch"
    else
      echo "$stage_name: no lap completion in batch $batch"
    fi

    rm -f "$batch_output"
  done

  echo "$stage_name completed after $batch batch(es), $total_episodes episode(s)."
  echo ""
}

echo "Stage 1: Train steer with fixed throttle and fixed gear"
run_stage "Steer training" "--train-steer --fixed-throttle --fixed-gear" "--load-model --save-model --steer-model-file $STEER_MODEL"

echo "Stage 2: Train throttle using trained steer model and fixed gear"
run_stage "Throttle training" "--train-throttle --load-model --steer-model-file $STEER_MODEL --fixed-gear" "--load-model --save-model --steer-model-file $STEER_MODEL --throttle-model-file $THROTTLE_MODEL"

echo "Stage 3: Train gear using trained steer and throttle models"
run_stage "Gear training" "--train-gear --load-model --steer-model-file $STEER_MODEL --throttle-model-file $THROTTLE_MODEL" "--load-model --save-model --steer-model-file $STEER_MODEL --throttle-model-file $THROTTLE_MODEL --gear-model-file $GEAR_MODEL"

echo "Training sequence completed: $(date)"

cat > "$REPORT_FILE" <<EOF
Training sequence report
========================

Date: $(date)
Episode batch size: $BATCH_SIZE

Models produced:
- Steer model: $STEER_MODEL
- Throttle model: $THROTTLE_MODEL
- Gear model: $GEAR_MODEL

Training logic:
- Each stage runs in batches of $BATCH_SIZE episodes.
- The stage continues until a lap completion is detected in the current batch.
- Steer stage uses fixed throttle and fixed gear.
- Throttle stage loads steer model and uses fixed gear.
- Gear stage loads steer and throttle models.

If any stage fails, inspect the output in this log and rerun the failed stage separately.
EOF

echo "Report written to $REPORT_FILE"
