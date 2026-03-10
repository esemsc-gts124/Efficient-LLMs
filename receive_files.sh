#!/usr/bin/env bash
set -u

# Command to run
cmd=(
  ./croc
  --yes
  --relay "147.224.199.121:9009"
  --pass "efficientllms"
)

# Environment for the command
export CROC_SECRET="efficientllms"

# Backoff settings
initial_backoff=30
max_backoff=3600
backoff=$initial_backoff

# Stop cleanly on Ctrl+C / SIGTERM
stop_requested=0
trap 'stop_requested=1' INT TERM

while true; do
  if (( stop_requested )); then
    echo "Stop requested, exiting."
    exit 0
  fi

  echo "[$(date '+%F %T')] Starting: CROC_SECRET=\"$CROC_SECRET\" ${cmd[*]}"
  "${cmd[@]}"
  exit_code=$?

  if (( stop_requested )); then
    echo "Stop requested, exiting."
    exit 0
  fi

  if [[ $exit_code -eq 0 ]]; then
    echo "[$(date '+%F %T')] Command exited successfully. Resetting backoff."
    backoff=$initial_backoff
    exit 0
  fi

  echo "[$(date '+%F %T')] Command failed with exit code $exit_code. Restarting in ${backoff}s..."
  sleep "$backoff"

  # Exponential backoff, capped
  backoff=$(( backoff * 2 ))
  if (( backoff > max_backoff )); then
    backoff=$max_backoff
  fi
done
