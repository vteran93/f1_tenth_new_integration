#!/usr/bin/env bash
# Launch the kata-curriculum PPO and SAC experiments concurrently (one Ray instance each)
# and leave them running detached. Logs go to examples/multiagent/logs/.
#
#   ./run_kata_overnight.sh            # start both
#   ./run_kata_overnight.sh status     # show progress (stage, steps, lap success, eval)
#   ./run_kata_overnight.sh stop       # stop both trainings
#
# Resume after a stop/crash: just run it again (tune resume="AUTO+ERRORED" picks up the
# latest checkpoint and the curriculum stage is restored from curriculum_state.json).
set -euo pipefail
cd "$(dirname "$0")"
PY=../../venv/bin/python
CFG=configs/experiments_kata.yaml
# Override with e.g.: KATA_EXPS="kata_v2_PPO_shared_ProgressTimePenalty kata_v2_SAC_shared_ProgressTimePenalty" ./run_kata_overnight.sh start
read -r -a EXPS <<< "${KATA_EXPS:-kata_PPO_shared_ProgressTimePenalty kata_SAC_shared_ProgressTimePenalty}"
mkdir -p logs

status() {
  $PY kata_status.py models_kata "${EXPS[@]}"
}

case "${1:-start}" in
  start)
    for e in "${EXPS[@]}"; do
      # anchored to the python process so a shell whose command line merely mentions the name does not match
      if pgrep -f "^[^ ]*python[^ ]* run.py .*--experiment $e\$" >/dev/null; then echo "$e already running"; continue; fi
      log=logs/${e}_$(date +%Y%m%d_%H%M%S).log
      nohup $PY run.py --config $CFG train --experiment "$e" > "$log" 2>&1 &
      echo "started $e (pid $!) -> $log"
      sleep 20   # let the first Ray instance come up before starting the second
    done
    ;;
  status) status ;;
  stop)
    for e in "${EXPS[@]}"; do pkill -f "^[^ ]*python[^ ]* run.py .*--experiment $e\$" && echo "stopped $e" || echo "$e not running"; done
    sleep 3; pkill -f "^ray::" || true
    ;;
  *) echo "usage: $0 [start|status|stop]"; exit 1 ;;
esac
