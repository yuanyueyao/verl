#!/bin/bash
# GPU monitor: wait until all 8 GPUs are free, then launch SFT experiments
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/_logs"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/$(date +%Y%m%d_%H%M%S)_sft_monitor.log"

echo "============================================================" | tee "$LOG"
echo " GPU Monitor: waiting for 8 GPUs to be free, then run SFT"
echo " 开始监控 @ $(date)"
echo "============================================================" | tee -a "$LOG"

check_free_gpus() {
    python3 -c "
import subprocess
out = subprocess.run(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader'],
                     capture_output=True, text=True).stdout.strip().split('\n')
free = [l.split(',')[0].strip() for l in out if int(l.split(',')[1].strip().split()[0]) < 1000]
print(len(free))
"
}

while true; do
    N_FREE=$(check_free_gpus)
    TIMESTAMP=$(date +"%H:%M:%S")
    echo "[${TIMESTAMP}] ${N_FREE}/8 GPUs free" | tee -a "$LOG"

    if [ "$N_FREE" -ge 4 ]; then
        echo "" | tee -a "$LOG"
        echo ">>> GPUs available (${N_FREE}/8), launching SFT experiments @ $(date)" | tee -a "$LOG"

        # Run SFT experiments
        bash "${SCRIPT_DIR}/run_sft_all.sh" 2>&1 | tee -a "$LOG"
        EXIT_CODE=$?

        echo "<<< SFT experiments done exit=${EXIT_CODE} @ $(date)" | tee -a "$LOG"
        echo "=== DONE SFT exit=${EXIT_CODE} ===" | tee -a "$LOG"
        exit $EXIT_CODE
    fi

    sleep 300  # check every 5 minutes
done
