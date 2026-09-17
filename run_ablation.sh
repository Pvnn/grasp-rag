#!/bin/bash

# Force Python to output UTF-8
export PYTHONIOENCODING="utf-8"

LOG_DIR="logs"
ABLATION_LOG="$LOG_DIR/master_ablation.log"

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

echo "======================================================" > "$ABLATION_LOG"
echo "Starting GRASP Ablation Study Pipeline" >> "$ABLATION_LOG"
echo "Date: $(date)" >> "$ABLATION_LOG"
echo "======================================================" >> "$ABLATION_LOG"

# Default to 300 samples if no argument is passed
NUM_SAMPLES=${1:-300}

echo -e "\n\033[0;36m[INFO] Starting Ablation Runner with N=$NUM_SAMPLES\033[0m"
echo -e "\033[0;33m[INFO] This script supports incremental saving. If stopped, it will resume where it left off.\033[0m\n"

python scripts/run_ablation.py -n "$NUM_SAMPLES" 2>&1 | tee -a "$ABLATION_LOG"

# Check exit status
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo -e "\n\033[0;31m[ERROR] Ablation runner failed.\033[0m"
    echo "[ERROR] Ablation runner failed." >> "$ABLATION_LOG"
else
    echo -e "\n\033[0;32m[SUCCESS] Ablation pipeline completed successfully.\033[0m"
    echo "[SUCCESS] Ablation pipeline completed successfully." >> "$ABLATION_LOG"
fi
