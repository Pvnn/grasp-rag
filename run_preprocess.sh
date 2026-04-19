#!/bin/bash

# Force Python to use UTF-8
export PYTHONIOENCODING=utf-8

LOG_DIR="logs"
MASTER_LOG="$LOG_DIR/master_preprocess.log"

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Initialize master log
echo "======================================================" > "$MASTER_LOG"
echo "Starting Preprocessing Pipeline" >> "$MASTER_LOG"
echo "Date: $(date)" >> "$MASTER_LOG"
echo "======================================================" >> "$MASTER_LOG"

# Exact order of preprocessing scripts
SCRIPTS=(
    "scripts/preprocess_2wiki.py"
    "scripts/preprocess_tqa.py"
    "scripts/preprocess_hotpotqa.py"
    "scripts/preprocess_asqa.py"
    "scripts/preprocess_nq.py"
)
TOTAL_SCRIPTS=${#SCRIPTS[@]}

for i in "${!SCRIPTS[@]}"; do
    SCRIPT_PATH="${SCRIPTS[$i]}"
    
    # Extract just the filename (e.g., preprocess_2wiki) for the log name
    SCRIPT_NAME=$(basename "$SCRIPT_PATH" .py)
    SCRIPT_LOG="$LOG_DIR/${SCRIPT_NAME}.log"
    
    echo -e "\n------------------------------------------------------" | tee -a "$MASTER_LOG"
    echo "[INFO] Running: $SCRIPT_PATH" | tee -a "$MASTER_LOG"
    
    # Execute python, capture stdout and stderr, write to individual log AND master log
    python "$SCRIPT_PATH" 2>&1 | tee "$SCRIPT_LOG" | tee -a "$MASTER_LOG"
    
    # Capture the exit code of the python command
    EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $EXIT_CODE -ne 0 ]; then
        echo "[ERROR] $SCRIPT_NAME failed with exit code $EXIT_CODE." | tee -a "$MASTER_LOG"
        echo "[INFO] Moving gracefully to the next script..." | tee -a "$MASTER_LOG"
    else
        echo "[SUCCESS] $SCRIPT_NAME completed successfully." | tee -a "$MASTER_LOG"
    fi
    
    # Sleep for 1 minute (60s) after every 2 scripts, unless it's the last script
    SCRIPT_NUM=$((i + 1))
    if [ $((SCRIPT_NUM % 2)) -eq 0 ] && [ $SCRIPT_NUM -ne $TOTAL_SCRIPTS ]; then
        echo "[WAIT] Sleeping for 60 seconds to let the system cool down..." | tee -a "$MASTER_LOG"
        sleep 60
    fi
done

echo -e "\n======================================================" | tee -a "$MASTER_LOG"
echo "[INFO] All preprocessing scripts have finished." | tee -a "$MASTER_LOG"