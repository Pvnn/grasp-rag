#!/bin/bash

# Force Python to use UTF-8
export PYTHONIOENCODING=utf-8

# Default to n=20 if no argument is provided
N_SAMPLES=${1:-20}
LOG_DIR="logs"
MASTER_LOG="$LOG_DIR/master_eval.log"

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Initialize master log
echo "======================================================" > "$MASTER_LOG"
echo "Starting Evaluation Pipeline with n=$N_SAMPLES" >> "$MASTER_LOG"
echo "Date: $(date)" >> "$MASTER_LOG"
echo "======================================================" >> "$MASTER_LOG"

# Exact order of datasets
DATASETS=("2wiki" "tqa" "hotpotqa" "asqa" "nq")
TOTAL_DATASETS=${#DATASETS[@]}

for i in "${!DATASETS[@]}"; do
    DATASET="${DATASETS[$i]}"
    SCRIPT_LOG="$LOG_DIR/${DATASET}.log"
    MODULE_NAME="src.eval.eval_${DATASET}"
    
    echo -e "\n------------------------------------------------------" | tee -a "$MASTER_LOG"
    echo "Running evaluation for: $DATASET" | tee -a "$MASTER_LOG"
    echo "Command: python -m $MODULE_NAME -n $N_SAMPLES" | tee -a "$MASTER_LOG"
    
    # Execute python, capture stdout and stderr, write to individual log AND master log
    python -m "$MODULE_NAME" -n "$N_SAMPLES" 2>&1 | tee "$SCRIPT_LOG" | tee -a "$MASTER_LOG"
    
    # Capture the exit code of the python command (not the tee command)
    EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $EXIT_CODE -ne 0 ]; then
        echo "ERROR: $DATASET evaluation failed with exit code $EXIT_CODE." | tee -a "$MASTER_LOG"
        echo "Moving gracefully to the next dataset..." | tee -a "$MASTER_LOG"
    else
        echo "SUCCESS: $DATASET evaluation completed." | tee -a "$MASTER_LOG"
    fi
    
    # Sleep for 2 minutes (120s) to let the GPU rest, unless it's the last dataset
    if [ $((i + 1)) -ne $TOTAL_DATASETS ]; then
        echo "Sleeping for 2 minutes to let GPU cool down..." | tee -a "$MASTER_LOG"
        sleep 120
    fi
done

echo -e "\n======================================================" | tee -a "$MASTER_LOG"
echo "All evaluations in the pipeline have finished." | tee -a "$MASTER_LOG"