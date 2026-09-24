#!/bin/bash
export PYTHONIOENCODING=utf-8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
N_SAMPLES=${1:-20}
LOG_DIR="logs"
MASTER_LOG="$LOG_DIR/master_eval.log"
mkdir -p "$LOG_DIR"
echo "======================================================" > "$MASTER_LOG"
echo "Starting Evaluation Pipeline with n=$N_SAMPLES" >> "$MASTER_LOG"
echo "Date: $(date)" >> "$MASTER_LOG"
echo "======================================================" >> "$MASTER_LOG"
DATASETS=("hotpotqa" "nq" "tqa" "musique" "popqa") # "2wiki" "asqa" commented out
TOTAL_DATASETS=${#DATASETS[@]}
for i in "${!DATASETS[@]}"; do
    DATASET="${DATASETS[$i]}"
    
    OUT_FOLDER_NAME=$DATASET
    if [ "$DATASET" = "hotpotqa" ]; then
        OUT_FOLDER_NAME="hotpot_qa"
    fi
    
    FINAL_CSV="src/eval_results/${OUT_FOLDER_NAME}/final_benchmark_results.csv"
    if [ -s "$FINAL_CSV" ]; then
        echo "⏭️ SKIPPING $DATASET: final_benchmark_results.csv already exists and is not empty." | tee -a "$MASTER_LOG"
        continue
    fi

    SCRIPT_LOG="$LOG_DIR/${DATASET}.log"
    MODULE_NAME="src.eval.eval_${DATASET}"
    
    echo -e "\n------------------------------------------------------" | tee -a "$MASTER_LOG"
    echo "Running evaluation for: $DATASET" | tee -a "$MASTER_LOG"
    echo "Command: python -m $MODULE_NAME -n $N_SAMPLES" | tee -a "$MASTER_LOG"
    
    python -m "$MODULE_NAME" -n "$N_SAMPLES" 2>&1 | tee "$SCRIPT_LOG" | tee -a "$MASTER_LOG"
    EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $EXIT_CODE -ne 0 ]; then
        echo "ERROR: $DATASET evaluation failed with exit code $EXIT_CODE." | tee -a "$MASTER_LOG"
        echo "Moving gracefully to the next dataset..." | tee -a "$MASTER_LOG"
    else
        echo "SUCCESS: $DATASET evaluation completed." | tee -a "$MASTER_LOG"
    fi

    if [ $i -lt $((TOTAL_DATASETS - 1)) ]; then
        echo "Sleeping for 1 minute to let the GPU cool down and clear VRAM..." | tee -a "$MASTER_LOG"
        sleep 60
    fi
done

echo -e "\n======================================================" >> "$MASTER_LOG"
echo "All evaluations in the pipeline have finished." >> "$MASTER_LOG"