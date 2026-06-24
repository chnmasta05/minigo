#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# ==========================================
# Configuration Variables
# ==========================================
BASE_DIR="/root/autodl-tmp/minigo_data"
MODELS_DIR="$BASE_DIR/saved_models"
SELFPLAY_BASE="$BASE_DIR/selfplay"
HOLDOUT_BASE="$BASE_DIR/holdout"
SGF_BASE="$BASE_DIR/sgf"
TF_LOGS="/root/tf-logs"

BOARD_SIZE=9
NUM_GAMES=3072

# We are starting by generating model 3, up to model 50.
START_GEN=39
END_GEN=50

echo "Starting Minigo pipeline from generation $START_GEN to $END_GEN..."

# ==========================================
# Training Loop
# ==========================================
for (( gen=$START_GEN; gen<=$END_GEN; gen++ ))
do
    # Calculate previous generation for the input model
    prev_gen=$((gen - 1))
    
    # Format numbers with leading zeros (e.g., 000002, 000003)
    PREV_MODEL_ID=$(printf "%06d" $prev_gen)
    CURR_MODEL_ID=$(printf "%06d" $gen)

    echo "=========================================================="
    echo " Generation $gen: Using $PREV_MODEL_ID to build $CURR_MODEL_ID"
    echo "=========================================================="

    # 1. Create isolated directories for the current generation's data
    # This prevents the train step from accidentally gobbling up old self-play data
    CURR_SELFPLAY_DIR="$SELFPLAY_BASE"
    CURR_HOLDOUT_DIR="$HOLDOUT_BASE"
    CURR_SGF_DIR="$SGF_BASE"
    
    mkdir -p "$CURR_SELFPLAY_DIR" "$CURR_HOLDOUT_DIR" "$CURR_SGF_DIR"

    # ---------------------------------------------------------
    # Step 1: Selfplay
    # ---------------------------------------------------------
    echo "[Step 1/3] Running selfplay with model $PREV_MODEL_ID.minigo..."
    START_TIME=$(date +%s.%N)
    bazel-bin/cc/concurrent_selfplay \
        --model="$MODELS_DIR/${PREV_MODEL_ID}.minigo" \
        --num_readouts=800 \
        --selfplay_threads=16 \
        --concurrent_games_per_thread=16 \
        --parallel_search=4 \
        --output_dir="$CURR_SELFPLAY_DIR" \
        --holdout_dir="$CURR_HOLDOUT_DIR" \
        --sgf_dir="$CURR_SGF_DIR" \
        --num_games=$NUM_GAMES \
        --dirichlet_alpha=0.15 \
        --noise_mix=0.4 \
        --policy_softmax_temp=0.4 \
        --disable_resign_pct=0.25 \
        --min_resign_threshold=-1.0 \
        --max_resign_threshold=-0.99 \
        --soft_pick_cutoff=20 \
        --soft_pick_uniform_mix=0.1
    END_TIME=$(date +%s.%N)

    t=$(python -c "import math; print(max(1, math.floor($END_TIME - $START_TIME + 0.5)))")
    NUM_GAMES=$(python -c "import math; print(math.ceil(1800 * $NUM_GAMES / $t))")
    echo "Selfplay took $t seconds. Adjusted NUM_GAMES to $NUM_GAMES for next iteration."

    # ---------------------------------------------------------
    # Step 2: Train
    # ---------------------------------------------------------
    echo "[Step 2/3] Training new model $CURR_MODEL_ID..."
    # Note: concurrent_selfplay creates a timestamped subdirectory (e.g., 2026-06-23-13). 
    # We grab the most recent 20 folders to use for training.

    RECENT_FOLDERS=$(ls -1td "$CURR_SELFPLAY_DIR"/*/ | head -n 20 | sed 's/$/\*/')
    
    set -f
    
    BOARD_SIZE=$BOARD_SIZE python train.py $RECENT_FOLDERS \
        --conv_width=128 \
        --trunk_layers=10 \
        --work_dir="$TF_LOGS" \
        --export_path="$MODELS_DIR/$CURR_MODEL_ID"

    set +f

    # ---------------------------------------------------------
    # Step 3: Freeze
    # ---------------------------------------------------------
    echo "[Step 3/3] Freezing graph for model $CURR_MODEL_ID..."
    BOARD_SIZE=$BOARD_SIZE python freeze_graph.py \
        --conv_width=128 \
        --trunk_layers=10 \
        --model_path="$MODELS_DIR/$CURR_MODEL_ID"

    echo "Generation $CURR_MODEL_ID successfully completed."
    echo "----------------------------------------------------------"
done

echo "Pipeline finished! $END_GEN generations completed."
echo "Final anticipated NUM_GAMES for next iteration: $NUM_GAMES"

