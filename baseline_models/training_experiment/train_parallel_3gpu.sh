#!/bin/bash
# Parallel Training on 3 H200 GPUs
# Runs 3 models simultaneously, then the 4th one

set -e

echo "========================================================================"
echo "PARALLEL TRAINING ON 3 H200 GPUs"
echo "========================================================================"
echo ""
echo "Strategy:"
echo "  Batch 1 (parallel): Train-1, Train-2, Train-3 on GPU 0,1,2"
echo "  Batch 2 (single):   Train-4 on GPU 0"
echo ""
echo "Expected time: ~3-4 hours (vs 8-10 hours sequential)"
echo ""

# Configuration
BASE_MODEL="../ckpts/web_click/epoch_9/model.pth"
TRAINING_DIR="./training_splits_advisor"
OUTPUT_DIR="./trained_models_advisor"

# Check if training splits exist
if [ ! -d "$TRAINING_DIR" ]; then
    echo "❌ ERROR: Training splits not found at $TRAINING_DIR"
    echo "   Run Step 1 first: python create_training_splits.py ..."
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "========================================================================"
echo "BATCH 1: Training 3 models in parallel"
echo "========================================================================"
echo ""

# Train Model 1 on GPU 0 (background)
echo "Starting Train-1 (All Recoverable) on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python train_advisor_experiments.py \
    --base_model "$BASE_MODEL" \
    --training_dir "$TRAINING_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --epochs 3 \
    --learning_rate 2e-5 \
    --single_split train1_all_recoverable \
    > "$OUTPUT_DIR/train1_log.txt" 2>&1 &
PID1=$!
echo "  PID: $PID1 (check log: $OUTPUT_DIR/train1_log.txt)"

sleep 5

# Train Model 2 on GPU 1 (background)
echo "Starting Train-2 (Middle U) on GPU 1..."
CUDA_VISIBLE_DEVICES=1 python train_advisor_experiments.py \
    --base_model "$BASE_MODEL" \
    --training_dir "$TRAINING_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --epochs 3 \
    --learning_rate 2e-5 \
    --single_split train2_middle_u_recoverable \
    > "$OUTPUT_DIR/train2_log.txt" 2>&1 &
PID2=$!
echo "  PID: $PID2 (check log: $OUTPUT_DIR/train2_log.txt)"

sleep 5

# Train Model 3 on GPU 2 (background)
echo "Starting Train-3 (All States) on GPU 2..."
CUDA_VISIBLE_DEVICES=2 python train_advisor_experiments.py \
    --base_model "$BASE_MODEL" \
    --training_dir "$TRAINING_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --epochs 3 \
    --learning_rate 2e-5 \
    --single_split train3_all_states \
    > "$OUTPUT_DIR/train3_log.txt" 2>&1 &
PID3=$!
echo "  PID: $PID3 (check log: $OUTPUT_DIR/train3_log.txt)"

echo ""
echo "All 3 processes started. Waiting for completion..."
echo ""
echo "Monitor progress:"
echo "  tail -f $OUTPUT_DIR/train1_log.txt"
echo "  tail -f $OUTPUT_DIR/train2_log.txt"
echo "  tail -f $OUTPUT_DIR/train3_log.txt"
echo ""

# Wait for all 3 to complete
wait $PID1
echo "✓ Train-1 complete!"

wait $PID2
echo "✓ Train-2 complete!"

wait $PID3
echo "✓ Train-3 complete!"

echo ""
echo "========================================================================"
echo "BATCH 2: Training final model"
echo "========================================================================"
echo ""

# Train Model 4 on GPU 0
echo "Starting Train-4 (High/Low U) on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python train_advisor_experiments_simple.py \
    --base_model "$BASE_MODEL" \
    --training_dir "$TRAINING_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --epochs 3 \
    --learning_rate 2e-5 \
    --single_split train4_high_low_u_recoverable

echo ""
echo "✓ Train-4 complete!"

echo ""
echo "========================================================================"
echo "✓ ALL 4 MODELS TRAINED SUCCESSFULLY"
echo "========================================================================"
echo ""
echo "Models saved in: $OUTPUT_DIR"
echo ""
echo "Next step: Evaluate all models"
echo "  python evaluate_advisor_experiments.py \\"
echo "    --models_dir $OUTPUT_DIR \\"
echo "    --test_tasks 200 \\"
echo "    --max_steps 15 \\"
echo "    --output_dir ./eval_advisor_results"
echo ""