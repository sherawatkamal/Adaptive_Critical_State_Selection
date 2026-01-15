#!/bin/bash
# Master Script: Advisor's Data Selection Experiment
# 
# Tests: Does Middle U data actually improve learning?
# 
# Pipeline:
# 1. Create 4 training splits from EEF results
# 2. Train 4 models (one per split)
# 3. Evaluate all 4 on same test set
# 4. Compare results

set -e  # Exit on error

echo "========================================================================"
echo "ADVISOR'S DATA SELECTION EXPERIMENT"
echo "========================================================================"
echo ""
echo "Research Question:"
echo "  Does training on Middle U data (curated) beat All Recoverable (volume)?"
echo ""
echo "Experiments:"
echo "  Train-1: All Recoverable (baseline - max volume)"
echo "  Train-2: Middle U Recoverable (quality > quantity hypothesis)"
echo "  Train-3: All States (does noise hurt?)"
echo "  Train-4: High/Low U Recoverable (control)"
echo ""

# Configuration - UPDATED PATHS
EEF_DIR="./eef_3000_stratified"
TRAINING_DIR="./training_splits_advisor"
MODELS_DIR="./trained_models_advisor"
EVAL_DIR="./eval_advisor_results"
BASE_MODEL="../ckpts/web_click/epoch_9/model.pth"  # ← One level up

# Check if EEF results exist
if [ ! -d "$EEF_DIR" ]; then
    echo "❌ ERROR: EEF directory not found: $EEF_DIR"
    echo "   Please run EEF pipeline first to generate data"
    exit 1
fi

# Check if base model exists
if [ ! -f "$BASE_MODEL" ]; then
    echo "❌ ERROR: Base model not found: $BASE_MODEL"
    echo "   Expected location: ../ckpts/web_click/epoch_9/model.pth"
    echo "   Current directory: $(pwd)"
    exit 1
fi

echo "✓ Found EEF data directory: $EEF_DIR"
echo "✓ Found base model: $BASE_MODEL"
echo ""

echo "========================================================================"
echo "STEP 1: CREATE 4 TRAINING SPLITS"
echo "========================================================================"
echo ""

# Find the simulation stats file
SIMULATION_STATS=$(ls ${EEF_DIR}/simulation_stats_*.json 2>/dev/null | head -1)
SUCCESS_SEGMENTS=$(ls ${EEF_DIR}/full_success_segments_*.json 2>/dev/null | head -1)
IMPROVEMENT_SEGMENTS=$(ls ${EEF_DIR}/improvement_segments_*.json 2>/dev/null | head -1)
FAILURE_SEGMENTS=$(ls ${EEF_DIR}/failure_segments_*.json 2>/dev/null | head -1)

if [ -z "$SIMULATION_STATS" ]; then
    echo "❌ ERROR: No simulation_stats file found in $EEF_DIR"
    exit 1
fi

echo "Input files:"
echo "  Simulation stats: $SIMULATION_STATS"
echo "  Success segments: $SUCCESS_SEGMENTS"
echo "  Improvement segments: $IMPROVEMENT_SEGMENTS"
echo "  Failure segments: $FAILURE_SEGMENTS"
echo ""

python create_training_splits.py \
    --simulation_stats "$SIMULATION_STATS" \
    --success_segments "$SUCCESS_SEGMENTS" \
    --improvement_segments "$IMPROVEMENT_SEGMENTS" \
    --failure_segments "$FAILURE_SEGMENTS" \
    --output_dir "$TRAINING_DIR" \
    --middle_u_min 0.4 \
    --middle_u_max 0.7

echo ""
echo "✓ Training splits created in: $TRAINING_DIR"
echo ""

echo "========================================================================"
echo "STEP 2: TRAIN 4 MODELS"
echo "========================================================================"
echo ""

python train_advisor_experiments.py \
    --base_model "$BASE_MODEL" \
    --training_dir "$TRAINING_DIR" \
    --output_dir "$MODELS_DIR" \
    --epochs 3 \
    --batch_size 16 \
    --learning_rate 2e-5

echo ""
echo "✓ Models trained and saved to: $MODELS_DIR"
echo ""

echo "========================================================================"
echo "STEP 3: EVALUATE ALL 4 MODELS"
echo "========================================================================"
echo ""

python evaluate_advisor_experiments.py \
    --models_dir "$MODELS_DIR" \
    --test_tasks 200 \
    --max_steps 15 \
    --output_dir "$EVAL_DIR"

echo ""
echo "✓ Evaluation results saved to: $EVAL_DIR"
echo ""

echo "========================================================================"
echo "EXPERIMENT COMPLETE!"
echo "========================================================================"
echo ""
echo "Results summary:"
echo "  Training splits: $TRAINING_DIR"
echo "  Trained models: $MODELS_DIR"
echo "  Evaluation results: $EVAL_DIR"
echo ""
echo "Key files to check:"
echo "  1. $TRAINING_DIR/split_summary.json - Dataset statistics"
echo "  2. $MODELS_DIR/training_summary.json - Training losses"
echo "  3. $EVAL_DIR/evaluation_results.json - SUCCESS RATES ← ANSWER HERE"
echo ""
echo "Next steps:"
echo "  1. Check evaluation_results.json for success rate comparison"
echo "  2. Does Train-2 (Middle U) beat Train-1 (All Recoverable)?"
echo "  3. If yes → Quality > Quantity (paper finding!)"
echo "  4. If no → Volume matters more than curation"
echo ""