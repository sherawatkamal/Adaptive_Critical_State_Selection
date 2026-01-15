#!/bin/bash
# Quick status check for parallel training

OUTPUT_DIR="./trained_models_advisor"

echo "========================================================================"
echo "PARALLEL TRAINING STATUS"
echo "========================================================================"
echo ""

# Check if processes are running
if ps -p 2583267 > /dev/null 2>&1; then
    echo "✓ Train-1 (GPU 0): RUNNING"
else
    echo "✓ Train-1 (GPU 0): COMPLETE"
fi

if ps -p 2583355 > /dev/null 2>&1; then
    echo "✓ Train-2 (GPU 1): RUNNING"
else
    echo "✓ Train-2 (GPU 1): COMPLETE"
fi

if ps -p 2583419 > /dev/null 2>&1; then
    echo "✓ Train-3 (GPU 2): RUNNING"
else
    echo "✓ Train-3 (GPU 2): COMPLETE"
fi

echo ""
echo "========================================================================"
echo "LATEST PROGRESS"
echo "========================================================================"
echo ""

echo "Train-1 (GPU 0):"
tail -1 "$OUTPUT_DIR/train1_log.txt" 2>/dev/null || echo "  No output yet..."
echo ""

echo "Train-2 (GPU 1):"
tail -1 "$OUTPUT_DIR/train2_log.txt" 2>/dev/null || echo "  No output yet..."
echo ""

echo "Train-3 (GPU 2):"
tail -1 "$OUTPUT_DIR/train3_log.txt" 2>/dev/null || echo "  No output yet..."
echo ""

echo "========================================================================"
echo "GPU USAGE"
echo "========================================================================"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | head -3

echo ""
echo "========================================================================"
echo ""
echo "Run 'bash monitor_training.sh' for live updates"
echo "Or check individual logs with: tail -f $OUTPUT_DIR/train1_log.txt"