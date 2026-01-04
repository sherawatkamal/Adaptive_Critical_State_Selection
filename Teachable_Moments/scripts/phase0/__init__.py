"""
Phase 0: Data Collection via Simulation

Scripts for collecting training data through student and teacher rollouts:
1. collect_student_failures.py - Run student model, collect failure cases
2. collect_teacher_demos.py - Run teacher model, collect demonstrations
3. analyze_gaps.py - Find teachable gaps (teacher succeeds, student fails)

Typical workflow:
    # Step 1: Collect student failures
    python scripts/phase0/collect_student_failures.py \
        --model-path checkpoints/student_v1 \
        --n-tasks 500 \
        --output results/phase0/student_failures.json
    
    # Step 2: Collect teacher demos on failed tasks
    python scripts/phase0/collect_teacher_demos.py \
        --model gpt-4o \
        --student-results results/phase0/student_failures.json \
        --failures-only \
        --output results/phase0/teacher_demos.json
    
    # Step 3: Analyze gaps
    python scripts/phase0/analyze_gaps.py \
        --student-results results/phase0/student_failures.json \
        --teacher-results results/phase0/teacher_demos.json \
        --output results/phase0/teachable_gaps.json
"""
