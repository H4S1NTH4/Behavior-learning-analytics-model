#!/bin/bash
# Training Monitor Script

echo "======================================"
echo "🔍 TRAINING STATUS CHECK"
echo "======================================"
echo

# Check if process is running
if pgrep -f "python train_model" > /dev/null; then
    echo "📊 Status: TRAINING IN PROGRESS"
    echo

    # Show process info
    ps aux | grep "python train_model" | grep -v grep | awk '{print "   PID:", $2, "| CPU:", $3"% | Memory:", $4"% | Runtime:", $10}'
    echo

    # Show log file size
    if [ -f training_output.log ]; then
        LOG_LINES=$(wc -l < training_output.log)
        echo "📝 Log file: $LOG_LINES lines"
        if [ $LOG_LINES -gt 0 ]; then
            echo
            echo "Latest output:"
            tail -10 training_output.log
        else
            echo "   (Output buffered, waiting for flush...)"
        fi
    fi

    echo
    echo "⏳ Training is still running. Check back in a few minutes."

else
    echo "✅ Status: TRAINING COMPLETED (or not started)"
    echo

    # Check for model files
    if [ -f models/trained_keystroke_model.pth ]; then
        echo "📦 Trained model found:"
        ls -lh models/trained_keystroke_model.pth
        ls -lh data/trained_templates.pkl 2>/dev/null
        echo

        # Show last 50 lines of training log
        if [ -f training_output.log ]; then
            LOG_LINES=$(wc -l < training_output.log)
            echo "📝 Training log ($LOG_LINES lines):"
            echo "-----------------------------------"
            tail -50 training_output.log
        fi
    else
        echo "⚠️  No trained model found. Training may have failed."
    fi
fi

echo
echo "======================================"
