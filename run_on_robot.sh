#!/bin/bash
# Helper script to run Python scripts on the Stretch robot with correct environment.
# Usage: ./run_on_robot.sh script_name.py [args...]
# Or: source run_on_robot.sh  (to set up environment in current shell)

# Deactivate conda if active
if [ ! -z "$CONDA_DEFAULT_ENV" ]; then
    echo "Deactivating conda ($CONDA_DEFAULT_ENV)..."
    conda deactivate 2>/dev/null || source deactivate 2>/dev/null
fi

# Set display for GUI (adjust :1 to :0 if needed)
export DISPLAY=:1

# If a script was provided, run it
if [ $# -gt 0 ]; then
    echo "Running: python3 $@"
    python3 "$@"
else
    echo "Environment ready. Now in system Python with stretch_body."
    echo "DISPLAY=$DISPLAY"
    echo ""
    echo "Run your script with: python3 your_script.py"
fi
