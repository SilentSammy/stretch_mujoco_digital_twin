# Stretch Toolkit

Cross-platform robot control for Hello Robot Stretch - works in both physical and simulated environments.

## Features

- **Unified Interface**: Write code once, run anywhere (physical robot or simulation)
- **Auto-Detection**: Automatically selects backend based on environment
- **Teleoperation**: Built-in gamepad/keyboard input handling
- **Normalized Commands**: All velocities are -1.0 to 1.0 for easy control

## Installation

```bash
# Clone this package
pip install inputs pynput  # For input handling

# Physical robot: stretch_body should already be installed
# Simulation: No additional dependencies needed (yet)
```

## Quick Start

```python
from stretch_toolkit import controller, teleop

while True:
    velocities = teleop.get_normalized_velocities()
    controller.set_velocities(velocities)
```

## Environment Selection

The toolkit automatically detects your environment:
- **Physical Robot**: If `stretch_body` is importable → uses physical backend
- **Simulation**: If `USE_SIM=1` environment variable or no stretch_body → uses sim backend

Force simulation mode:
```bash
export USE_SIM=1
python your_script.py
```

### Running on Physical Robot

On the Stretch robot, ensure you're using the system Python (not conda base):

```bash
# Option 1: Deactivate conda if active
conda deactivate

# Option 2: Use the helper script (from repo root)
./run_on_robot.sh your_script.py

# Option 3: Disable conda auto-activation (one-time setup)
conda config --set auto_activate_base false
# Then log out and back in
```

**Important**: The robot's system Python has `stretch_body` installed, but conda's base environment does not. If you see:
```
[stretch_toolkit] WARNING: But stretch_body import failed: No module named 'stretch_body'
```
...then deactivate conda first.

For GUI applications (OpenCV windows, matplotlib), set `DISPLAY`:
```bash
DISPLAY=:1 python3 your_script.py
```

## File Structure

```
stretch_toolkit/
├── __init__.py       # Auto-selects backend, exports controller & teleop
├── base.py           # Base classes (TeleopProvider, JointController)
├── physical.py       # Physical robot implementation
└── sim.py            # Simulation implementation
```

## Examples

See `example_teleop.py` for a complete teleoperation demo.

## Joint Names

All commands use descriptive joint names:
- `base_forward`, `base_counterclockwise`
- `lift_up`, `arm_out`
- `wrist_roll_counterclockwise`, `wrist_pitch_up`, `wrist_yaw_counterclockwise`
- `head_pan_counterclockwise`, `head_tilt_up`
- `gripper_open`

## Development

To add a new simulation backend, edit `sim.py` and implement the interface mapping.
