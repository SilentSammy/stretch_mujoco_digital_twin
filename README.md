# Stretch MuJoCo Digital Twin

A MuJoCo simulator for the Hello Robot Stretch 3, with a small API designed to
make robot scripts portable between the simulator and a physical Stretch.

The project includes position and velocity control for the base, lift, arm,
head, wrist, and gripper; RGB and depth cameras; 2D LiDAR; keyboard and
gamepad teleoperation; RoboCasa scenes; and configurable simulator objects.

## Install and Run

Install [uv](https://docs.astral.sh/uv/), then clone the repository with its
submodules:

```bash
git clone https://github.com/SilentSammy/stretch_mujoco_digital_twin --recurse-submodules
cd stretch_mujoco_digital_twin
uv sync
```

Run the camera-enabled teleoperation demo:

```bash
uv run examples/teleop_demo_cam.py
```

It starts with camera feeds disabled to keep the simulator responsive. Press
`1`--`5` to toggle head RGB, head depth, wrist RGB, wrist depth, and wide RGB.
Use `Q`, Escape, or Ctrl+C to exit.

On Linux, build errors mentioning `evdev` or `gcc` can usually be resolved with:

```bash
sudo apt install build-essential python3-dev linux-headers-$(uname -r)
```

## Cross-Environment Robot API

Use this import pattern in scripts that should work unchanged on both systems:

```python
try:
    import stretch_body.robot as robot
except ImportError:
    import stretch_mujoco_api.robot as robot

stretch = robot.Robot()
stretch.startup()
stretch.enable_collision_mgmt()
```

The simulated API covers the primary joint commands, robot status, head poses,
base movement, and stow/home commands. See `examples/stretch_boilerplate.py`
and `examples/robot_showcase.py` for small starting points.

## `stretch_tools`

`stretch_tools` provides environment-agnostic helpers used by both the real
robot and simulator:

- `TeleopProvider` reads keyboard and gamepad input.
- `NormVelController` applies normalized velocity dictionaries.
- `StateController` maintains joint target positions.
- `HEAD_CAMERA`, `WRIST_CAMERA`, and `NAVIGATION_CAMERA` provide camera frames.
- `LidarPlotter` and `filter_mast_points` support 2D LiDAR programs.
- `RobotTransforms` and `ObjectPlotter` support camera-based localization.

The examples directory includes teleoperation, camera, LiDAR, object-location,
and object-grabbing demonstrations.

## Configuring the Simulator

The root-level `sim_config.py` is loaded when the simulated robot starts. It
contains ready-made configurations for imported meshes, cubes, ArUco cubes,
robot start poses, RoboCasa kitchens, and interactive object controls. Select
one by assigning it to `CONFIG`:

```python
CONFIG = _CUBES
```

Dynamic and kinematic objects can be moved without changing a robot script.
Press `T` to enter object mode, `M`/`N` to select an object, `WASD` for world
`x`/`y` motion, and `Z`/`X` for world `z` motion. Leaving object mode prints
the changed object poses for reuse in `sim_config.py`.

## RoboCasa

RoboCasa is optional and requires its assets:

```bash
uv pip install -e ".[robocasa]"
uv pip install -e "robocasa@third_party/robocasa"
uv pip install -e "robosuite@third_party/robosuite"
uv run third_party/robosuite/robosuite/scripts/setup_macros.py
uv run third_party/robocasa/robocasa/scripts/setup_macros.py
uv run third_party/robocasa/robocasa/scripts/download_kitchen_assets.py
```

Then select `_ROBOCASA` or `_ROBOCASA_OBJECT` in `sim_config.py` and run an
example again.

## Activities

The course material is in `activities/`. The simulator scene reference is
`activities/simulator_scene_configuration_guide/simulator_scene_configuration_guide.tex`.

## Acknowledgment

This project builds on the Stretch MuJoCo work by Hello Robot, MuJoCo, Google
DeepMind, RoboCasa, and the MuJoCo Scanned Objects collection.
