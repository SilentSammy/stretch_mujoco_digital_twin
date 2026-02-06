"""
Stretch Toolkit - Unified interface for physical and simulated robot control.

Environment auto-detection:
- If stretch_body is available → Physical robot mode
- If USE_SIM=1 environment variable → Simulation mode
- Otherwise → Simulation mode (default for dev environments)

Usage:
    from stretch_toolkit import controller, teleop
    
    while True:
        velocities = teleop.get_normalized_velocities()
        controller.set_velocities(velocities)
"""
import os
import sys

# Determine which backend to use
USE_PHYSICAL = False
BACKEND_NAME = "simulation"

# Check for explicit simulation flag
if os.getenv('USE_SIM', '0') == '1':
    USE_PHYSICAL = False
    BACKEND_NAME = "simulation"
else:
    # Try to import stretch_body to detect physical robot
    try:
        import stretch_body.robot
        USE_PHYSICAL = True
        BACKEND_NAME = "physical"
    except ImportError:
        USE_PHYSICAL = False
        BACKEND_NAME = "simulation"

print(f"[stretch_toolkit] Loading {BACKEND_NAME} backend")

# Import base classes (always available)
from .base import TeleopProvider, JointController, merge_proportional

# Import state control
from .state_control import StateController
if USE_PHYSICAL:
    try:
        from .physical import PhysicalJointController, HEAD_CAMERA, WRIST_CAMERA, NAVIGATION_CAMERA
        import stretch_body.robot as rb
        
        # Create robot instance
        robot = rb.Robot()
        robot.startup()
        robot.enable_collision_mgmt()
        
        # Create controller
        controller = PhysicalJointController(robot=robot)
        teleop = TeleopProvider(is_stretch_env=True)
        
        print("[stretch_toolkit] Physical robot initialized")
    except Exception as e:
        print(f"[stretch_toolkit] ERROR: Failed to initialize physical robot: {e}")
        print("[stretch_toolkit] Falling back to simulation mode")
        USE_PHYSICAL = False

if not USE_PHYSICAL:
    from .sim import SimulatedJointController, HEAD_CAMERA, WRIST_CAMERA, NAVIGATION_CAMERA
    from stretch_mujoco import StretchMujocoSimulator
    from stretch_mujoco.enums.stretch_cameras import StretchCameras
    
    # Lazy initialization - only create sim when first accessed
    _sim = None
    _controller = None
    
    def _get_controller():
        global _sim, _controller
        if _controller is None:
            # Initialize with NO cameras by default (better performance)
            # Users can explicitly enable cameras if needed
            # _sim = StretchMujocoSimulator(cameras_to_use=[StretchCameras.cam_d405_rgb, StretchCameras.cam_d405_depth])
            _sim = StretchMujocoSimulator(cameras_to_use=[])
            _sim.start()
            _controller = SimulatedJointController(sim=_sim)
            print("[stretch_toolkit] MuJoCo simulation initialized (no cameras for performance)")
            print("[stretch_toolkit] To enable cameras, reinitialize with desired camera list")
        return _controller
    
    # Create a proxy object that initializes on first use
    class _ControllerProxy:
        def __getattr__(self, name):
            return getattr(_get_controller(), name)
    
    controller = _ControllerProxy()
    teleop = TeleopProvider(is_stretch_env=False)
    
    print("[stretch_toolkit] Simulation mode ready (lazy init)")

# Export public API
__all__ = [
    'controller',
    'teleop',
    'TeleopProvider',
    'JointController',
    'StateController',
    'merge_proportional',
    'USE_PHYSICAL',
    'BACKEND_NAME',
    'HEAD_CAMERA',
    'WRIST_CAMERA',
    'NAVIGATION_CAMERA',
]
