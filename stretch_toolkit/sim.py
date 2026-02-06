"""Simulated robot implementation - placeholder for simulation backends."""
import time
import numpy as np
from .base import JointController, DepthCamInfo, RGBCamInfo
from stretch_mujoco import StretchMujocoSimulator
from stretch_mujoco.enums.actuators import Actuators
from stretch_mujoco.enums.stretch_cameras import StretchCameras


class SimulatedJointController(JointController):
    """Controls simulated robot joints using normalized velocities (-1.0 to 1.0)."""
    
    def __init__(self, sim: StretchMujocoSimulator, max_linear_accel: float = 0.15, max_angular_accel: float = 1.78):
        """Initialize simulated controller.
        
        Args:
            sim: StretchMujocoSimulator instance to control
            max_linear_accel: Maximum linear acceleration (m/s^2)
            max_angular_accel: Maximum angular acceleration (rad/s^2)
        """
        self.sim = sim
        
        # Current velocities for smoothing
        self.current_v_linear = 0.0
        self.current_omega = 0.0
        self.current_joint_vels = {}  # Track current velocity for each joint
        
        # Acceleration limits (m/s^2 and rad/s^2)
        self.max_linear_accel = max_linear_accel  # m/s^2
        self.max_angular_accel = max_angular_accel  # rad/s^2
        
        # Time tracking
        self.last_update_time = None
        
        # Max joint velocities (from original keyboard_teleop.py move_by values)
        # These represent the max increment per call, creating velocity-like behavior
        self.joint_max_speeds = {
            'lift_up': 0.2,
            'arm_out': 0.1,
            'head_tilt_up': 0.5,
            'head_pan_counterclockwise': -0.5,  # Negated for counterclockwise
            'wrist_yaw_counterclockwise': -1.0,  # Negated for counterclockwise
            'wrist_pitch_up': 0.05,
            'wrist_roll_counterclockwise': -0.25,  # Negated for counterclockwise
            'gripper_open': 0.07,
        }
        
        # Max joint accelerations (units/s^2 - matching the joint_max_speeds units)
        # Conservative values for smooth motion
        self.joint_max_accels = {
            'lift_up': 10,
            'arm_out': 10,
            'head_tilt_up': 1.0,
            'head_pan_counterclockwise': 1.0,
            'wrist_yaw_counterclockwise': 1.0,
            'wrist_pitch_up': 1.0,
            'wrist_roll_counterclockwise': 1.0,
            'gripper_open': 0.35,
        }
        
        # Map velocity dict keys to Actuator enums
        self.joint_actuator_map = {
            'lift_up': Actuators.lift,
            'arm_out': Actuators.arm,
            'head_tilt_up': Actuators.head_tilt,
            'head_pan_counterclockwise': Actuators.head_pan,
            'wrist_yaw_counterclockwise': Actuators.wrist_yaw,
            'wrist_pitch_up': Actuators.wrist_pitch,
            'wrist_roll_counterclockwise': Actuators.wrist_roll,
            'gripper_open': Actuators.gripper,
        }
    
    def _set_base_velocities(self, vel_dict, dt):
        """Set base velocities with acceleration smoothing and unit conversion.
        
        Args:
            vel_dict: Dictionary of normalized velocities
            dt: Time delta since last update (seconds)
        """
        # Real-world max velocities (m/s and rad/s)
        MAX_LINEAR_VEL_REAL = 0.1  # m/s - real robot max linear velocity
        MAX_ANGULAR_VEL_REAL = 1.77  # rad/s - real robot max angular velocity
        
        # Sim conversion factors (empirically determined)
        # sim_units = real_units * conversion_factor
        LINEAR_CONVERSION = 15.6  # 4.68 sim units = 0.3 m/s real
        ANGULAR_CONVERSION = 5.0  # Empirically determined
        
        # Calculate target velocities in real-world units
        target_v_linear = vel_dict.get('base_forward', 0.0) * MAX_LINEAR_VEL_REAL
        target_omega = vel_dict.get('base_counterclockwise', 0.0) * MAX_ANGULAR_VEL_REAL
        
        # Apply acceleration limits (in real-world units)
        max_linear_delta = self.max_linear_accel * dt
        max_angular_delta = self.max_angular_accel * dt
        
        # Ramp linear velocity
        v_linear_diff = target_v_linear - self.current_v_linear
        if abs(v_linear_diff) > max_linear_delta:
            self.current_v_linear += max_linear_delta if v_linear_diff > 0 else -max_linear_delta
        else:
            self.current_v_linear = target_v_linear
        
        # Ramp angular velocity
        omega_diff = target_omega - self.current_omega
        if abs(omega_diff) > max_angular_delta:
            self.current_omega += max_angular_delta if omega_diff > 0 else -max_angular_delta
        else:
            self.current_omega = target_omega

        # Convert to sim units and apply
        sim_v_linear = self.current_v_linear * LINEAR_CONVERSION
        sim_omega = self.current_omega * ANGULAR_CONVERSION
        self.sim.set_base_velocity(sim_v_linear, -sim_omega)
    
    def _set_joint_velocities(self, vel_dict, dt):
        """Set joint velocities via move_by with acceleration smoothing.
        
        Args:
            vel_dict: Dictionary of normalized velocities
            dt: Time delta since last update (seconds)
        """
        for joint_name, max_speed in self.joint_max_speeds.items():
            # Get target velocity
            normalized_vel = vel_dict.get(joint_name, 0.0)
            target_vel = normalized_vel * max_speed
            
            # Get current velocity for this joint (initialize if needed)
            if joint_name not in self.current_joint_vels:
                self.current_joint_vels[joint_name] = 0.0
            
            current_vel = self.current_joint_vels[joint_name]
            
            # Apply acceleration limit
            max_accel = self.joint_max_accels[joint_name]
            max_delta = max_accel * dt
            
            vel_diff = target_vel - current_vel
            if abs(vel_diff) > max_delta:
                current_vel += max_delta if vel_diff > 0 else -max_delta
            else:
                current_vel = target_vel
            
            # Store updated velocity
            self.current_joint_vels[joint_name] = current_vel
            
            # Apply movement if velocity is significant
            if abs(current_vel) > 0.001:  # Small deadzone to avoid jitter
                actuator = self.joint_actuator_map[joint_name]
                self.sim.move_by(actuator, current_vel)
    
    def set_velocities(self, vel_dict):
        """Set normalized joint velocities in simulation with acceleration smoothing.
        
        Args:
            vel_dict: Dict mapping joint names to velocities (-1.0 to 1.0)
                     Example: {'base_forward': 0.3, 'base_counterclockwise': 0.1}
        """
        # Calculate actual time delta
        current_time = time.perf_counter()
        if self.last_update_time is None:
            dt = 1/30  # Default for first call
        else:
            dt = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # Update base velocities
        self._set_base_velocities(vel_dict, dt)
        
        # Update joint positions with acceleration smoothing
        self._set_joint_velocities(vel_dict, dt)
    
    def get_state(self):
        """Get current joint positions and base odometry from simulation.
        
        Returns:
            dict: Joint positions with keys:
                - base_x, base_y, base_theta (odometry)
                - lift_up, arm_out (meters)
                - wrist/head joints (radians)
                - gripper_open (radians)
        """
        status = self.sim.pull_status()
        
        state = {
            # Base odometry
            'base_x': status.base.x,
            'base_y': status.base.y,
            'base_theta': status.base.theta,
            
            # Linear joints
            'lift_up': status.lift.pos,
            'arm_out': status.arm.pos,
            
            # Wrist joints
            'wrist_yaw_counterclockwise': status.wrist_yaw.pos,
            'wrist_pitch_up': status.wrist_pitch.pos,
            'wrist_roll_counterclockwise': status.wrist_roll.pos,
            
            # Head joints
            'head_pan_counterclockwise': status.head_pan.pos,
            'head_tilt_up': status.head_tilt.pos,
            
            # Gripper
            'gripper_open': status.gripper.pos
        }
        return state
    
    def stop(self):
        """Stop the simulated robot."""
        self.sim.set_base_velocity(0.0, 0.0)


# Frame getter functions for simulation cameras
def _get_head_cam_frames():
    """Get head camera frames from simulation."""
    # This will be called by camera system - needs access to global sim instance
    from . import _sim
    if _sim is None:
        return None, None
    try:
        camera_data = _sim.pull_camera_data()
        all_frames = camera_data.get_all(use_depth_color_map=False)
        rgb = all_frames.get(StretchCameras.cam_d435i_rgb)
        depth = all_frames.get(StretchCameras.cam_d435i_depth)
        return rgb, depth
    except:
        return None, None


def _get_wrist_cam_frames():
    """Get wrist camera frames from simulation."""
    from . import _sim
    if _sim is None:
        return None, None
    try:
        camera_data = _sim.pull_camera_data()
        all_frames = camera_data.get_all(use_depth_color_map=False)
        rgb = all_frames.get(StretchCameras.cam_d405_rgb)
        depth = all_frames.get(StretchCameras.cam_d405_depth)
        return rgb, depth
    except:
        return None, None


def _get_nav_cam_frame():
    """Get navigation camera frame from simulation."""
    from . import _sim
    if _sim is None:
        return None
    
    # Auto-register camera on first call (for testing dynamic camera management)
    if StretchCameras.cam_nav_rgb not in _sim.get_active_cameras():
        print("[sim.py] Auto-registering navigation camera...")
        _sim.add_camera(StretchCameras.cam_nav_rgb)
        # Give it a moment to initialize
        import time
        time.sleep(0.1)
    
    try:
        camera_data = _sim.pull_camera_data()
        all_frames = camera_data.get_all(use_depth_color_map=False)
        rgb = all_frames.get(StretchCameras.cam_nav_rgb)
        return rgb
    except:
        return None


# Camera instances for simulated robot
# D435i head camera (simulated)
HEAD_CAMERA = DepthCamInfo(
    name="D435i Head (Sim)",
    frame_getter=_get_head_cam_frames,
    camera_matrix=np.array([
        [303.07223511, 0.0,         122.78679657],
        [0.0,          303.06060791, 210.94392395],
        [0.0,          0.0,          1.0]
    ]),
    depth_scale=1e-03,
    distortion_coeffs=np.array([0., 0., 0., 0., 0.]),
    distortion_model="inverse_brown_conrady",
    depth_camera_matrix=np.array([
        [214.76873779, 0.0,         120.41242218],
        [0.0,          214.76873779, 209.7878418],
        [0.0,          0.0,          1.0]
    ]),
    depth_distortion_coeffs=np.array([0., 0., 0., 0., 0.]),
    depth_distortion_model="brown_conrady"
)

# D405 wrist camera (simulated)
WRIST_CAMERA = DepthCamInfo(
    name="D405 Wrist (Sim)",
    frame_getter=_get_wrist_cam_frames,
    camera_matrix=np.array([
        [385.62329102, 0.0,         314.58789062],
        [0.0,          385.1807251,  243.30551147],
        [0.0,          0.0,          1.0]
    ]),
    depth_scale=1e-04,
    distortion_coeffs=np.array([-5.52569292e-02, 5.98766357e-02, -8.58005136e-04,
                                 -9.32277253e-05, -1.93387289e-02]),
    distortion_model="inverse_brown_conrady",
    depth_camera_matrix=np.array([
        [378.52832031, 0.0,         318.47045898],
        [0.0,          378.52832031, 241.03790283],
        [0.0,          0.0,          1.0]
    ]),
    depth_distortion_coeffs=np.array([0., 0., 0., 0., 0.]),
    depth_distortion_model="brown_conrady"
)

# OV9782 navigation camera (simulated)
NAVIGATION_CAMERA = RGBCamInfo(
    name="OV9782 Navigation (Sim)",
    frame_getter=_get_nav_cam_frame,
)
