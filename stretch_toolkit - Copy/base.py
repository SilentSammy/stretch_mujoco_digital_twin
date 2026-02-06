"""Base classes for robot control - environment agnostic."""
from . import input as ci

import numpy as np
import math

class CamInfo:
    """Base class for camera configuration with intrinsics and projection methods."""
    
    def __init__(self, name, camera_matrix=None, distortion_coeffs=None, distortion_model=None):
        """
        Args:
            name: Camera identifier (e.g., "D435i Head")
            camera_matrix: 3x3 numpy array with camera intrinsics (optional)
            distortion_coeffs: Camera distortion coefficients (optional)
            distortion_model: Camera distortion model type (optional)
        """
        self.name = name
        self.camera_matrix = camera_matrix
        self.distortion_coeffs = distortion_coeffs
        self.distortion_model = distortion_model
    
    @property
    def has_intrinsics(self):
        """Check if camera intrinsics are available."""
        return self.camera_matrix is not None
    
    @property
    def fx(self):
        if not self.has_intrinsics:
            raise ValueError(f"Camera '{self.name}' has no intrinsics configured")
        return self.camera_matrix[0, 0]
    
    @property
    def fy(self):
        if not self.has_intrinsics:
            raise ValueError(f"Camera '{self.name}' has no intrinsics configured")
        return self.camera_matrix[1, 1]
    
    @property
    def cx(self):
        if not self.has_intrinsics:
            raise ValueError(f"Camera '{self.name}' has no intrinsics configured")
        return self.camera_matrix[0, 2]
    
    @property
    def cy(self):
        if not self.has_intrinsics:
            raise ValueError(f"Camera '{self.name}' has no intrinsics configured")
        return self.camera_matrix[1, 2]
    
    def pixel_to_normalized(self, centroid):
        """Convert pixel coordinates to normalized camera coordinates.
        
        Args:
            centroid: (x, y) pixel coordinates
        
        Returns:
            (x_norm, y_norm): Normalized coordinates where z=1 in camera frame
        """
        if not self.has_intrinsics:
            raise ValueError(f"Camera '{self.name}' requires intrinsics for pixel_to_normalized")
        
        x_norm = (centroid[0] - self.cx) / self.fx
        y_norm = (centroid[1] - self.cy) / self.fy
        
        return x_norm, y_norm
    
    def pixel_to_object_angles(self, centroid):
        """Convert pixel to angular position (pitch, yaw) in degrees relative to camera.
        
        Args:
            centroid: (x, y) pixel coordinates
        
        Returns:
            (pitch, yaw): Angular position in degrees
        """
        x_norm, y_norm = self.pixel_to_normalized(centroid)
        
        yaw_rad = math.atan(x_norm)
        pitch_rad = math.atan(y_norm)
        
        return math.degrees(pitch_rad), math.degrees(yaw_rad)
    
    def object_angles_to_pixel(self, yaw, pitch):
        """Convert angular position (yaw, pitch) to pixel coordinates.
        
        Args:
            yaw: Yaw angle in degrees
            pitch: Pitch angle in degrees
        
        Returns:
            (x, y): Pixel coordinates
        """
        if not self.has_intrinsics:
            raise ValueError(f"Camera '{self.name}' requires intrinsics for object_angles_to_pixel")
        
        yaw_rad = math.radians(yaw)
        pitch_rad = math.radians(pitch)
        
        x_norm = math.tan(yaw_rad)
        y_norm = math.tan(pitch_rad)
        
        x = x_norm * self.fx + self.cx
        y = y_norm * self.fy + self.cy
        
        return x, y


class RGBCamInfo(CamInfo):
    """Camera configuration for RGB-only cameras with optional intrinsics."""
    
    def __init__(self, name, frame_getter, camera_matrix=None,
                 distortion_coeffs=None, distortion_model=None):
        """
        Args:
            name: Camera identifier (e.g., "OV9782 Navigation")
            frame_getter: Function that returns rgb_frame (single frame, not tuple)
            camera_matrix: 3x3 numpy array with camera intrinsics (optional)
            distortion_coeffs: Camera distortion coefficients (optional)
            distortion_model: Camera distortion model type (optional)
        """
        super().__init__(name, camera_matrix, distortion_coeffs, distortion_model)
        self.get_frame = frame_getter


class DepthCamInfo(CamInfo):
    """Camera configuration for depth cameras with intrinsics and frame getter."""
    
    def __init__(self, name, frame_getter, camera_matrix, depth_scale,
                 distortion_coeffs=None, distortion_model=None,
                 depth_camera_matrix=None, depth_distortion_coeffs=None, depth_distortion_model=None):
        """
        Args:
            name: Camera identifier (e.g., "D435i Head", "D405 Wrist")
            frame_getter: Function that returns (rgb_frame, depth_frame) tuple
            camera_matrix: 3x3 numpy array with RGB camera intrinsics
            depth_scale: Meters per depth unit (e.g., 1e-03 for D435i)
            distortion_coeffs: RGB camera distortion coefficients (optional)
            distortion_model: RGB camera distortion model type (optional)
            depth_camera_matrix: 3x3 numpy array with depth camera intrinsics (optional, defaults to camera_matrix)
            depth_distortion_coeffs: Depth camera distortion coefficients (optional)
            depth_distortion_model: Depth camera distortion model type (optional)
        """
        super().__init__(name, camera_matrix, distortion_coeffs, distortion_model)
        self.get_frames = frame_getter
        self.depth_scale = depth_scale
        
        # Depth camera intrinsics (may differ from RGB)
        self.depth_camera_matrix = depth_camera_matrix if depth_camera_matrix is not None else camera_matrix
        self.depth_distortion_coeffs = depth_distortion_coeffs
        self.depth_distortion_model = depth_distortion_model
    
    @property
    def depth_fx(self):
        return self.depth_camera_matrix[0, 0]
    
    @property
    def depth_fy(self):
        return self.depth_camera_matrix[1, 1]
    
    @property
    def depth_cx(self):
        return self.depth_camera_matrix[0, 2]
    
    @property
    def depth_cy(self):
        return self.depth_camera_matrix[1, 2]

    def get_depth(self, centroid, depth_image, sample_radius=None):
        """Get distance to object in meters using depth camera intrinsics.
        
        Projects RGB pixel coordinate to depth camera coordinate system,
        then samples depth values around that location.
        
        Args:
            centroid: (x, y) pixel coordinates in RGB frame
            depth_image: Depth image array
            sample_radius: Radius in pixels to sample around centroid (default: 3)
            visualize: If True, show debug visualization of depth sampling (default: False)
        
        Returns:
            Median distance in meters, or None if no valid depth samples
        """
        if sample_radius is None:
            sample_radius = 3
        
        # Project RGB pixel to depth camera coordinate system
        # 1. Convert RGB pixel to normalized coordinates
        x_norm = (centroid[0] - self.cx) / self.fx
        y_norm = (centroid[1] - self.cy) / self.fy
        
        # 2. Project to depth camera pixel coordinates
        depth_x = x_norm * self.depth_fx + self.depth_cx
        depth_y = y_norm * self.depth_fy + self.depth_cy
        
        x, y = int(depth_x), int(depth_y)
        height, width = depth_image.shape[:2]
        
        # Collect depth samples in a square region
        samples = []
        for dy in range(-sample_radius, sample_radius + 1):
            for dx in range(-sample_radius, sample_radius + 1):
                # Clamp to image bounds
                px = max(0, min(width - 1, x + dx))
                py = max(0, min(height - 1, y + dy))
                
                depth_value = depth_image[py, px]
                if depth_value > 0:  # Only include valid depth values
                    samples.append(depth_value)
        
        if not samples:
            return None
        
        # Use median to reduce noise
        median_depth = np.median(samples)
        
        return median_depth * self.depth_scale


class TeleopProvider:
    """Provides teleoperation commands as normalized joint velocities."""
    def __init__(self, is_stretch_env=False):
        self.is_stretch_env = is_stretch_env
        if is_stretch_env:
            ci.INVERT_Y_AXIS = True
        
        # Toggle states
        self.dpad_controls_head = False  # False = wrist, True = head
        self.manual_mode_enabled = True  # True = manual control, False = algorithmic
        
        # Base joint mappings (always active)
        self.base_mappings = {
            'base_forward': ('w', 's', 'LY'),           # Left stick Y
            'base_counterclockwise': ('d', 'a', 'LX'), # Left stick X
            'lift_up': ('z', 'x', 'RY'),               # Right stick Y
            'arm_out': ('v', 'c', 'RX'),               # Right stick X
            'gripper_open': ('m', 'n', 'B', 'A'),                    # m=open, n=close
        }
        
        # D-pad mappings for wrist control
        self.dpad_wrist_mappings = {
            'wrist_yaw_counterclockwise': ('l', 'j', 'RB', 'LB'),  # Gamepad only
            'wrist_roll_counterclockwise': ('u', 'o', None, 'DPAD_X'),  # o=CCW, l=CW
            'wrist_pitch_up': ('i', 'k', None, 'DPAD_Y'),               # i=up, k=down
            'head_pan_counterclockwise': (),
            'head_tilt_up': (),
        }
        
        # D-pad mappings for head control
        self.dpad_head_mappings = {
            'wrist_roll_counterclockwise': (),
            'wrist_pitch_up': (),
            'head_pan_counterclockwise': ('l', 'j', 'DPAD_X'),   # o=CCW, l=CW
            'head_tilt_up': ('i', 'k', None, 'DPAD_Y'),          # i=up, k=down
        }
        
        self.joint_mappings = {}
        self._update_joint_mappings()

    def _update_joint_mappings(self):
        """Update joint mappings based on current toggle states."""
        # Start with base mappings
        self.joint_mappings = self.base_mappings.copy()
        
        # Add D-pad mappings based on mode
        if self.dpad_controls_head:
            self.joint_mappings.update(self.dpad_head_mappings)
        else:
            self.joint_mappings.update(self.dpad_wrist_mappings)

    def _normalize_mapping(self, mapping):
        """Normalize mapping tuple to 6 elements with defaults.
        
        Args:
            mapping: Tuple of (high_key, low_key, high_game, low_game, [keyboard_scale], [game_scale])
        
        Returns:
            tuple: 6-element tuple with defaults filled in
        """
        if not mapping:
            return (None, None, None, None, 1.0, 1.0)
        
        defaults = (None, None, None, None, 1.0, 1.0)
        return mapping + defaults[len(mapping):]

    def _get_joint_velocity(self, mapping):
        """Get normalized velocity from a joint mapping.
        
        Args:
            mapping: Tuple of (high_key, low_key, high_game, low_game, [keyboard_scale], [game_scale])
        
        Returns:
            float: Normalized velocity from -1.0 to 1.0
        """
        normalized = self._normalize_mapping(mapping)
        return ci.get_bipolar_ctrl(*normalized)

    def _button_pressed(self, button):
        """Check if a button was just pressed (rising edge).
        
        Args:
            button: Button name string
        
        Returns:
            bool: True if button was just pressed
        """
        if self.is_stretch_env:
            # Swap X and Y buttons in Stretch environment
            if button == 'X':
                button = 'Y'
            elif button == 'Y':
                button = 'X'
        return ci.rising_edge(button)

    def _check_toggles(self):
        """Check for toggle button presses and update states."""
        # X button toggles D-pad control mode
        if self._button_pressed('X') or self._button_pressed('h'):
            self.dpad_controls_head = not self.dpad_controls_head
            mode = "HEAD" if self.dpad_controls_head else "WRIST"
            print(f"D-pad now controls: {mode}")
            self._update_joint_mappings()

    def get_normalized_velocities(self):
        """Get normalized joint velocities from input devices.
        
        Returns:
            dict: Normalized velocities (-1.0 to 1.0) for all joints
        """
        # Check for toggle button presses
        self._check_toggles()
        
        result = {}
        for joint, mapping in self.joint_mappings.items():
            result[joint] = self._get_joint_velocity(mapping)
        return result


class JointController:
    """Base class for joint controllers."""
    def set_velocities(self, vel_dict):
        """Set normalized joint velocities.
        
        Args:
            vel_dict: Dict mapping joint names to velocities (-1.0 to 1.0)
        """
        raise NotImplementedError("Subclasses must implement set_velocities()")
    
    def get_state(self):
        """Get current joint positions and base odometry.
        
        Returns:
            dict: Joint positions with keys:
                - base_x, base_y, base_theta (odometry in meters/radians)
                - lift_up, arm_out (meters)
                - wrist joints, head joints (radians)
                - gripper_open (radians)
        """
        raise NotImplementedError("Subclasses must implement get_state()")

    def stop(self):
        """Stop all robot motion."""
        raise NotImplementedError("Subclasses must implement stop()")


def merge_proportional(cmd_primary, cmd_secondary, deadband=0.05):
    """Merge two command dictionaries with proportional blending.
    
    Primary command overrides secondary based on input magnitude.
    When primary input is below deadband, secondary is used.
    Otherwise, primary input strength determines blend between secondary and full output.
    
    Args:
        cmd_primary: Primary command dict (e.g., from teleop)
        cmd_secondary: Secondary command dict (e.g., from autonomous controller)
        deadband: Threshold below which primary is considered inactive (default 0.05)
    
    Returns:
        dict: Merged command with proportional blending
    """
    cmd_final = {}
    
    # Handle all joints from both commands
    all_joints = set(cmd_primary.keys()) | set(cmd_secondary.keys())
    
    for joint in all_joints:
        primary_input = cmd_primary.get(joint, 0.0)
        secondary_input = cmd_secondary.get(joint, 0.0)
        
        if abs(primary_input) < deadband:
            # No primary input - use secondary
            cmd_final[joint] = secondary_input
        else:
            # Primary input interpolates between secondary and desired value
            # abs(primary_input) determines how much override (0 to 1)
            # sign(primary_input) determines direction
            override_strength = abs(primary_input)
            desired_value = 1.0 if primary_input > 0 else -1.0
            cmd_final[joint] = (1 - override_strength) * secondary_input + override_strength * desired_value
    
    return cmd_final
