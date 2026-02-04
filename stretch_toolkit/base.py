"""Base classes for robot control - environment agnostic."""
from . import input as ci

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
