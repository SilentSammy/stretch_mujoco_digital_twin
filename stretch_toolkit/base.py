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
            'arm_out': ('c', 'v', 'RX'),               # Right stick X
            'wrist_yaw_counterclockwise': (None, None, 'RB', 'LB'),  # Gamepad only
            'gripper_open': (None, None, 'B', 'A'),                  # Gamepad only
        }
        
        # D-pad mappings for wrist control
        self.dpad_wrist_mappings = {
            'wrist_roll_counterclockwise': (None, None, None, 'DPAD_X'),
            'wrist_pitch_up': (None, None, None, 'DPAD_Y'),
            'head_pan_counterclockwise': (),
            'head_tilt_up': (),
        }
        
        # D-pad mappings for head control
        self.dpad_head_mappings = {
            'wrist_roll_counterclockwise': (),
            'wrist_pitch_up': (),
            'head_pan_counterclockwise': (None, None, 'DPAD_X'),
            'head_tilt_up': (None, None, None, 'DPAD_Y'),
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
        if self._button_pressed('X'):
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
