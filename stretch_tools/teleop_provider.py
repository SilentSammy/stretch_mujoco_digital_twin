from . import input as ci

import numpy as np
import math
import json
import os
import time
from pathlib import Path


class TeleopProvider:
    """Provides teleoperation commands as normalized joint velocities."""
    def __init__(self, is_stretch_env=None, config_file='teleop_mappings.json'):
        if is_stretch_env is None:
            try:
                import stretch_body

                is_stretch_env = True
            except ImportError:
                is_stretch_env = False
        self.is_stretch_env = is_stretch_env
        
        # Store config file in this script's directory
        script_dir = Path(__file__).parent
        self.config_file = script_dir / config_file
        self.last_mtime = None
        
        # Toggle states
        self.dpad_controls_head = False  # False = wrist, True = head
        self.manual_mode_enabled = False  # False = autonomous mode, True = manual override
        
        # Load mappings from config file
        self._load_or_create_config()
        
        self.joint_mappings = {}
        self._update_joint_mappings()

    def _get_default_config(self):
        """Get default teleop mappings configuration."""
        return {
            'irl': {
                'base_mappings': {
                    'base_forward': ['w', 's', 'LY'],
                    'base_counterclockwise': ['d', 'a', 'LX'],
                    'lift_up': ['z', 'x', 'RY'],
                    'arm_out': ['v', 'c', 'RX'],
                    'gripper_open': ['m', 'n', 'B', 'A'],
                    'wrist_yaw_counterclockwise': ['l', 'j', 'RB', 'LB'],
                    'wrist_roll_counterclockwise': ['u', 'o', None, 'DPAD_X'],
                    'wrist_pitch_up': ['i', 'k', None, 'DPAD_Y'],
                },
                'dpad_head_mappings': {
                    'wrist_yaw_counterclockwise': [],
                    'wrist_roll_counterclockwise': [],
                    'wrist_pitch_up': [],
                    'head_pan_counterclockwise': ['l', 'j', 'DPAD_X'],
                    'head_tilt_up': ['i', 'k', None, 'DPAD_Y'],
                },
                'toggle_buttons': {
                    'head_wrist_toggle': ['X', 'h'],
                    'manual_mode_toggle': ['X', 'y']
                }
            },
            'sim': {
                # Sim-specific overrides can go here
            }
        }

    def _load_or_create_config(self):
        """Load configuration from JSON file, creating with defaults if it doesn't exist."""
        if not self.config_file.exists():
            # Create file with defaults
            defaults = self._get_default_config()
            with open(self.config_file, 'w') as f:
                json.dump(defaults, f, indent=2)
            print(f"Created default teleop config: {self.config_file}")
        
        # Load from file
        self._load_config()

    def _load_config(self):
        """Load configuration from JSON file and update modification time."""
        with open(self.config_file, 'r') as f:
            config = json.load(f)
        
        # Start with 'irl' config (base/default)
        irl_config = config.get('irl', config)  # Fallback to root if no 'irl' key
        
        # If not stretch_env (i.e., simulation), recursively override with 'sim' config
        if not self.is_stretch_env and 'sim' in config:
            final_config = self._recursive_merge(irl_config, config['sim'])
        else:
            final_config = irl_config
        
        # Convert lists back to tuples
        self.base_mappings = {k: tuple(v) for k, v in final_config.get('base_mappings', {}).items()}
        self.dpad_head_mappings = {k: tuple(v) for k, v in final_config.get('dpad_head_mappings', {}).items()}
        
        # Load toggle buttons
        self.toggle_buttons = final_config.get('toggle_buttons', {'head_wrist_toggle': ['X', 'h']})
        
        # Update modification time
        self.last_mtime = os.path.getmtime(self.config_file)
    
    def _recursive_merge(self, base, override):
        """Recursively merge override dict into base dict.
        
        Args:
            base: Base dictionary
            override: Override dictionary (values override base)
        
        Returns:
            Merged dictionary
        """
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursively merge nested dicts
                result[key] = self._recursive_merge(result[key], value)
            else:
                # Override value
                result[key] = value
        return result

    def _check_and_reload_config(self):
        """Check if config file has been modified and reload if necessary."""
        if not self.config_file.exists():
            return
        
        current_mtime = os.path.getmtime(self.config_file)
        if current_mtime != self.last_mtime:
            print(f"Teleop config file changed, reloading: {self.config_file}")
            self._load_config()
            self._update_joint_mappings()

    def _update_joint_mappings(self):
        """Update joint mappings based on current toggle states."""
        # Start with base mappings
        self.joint_mappings = self.base_mappings.copy()
        
        # Add head mappings when toggle is active
        if self.dpad_controls_head:
            self.joint_mappings.update(self.dpad_head_mappings)

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
        return ci.rising_edge(button)

    def _check_toggles(self):
        """Check for toggle button presses and update states."""
        # Check head/wrist toggle buttons
        toggle_buttons = self.toggle_buttons.get('head_wrist_toggle', [])
        if any(self._button_pressed(btn) for btn in toggle_buttons if btn):
            self.dpad_controls_head = not self.dpad_controls_head
            mode = "HEAD (override wrist)" if self.dpad_controls_head else "WRIST (default)"
            print(f"Controls: {mode}")
            self._update_joint_mappings()
        
        # Check manual mode toggle buttons
        manual_toggle_buttons = self.toggle_buttons.get('manual_mode_toggle', [])
        if any(self._button_pressed(btn) for btn in manual_toggle_buttons if btn):
            self.manual_mode_enabled = not self.manual_mode_enabled
            mode = "MANUAL" if self.manual_mode_enabled else "AUTONOMOUS"
            print(f"Mode: {mode}")

    def get_normalized_velocities(self):
        """Get normalized joint velocities from input devices.
        
        Returns:
            dict: Normalized velocities (-1.0 to 1.0) for all joints
        """
        # Check for config file updates
        self._check_and_reload_config()
        
        # Check for toggle button presses
        self._check_toggles()
        
        result = {}
        for joint, mapping in self.joint_mappings.items():
            result[joint] = self._get_joint_velocity(mapping)
        return result

    def get_manual_override(self, cmd_autonomous):
        """Merge an autonomous command with teleop input, giving the operator priority.

        When the operator moves a joint, their input proportionally overrides the
        autonomous command for that joint. Joints the operator is not touching
        continue to follow the autonomous command unmodified.

        If manual_mode_enabled is True, ignores cmd_autonomous completely and returns
        pure teleop control.

        Args:
            cmd_autonomous: Dict of normalized velocities from an autonomous controller.

        Returns:
            dict: Merged command to pass directly to controller.set_velocities().
        """
        cmd_teleop = self.get_normalized_velocities()
        if self.manual_mode_enabled:
            # Pure manual control - ignore autonomous command
            return cmd_teleop
        # Proportional blend - operator can override autonomous
        return merge_proportional(cmd_teleop, cmd_autonomous)
