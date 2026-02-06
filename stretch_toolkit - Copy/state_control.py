"""State-based position control using P-controllers.

Converts desired joint positions into velocity commands that work with both
physical and simulated backends.
"""
from .base import JointController


# Mapping between simplified names and toolkit velocity command names
JOINT_TO_COMMAND = {
    "arm": "arm_out",
    "lift": "lift_up",
    "wrist_roll": "wrist_roll_counterclockwise",
    "wrist_pitch": "wrist_pitch_up",
    "wrist_yaw": "wrist_yaw_counterclockwise",
    "head_pan": "head_pan_counterclockwise",
    "head_tilt": "head_tilt_up",
    "gripper": "gripper_open",
}

# Reverse mapping for reading state
COMMAND_TO_JOINT = {v: k for k, v in JOINT_TO_COMMAND.items()}


class StateController:
    """Position-based controller that generates velocity commands to reach desired states.
    
    Uses proportional control to smoothly drive joints to target positions.
    Works with both physical and simulated JointController backends.
    """
    
    def __init__(self, controller: JointController, desired_state: dict):
        """Initialize state controller.
        
        Args:
            controller: JointController instance (physical or simulated)
            desired_state: Dict of desired joint positions, e.g.:
                {
                    "arm": 0.5,           # meters
                    "lift": 1.1,          # meters
                    "wrist_yaw": 0.0,     # radians
                    "gripper": 1.57,      # radians
                }
        """
        self.controller = controller
        self.desired_state = desired_state
        
        # Individual Kp values for different joint types
        self.Kp = {
            "wrist_roll": 1.0,       # rad -> normalized velocity
            "wrist_pitch": 1.0,      # rad -> normalized velocity
            "wrist_yaw": 1.0,        # rad -> normalized velocity
            "lift": 10.0,             # m -> normalized velocity
            "arm": 5.0,              # m -> normalized velocity
            "head_pan": 1.0,         # rad -> normalized velocity
            "head_tilt": 1.0,        # rad -> normalized velocity
            "gripper": 0.25          # rad -> normalized velocity
        }
        
        # Maximum velocity limits (overrides default 1.0)
        self.max_velocity = {
            "lift": 0.75,            # Limit lift to 75% max speed
            # Add other joint-specific limits here as needed
        }
        
        # Position tolerance for each joint
        self.tolerance = {
            "wrist_roll": 0.02,      # rad
            "wrist_pitch": 0.02,     # rad  
            "wrist_yaw": 0.02,       # rad
            "lift": 0.01,            # m
            "arm": 0.01,             # m
            "head_pan": 0.02,        # rad
            "head_tilt": 0.02,       # rad
            "gripper": 0.1           # rad
        }
    
    def get_current_state(self):
        """Get current joint positions from controller.
        
        Returns:
            dict: Current positions using simplified joint names
        """
        # Get full state from controller
        full_state = self.controller.get_state()
        
        # Convert to simplified names, filtering to only desired joints
        current_state = {}
        for simple_name in self.desired_state.keys():
            toolkit_name = JOINT_TO_COMMAND.get(simple_name)
            if toolkit_name and toolkit_name in full_state:
                current_state[simple_name] = full_state[toolkit_name]
        
        return current_state
    
    def is_at_goal(self):
        """Check if robot is within tolerance of desired state.
        
        Returns:
            bool: True if all joints are within tolerance
        """
        current_state = self.get_current_state()
        
        for joint, desired_pos in self.desired_state.items():
            if joint in current_state:
                error = abs(current_state[joint] - desired_pos)
                if error > self.tolerance.get(joint, 0.01):
                    return False
        return True
    
    def get_progress(self, previous_state):
        """Calculate progress from previous_state to desired_state.
        
        Args:
            previous_state: Dict of joint positions (simplified names)
        
        Returns:
            dict: Progress (0.0 to 1.0) for each joint
        """
        current_state = self.get_current_state()
        progress = {}
        
        for joint, desired_pos in self.desired_state.items():
            if joint in current_state and joint in previous_state:
                current_pos = current_state[joint]
                prev_pos = previous_state[joint]
                
                total_distance = abs(desired_pos - prev_pos)
                distance_covered = abs(prev_pos - current_pos)
                progress[joint] = distance_covered / total_distance if total_distance > 0 else 1.0
        
        return progress
    
    def get_command(self):
        """Generate velocity commands to reach desired state.
        
        Returns:
            dict: Normalized velocity commands using toolkit naming
        """
        current_state = self.get_current_state()
        command = {}
        
        for joint, desired_pos in self.desired_state.items():
            if joint in current_state:
                # Calculate position error
                error = desired_pos - current_state[joint]
                
                # Set to zero velocity if within tolerance
                if abs(error) <= self.tolerance.get(joint, 0.01):
                    velocity = 0.0
                else:
                    # Calculate proportional velocity using joint-specific Kp
                    kp = self.Kp.get(joint, 1.0)
                    velocity = kp * error
                    
                    # Clamp velocity to joint-specific max (default 1.0)
                    max_vel = self.max_velocity.get(joint, 1.0)
                    velocity = max(-max_vel, min(max_vel, velocity))
                
                # Map to toolkit command name
                command_name = JOINT_TO_COMMAND.get(joint)
                if command_name:
                    command[command_name] = velocity
        
        return command
