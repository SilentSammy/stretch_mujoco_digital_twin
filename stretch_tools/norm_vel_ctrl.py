def _clip_normalized(value):
    return max(-1.0, min(value, 1.0))


class NormVelController:
    SAFE_BASE_LIMIT = 0.33
    MAX_VELOCITIES = {
        "base_forward": 0.3,
        "base_counterclockwise": 1.9,
        "lift_up": 0.1,
        "arm_out": 0.1,
        "wrist_roll_counterclockwise": 1.0,
        "wrist_pitch_up": 0.8,
        "wrist_yaw_counterclockwise": 1.0,
        "head_pan_counterclockwise": 2.0,
        "head_tilt_up": 2.0,
        "gripper_open": 4.5,
    }

    def __init__(self, robot, safe_base_mode: bool = False) -> None:
        self.robot = robot
        self.safe_mode = safe_base_mode
        self.robot.enable_collision_mgmt()
        self.zero_cmd = {name: 0.0 for name in self.MAX_VELOCITIES}
        setters = {
            "base_forward": robot.base.set_translate_velocity,
            "base_counterclockwise": robot.base.set_rotational_velocity,
            "lift_up": robot.lift.set_velocity,
            "arm_out": robot.arm.set_velocity,
            "wrist_roll_counterclockwise": robot.end_of_arm.get_joint(
                "wrist_roll"
            ).set_velocity,
            "wrist_pitch_up": robot.end_of_arm.get_joint(
                "wrist_pitch"
            ).set_velocity,
            "wrist_yaw_counterclockwise": robot.end_of_arm.get_joint(
                "wrist_yaw"
            ).set_velocity,
            "head_pan_counterclockwise": robot.head.get_joint(
                "head_pan"
            ).set_velocity,
            "head_tilt_up": robot.head.get_joint("head_tilt").set_velocity,
            "gripper_open": robot.end_of_arm.get_joint(
                "stretch_gripper"
            ).set_velocity,
        }
        self._joint_setters = {
            name: lambda normalized, setter=setter, maximum=self.MAX_VELOCITIES[
                name
            ]: setter(normalized * maximum)
            for name, setter in setters.items()
        }

    def set_command(self, command: dict[str, float]) -> None:
        command = {
            name: _clip_normalized(normalized)
            for name, normalized in command.items()
        }
        base_names = {"base_forward", "base_counterclockwise"}
        combined_base = base_names <= command.keys()
        if combined_base:
            linear = command["base_forward"]
            angular = command["base_counterclockwise"]
            if self.safe_mode:
                linear = max(-self.SAFE_BASE_LIMIT, min(linear, self.SAFE_BASE_LIMIT))
                angular = max(-self.SAFE_BASE_LIMIT, min(angular, self.SAFE_BASE_LIMIT))
            self.robot.base.set_velocity(
                linear * self.MAX_VELOCITIES["base_forward"],
                angular * self.MAX_VELOCITIES["base_counterclockwise"],
            )

        for name, normalized in command.items():
            if name in base_names and combined_base:
                continue
            if self.safe_mode and name.startswith("base_"):
                normalized = max(
                    -self.SAFE_BASE_LIMIT,
                    min(normalized, self.SAFE_BASE_LIMIT),
                )
            self._joint_setters[name](normalized)

        self.robot.push_command()

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
        primary_input = _clip_normalized(cmd_primary.get(joint, 0.0))
        secondary_input = _clip_normalized(cmd_secondary.get(joint, 0.0))
        
        if abs(primary_input) < deadband:
            # No primary input - use secondary
            cmd_final[joint] = secondary_input
        else:
            # Primary input interpolates between secondary and desired value
            # abs(primary_input) determines how much override (0 to 1)
            # sign(primary_input) determines direction
            override_strength = abs(primary_input)
            desired_value = 1.0 if primary_input > 0 else -1.0
            cmd_final[joint] = _clip_normalized(
                (1 - override_strength) * secondary_input
                + override_strength * desired_value
            )
    
    return cmd_final
