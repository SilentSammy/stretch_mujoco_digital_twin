class NormalizedVelocityControl:
    MAX_VELOCITIES = {
        "base_forward": 0.1,
        "base_counterclockwise": 0.6,
        "lift_up": 0.1,
        "arm_out": 0.1,
        "wrist_roll_counterclockwise": 1.0,
        "wrist_pitch_up": 0.8,
        "wrist_yaw_counterclockwise": 1.0,
        "head_pan_counterclockwise": 2.0,
        "head_tilt_up": 2.0,
        "gripper_open": 4.5,
    }

    def __init__(self, robot) -> None:
        self.robot = robot
        self.robot.enable_collision_mgmt()
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
        self.robot.base.set_velocity(
            command["base_forward"] * self.MAX_VELOCITIES["base_forward"],
            command["base_counterclockwise"]
            * self.MAX_VELOCITIES["base_counterclockwise"],
        )

        for name, set_velocity in self._joint_setters.items():
            if not name.startswith("base_"):
                set_velocity(command[name])

        self.robot.push_command()
