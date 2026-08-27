import time


class StateController:
    """Generate normalized velocities for joint targets; gripper uses percent."""

    GRIPPER_VELOCITY_TOLERANCE = 0.05
    GRIPPER_SETTLE_TIME = 0.5

    KP = {
        "wrist_roll_counterclockwise": 1.0,
        "wrist_pitch_up": 1.0,
        "wrist_yaw_counterclockwise": 1.0,
        "lift_up": 10.0,
        "arm_out": 5.0,
        "head_pan_counterclockwise": 1.0,
        "head_tilt_up": 1.0,
        "gripper_open": 0.04,
    }
    MAX_VELOCITY = {"lift_up": 0.75}
    TOLERANCE = {
        "wrist_roll_counterclockwise": 0.05,
        "wrist_pitch_up": 0.05,
        "wrist_yaw_counterclockwise": 0.05,
        "lift_up": 0.05,
        "arm_out": 0.05,
        "head_pan_counterclockwise": 0.05,
        "head_tilt_up": 0.05,
        "gripper_open": 2.0,
    }

    def __init__(self, robot, desired_state):
        self.robot = robot
        self.desired_state = desired_state
        self.kp = self.KP.copy()
        self.max_velocity = self.MAX_VELOCITY.copy()
        self.tolerance = self.TOLERANCE.copy()
        self._gripper_target = desired_state.get("gripper_open")
        self._gripper_commanded = False
        self._gripper_settled_since = None

    def _sync_gripper_target(self):
        target = self.desired_state.get("gripper_open")
        if target != self._gripper_target:
            self._gripper_target = target
            self._gripper_commanded = False
            self._gripper_settled_since = None

    def get_current_state(self, status=None):
        status = self.robot.get_status() if status is None else status
        state = {
            "lift_up": status["lift"]["pos"],
            "arm_out": status["arm"]["pos"],
            "head_pan_counterclockwise": status["head"]["head_pan"]["pos"],
            "head_tilt_up": status["head"]["head_tilt"]["pos"],
            "wrist_roll_counterclockwise": status["end_of_arm"]["wrist_roll"]["pos"],
            "wrist_pitch_up": status["end_of_arm"]["wrist_pitch"]["pos"],
            "wrist_yaw_counterclockwise": status["end_of_arm"]["wrist_yaw"]["pos"],
            "gripper_open": status["end_of_arm"]["stretch_gripper"]["pos_pct"],
        }
        return {name: state[name] for name in self.desired_state}

    def is_at_goal(self):
        self._sync_gripper_target()
        status = self.robot.get_status()
        current = self.get_current_state(status)

        for name, target in self.desired_state.items():
            position_reached = (
                abs(current[name] - target) <= self.tolerance.get(name, 0.01)
            )
            if name != "gripper_open":
                if not position_reached:
                    return False
                continue
            if not self._gripper_commanded:
                if position_reached:
                    continue
                return False

            velocity = status["end_of_arm"]["stretch_gripper"]["vel"]
            if abs(velocity) > self.GRIPPER_VELOCITY_TOLERANCE:
                self._gripper_settled_since = None
                return False
            if self._gripper_settled_since is None:
                self._gripper_settled_since = time.monotonic()
                return False
            if (
                time.monotonic() - self._gripper_settled_since
                < self.GRIPPER_SETTLE_TIME
            ):
                return False

        return True

    def get_progress(self, previous_state):
        current = self.get_current_state()
        progress = {}
        for name, target in self.desired_state.items():
            distance = abs(target - previous_state[name])
            covered = abs(current[name] - previous_state[name])
            progress[name] = min(covered / distance, 1.0) if distance else 1.0
        return progress

    def get_command(self):
        self._sync_gripper_target()
        current = self.get_current_state()
        command = {}
        for name, target in self.desired_state.items():
            error = target - current[name]
            if abs(error) <= self.tolerance.get(name, 0.01):
                command[name] = 0.0
                continue

            if name == "gripper_open":
                self._gripper_commanded = True

            maximum = self.max_velocity.get(name, 1.0)
            command[name] = max(-maximum, min(maximum, self.kp.get(name, 1.0) * error))
        return command
