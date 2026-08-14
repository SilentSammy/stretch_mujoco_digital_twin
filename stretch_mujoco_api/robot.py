"""A small ``stretch_body.robot``-compatible facade for MuJoCo."""

import math
import threading
import time

from stretch_mujoco.enums.actuators import Actuators
from stretch_mujoco.stretch_mujoco_simulator import StretchMujocoSimulator


class Joint:
    def __init__(
        self,
        robot: "Robot",
        actuator: Actuators,
        default_velocity: float,
        max_velocity: float,
        requires_push: bool,
        limits: tuple[float, float] | None = None,
        position_tolerance: float = 0.005,
        velocity_tolerance: float = 0.005,
        correction_gain: float = 20.0,
        max_correction: float = 0.015,
    ) -> None:
        self._robot = robot
        self._actuator = actuator
        self.default_velocity = default_velocity
        self.max_velocity = max_velocity
        self.requires_push = requires_push
        self._limits = limits
        self.position_tolerance = position_tolerance
        self.velocity_tolerance = velocity_tolerance
        self.correction_gain = correction_gain
        self.max_correction = max_correction
        self._pending_motion: tuple[float, float] | None = None
        self._motion: tuple[float, float] | None = None
        self._desired_position = 0.0

    @property
    def status(self) -> dict:
        status = self._robot._sim.pull_status()
        position = self._actuator.get_position(status)
        velocity = self._actuator.get_velocity(status)
        destination = self._motion[0] if self._motion is not None else position
        near_pos_setpoint = abs(destination - position) <= self.position_tolerance
        is_moving = abs(velocity) > self.velocity_tolerance
        timestamp = time.time()
        return {
            "timestamp_pc": timestamp,
            "pos": position,
            "vel": velocity,
            "motor": {
                "timestamp": timestamp,
                "pos_calibrated": True,
                "runstop_on": False,
                "near_pos_setpoint": near_pos_setpoint,
                "is_moving": is_moving,
                "is_moving_filtered": is_moving,
                "is_mg_moving": not near_pos_setpoint or is_moving,
                "is_mg_accelerating": False,
                "in_guarded_event": False,
                "in_safety_event": False,
                "calibration_rcvd": True,
            },
        }

    @property
    def limits(self) -> tuple[float, float]:
        return self._limits or self._robot._sim.pull_joint_limits()[self._actuator]

    @property
    def total_range(self) -> float:
        lower, upper = self.limits
        return float(upper - lower)

    def _move(self, to: float, at: float) -> None:
        lower, upper = self.limits
        destination = min(max(to, lower), upper)
        velocity = min(abs(at), self.max_velocity)
        self._pending_motion = destination, velocity
        if not self.requires_push:
            self._robot._activate_joint(self)

    def _startup(self, status) -> None:
        self._desired_position = self._actuator.get_position(status)
        self._motion = self._desired_position, 0.0

    def _push_command(self) -> bool:
        if self._pending_motion is None:
            return False
        self._motion = self._pending_motion
        self._pending_motion = None
        return True

    def _update(self, status, elapsed: float) -> bool:
        if self._motion is None:
            return True

        destination, velocity = self._motion
        remaining = destination - self._desired_position

        if velocity == 0 or abs(remaining) <= velocity * elapsed:
            self._desired_position = destination
        else:
            direction = 1 if remaining > 0 else -1
            self._desired_position += direction * velocity * elapsed

        actual_position = self._actuator.get_position(status)
        error = self._desired_position - actual_position
        correction = min(
            max(self.correction_gain * error, -self.max_correction),
            self.max_correction,
        )
        lower, upper = self.limits
        actuator_target = min(max(self._desired_position + correction, lower), upper)
        self._robot._sim.move_to(self._actuator, actuator_target)

        return (
            self._desired_position == destination
            and abs(destination - actual_position) <= self.position_tolerance
            and abs(self._actuator.get_velocity(status)) <= self.velocity_tolerance
        )


class PrismaticJoint(Joint):
    def __init__(
        self,
        robot: "Robot",
        actuator: Actuators,
        default_velocity: float,
        max_velocity: float,
        limits: tuple[float, float] | None = None,
    ) -> None:
        super().__init__(
            robot,
            actuator,
            default_velocity,
            max_velocity,
            requires_push=True,
            limits=limits,
        )

    # TODO: Trajectory support
    def home(self) -> None:
        """Compatibility no-op; the simulated robot is always homed."""

    def move_to(self, position_m: float, v_m: float | None = None) -> None:
        self._move(position_m, self.default_velocity if v_m is None else v_m)

    def move_by(self, distance_m: float, v_m: float | None = None) -> None:
        velocity = self.default_velocity if v_m is None else v_m
        self._move(self.status["pos"] + distance_m, velocity)

    def set_velocity(self, velocity_m: float) -> None:
        lower, upper = self.limits
        destination = upper if velocity_m > 0 else lower
        if velocity_m == 0:
            destination = self.status["pos"]
        self._move(destination, velocity_m)


class RevoluteJoint(Joint):
    MOTION_THRESHOLD = 0.01

    def __init__(
        self,
        robot: "Robot",
        actuator: Actuators,
        default_velocity: float,
        max_velocity: float,
        limits: tuple[float, float] | None = None,
        correction_gain: float = 0.0,
        max_correction: float = 0.0,
    ) -> None:
        super().__init__(
            robot,
            actuator,
            default_velocity,
            max_velocity,
            requires_push=False,
            limits=limits,
            position_tolerance=0.01,
            velocity_tolerance=float("inf"),
            correction_gain=correction_gain,
            max_correction=max_correction,
        )

    @property
    def status(self) -> dict:
        status = self._robot._sim.pull_status()
        velocity = self._actuator.get_velocity(status)
        return {
            "timestamp_pc": time.time(),
            "pos": self._actuator.get_position(status),
            "vel": velocity,
            "stalled": abs(velocity) <= self.MOTION_THRESHOLD,
        }

    def move_to(self, position_rad: float, v_r: float | None = None) -> None:
        self._move(position_rad, self.default_velocity if v_r is None else v_r)

    def move_by(self, angle_rad: float, v_r: float | None = None) -> None:
        velocity = self.default_velocity if v_r is None else v_r
        self._move(self.status["pos"] + angle_rad, velocity)

    def set_velocity(self, velocity_r: float) -> None:
        lower, upper = self.limits
        destination = upper if velocity_r > 0 else lower
        if velocity_r == 0:
            destination = self.status["pos"]
        self._move(destination, velocity_r)


class GripperJoint(Joint):
    FINGER_RAD_PER_PCT = 0.003759
    MOTOR_RAD_PER_FINGER_RAD = 11.42
    APERTURE_M_PER_FINGER_RAD = 0.342
    SIM_TO_FINGER_VELOCITY = 15.6
    MOTION_THRESHOLD = 0.01

    def __init__(self, robot: "Robot") -> None:
        super().__init__(
            robot,
            Actuators.gripper,
            default_velocity=0.4,
            max_velocity=0.4,
            requires_push=False,
            limits=(-0.376, 0.56),
            position_tolerance=0.005,
            velocity_tolerance=float("inf"),
            correction_gain=1.0,
            max_correction=0.01,
        )

    @property
    def status(self) -> dict:
        status = self._robot._sim.pull_status()
        finger_rad = self._actuator.get_position(status)
        finger_vel = (
            self._actuator.get_velocity(status) * self.SIM_TO_FINGER_VELOCITY
        )
        motor_rad = finger_rad * self.MOTOR_RAD_PER_FINGER_RAD
        motor_vel = finger_vel * self.MOTOR_RAD_PER_FINGER_RAD
        return {
            "timestamp_pc": time.time(),
            "pos": motor_rad,
            "vel": motor_vel,
            "effort": 0.0,
            "stalled": abs(motor_vel) <= self.MOTION_THRESHOLD,
            "pos_pct": finger_rad / self.FINGER_RAD_PER_PCT,
            "gripper_conversion": {
                "aperture_m": finger_rad * self.APERTURE_M_PER_FINGER_RAD,
                "finger_rad": finger_rad,
                "finger_effort": 0.0,
                "finger_vel": finger_vel,
            },
        }

    def move_to(self, position_pct: float, v_r: float | None = None) -> None:
        velocity = (
            self.default_velocity
            if v_r is None
            else v_r / self.MOTOR_RAD_PER_FINGER_RAD
        )
        self._move(position_pct * self.FINGER_RAD_PER_PCT, velocity)

    def move_by(self, distance_pct: float, v_r: float | None = None) -> None:
        status = self._robot._sim.pull_status()
        position = self._actuator.get_position(status)
        velocity = (
            self.default_velocity
            if v_r is None
            else v_r / self.MOTOR_RAD_PER_FINGER_RAD
        )
        self._move(position + distance_pct * self.FINGER_RAD_PER_PCT, velocity)


class Base:
    WHEEL_RADIUS = 0.0508
    WHEEL_SEPARATION = 0.3153
    MOTOR_GEAR_RATIO = 4.0
    VELOCITY_SCALE = 1.22
    LINEAR_ACCELERATION = 0.10
    ANGULAR_ACCELERATION = 0.4

    def __init__(self, robot: "Robot") -> None:
        self._robot = robot
        self._pending_command: tuple | None = None
        self._motion: tuple | None = None
        self._left_pos = 0.0
        self._right_pos = 0.0
        self._rotation_traveled = 0.0
        self._last_theta = 0.0
        self._linear_command = 0.0
        self._angular_command = 0.0

    @property
    def status(self) -> dict:
        status = self._robot._sim.pull_status()
        linear = status.base.x_vel
        angular = status.base.theta_vel
        left_vel, right_vel = self._wheel_velocities(linear, angular)
        timestamp = time.time()
        return {
            "timestamp_pc": timestamp,
            "x": status.base.x,
            "y": status.base.y,
            "theta": status.base.theta % (2 * math.pi),
            "x_vel": linear,
            "y_vel": 0.0,
            "theta_vel": angular,
            "pose_time_s": status.time,
            "effort": [0.0, 0.0],
            "left_wheel": self._wheel_status(
                timestamp, self._left_pos, left_vel
            ),
            "right_wheel": self._wheel_status(
                timestamp, self._right_pos, right_vel
            ),
            "translation_force": 0.0,
            "rotation_torque": 0.0,
        }

    def translate_by(
        self,
        distance_m: float,
        v_m: float | None = None,
        a_m: float | None = None,
    ) -> None:
        self._pending_command = "translate", distance_m, 0.11 if v_m is None else v_m

    def rotate_by(
        self,
        angle_rad: float,
        v_r: float | None = None,
        a_r: float | None = None,
    ) -> None:
        self._pending_command = "rotate", angle_rad, 0.4 if v_r is None else v_r

    def set_velocity(self, linear_m_s: float, angular_rad_s: float) -> None:
        self._pending_command = "velocity", linear_m_s, angular_rad_s

    def _startup(self) -> None:
        self._motion = None

    def _push_command(self) -> bool:
        if self._pending_command is None:
            return False

        command = self._pending_command
        self._pending_command = None
        status = self._robot._sim.pull_status().base

        if command[0] == "velocity":
            self._motion = command
            return False

        self._motion = command + (status.x, status.y, status.theta)
        self._linear_command = status.x_vel
        self._angular_command = status.theta_vel
        self._rotation_traveled = 0.0
        self._last_theta = status.theta
        return True

    def _update(self, status, elapsed: float) -> bool:
        left_vel, right_vel = self._wheel_velocities(
            status.base.x_vel, status.base.theta_vel
        )
        self._left_pos += left_vel * elapsed
        self._right_pos += right_vel * elapsed

        if self._motion is None:
            return True

        if self._motion[0] == "velocity":
            linear, angular = self._motion[1:]
            self._ramp_velocity(linear, angular, elapsed)
            return True

        kind, amount, speed, start_x, start_y, start_theta = self._motion
        if kind == "translate":
            dx = status.base.x - start_x
            dy = status.base.y - start_y
            traveled = dx * math.cos(start_theta) + dy * math.sin(start_theta)
            remaining = abs(amount) - abs(traveled)
            command = min(
                abs(speed),
                math.sqrt(max(2 * self.LINEAR_ACCELERATION * remaining, 0.0)),
            )
            linear = command if amount >= 0 else -command
            angular = 0.0
            settled = abs(status.base.x_vel) <= 0.005
        else:
            delta = (status.base.theta - self._last_theta + math.pi) % (
                2 * math.pi
            ) - math.pi
            self._rotation_traveled += delta
            self._last_theta = status.base.theta
            remaining = abs(amount) - abs(self._rotation_traveled)
            command = min(
                abs(speed),
                math.sqrt(max(2 * self.ANGULAR_ACCELERATION * remaining, 0.0)),
            )
            linear = 0.0
            angular = command if amount >= 0 else -command
            settled = abs(status.base.theta_vel) <= 0.01

        if remaining > 0:
            self._ramp_velocity(linear, angular, elapsed)
            return False

        self._ramp_velocity(0.0, 0.0, elapsed)
        if settled:
            self._motion = None
            return True
        return False

    def _ramp_velocity(self, linear: float, angular: float, elapsed: float) -> None:
        self._linear_command = self._approach(
            self._linear_command,
            linear,
            self.LINEAR_ACCELERATION * elapsed,
        )
        self._angular_command = self._approach(
            self._angular_command,
            angular,
            self.ANGULAR_ACCELERATION * elapsed,
        )
        self._robot._sim.set_base_velocity(
            self._linear_command * self.VELOCITY_SCALE,
            self._angular_command * self.VELOCITY_SCALE,
        )

    @staticmethod
    def _approach(current: float, target: float, step: float) -> float:
        if abs(target - current) <= step:
            return target
        return current + (step if target > current else -step)

    def _wheel_velocities(self, linear: float, angular: float) -> tuple[float, float]:
        left = linear - angular * self.WHEEL_SEPARATION / 2
        right = linear + angular * self.WHEEL_SEPARATION / 2
        scale = self.MOTOR_GEAR_RATIO / self.WHEEL_RADIUS
        return left * scale, right * scale

    @staticmethod
    def _wheel_status(timestamp: float, position: float, velocity: float) -> dict:
        moving = abs(velocity) > 0.05
        return {
            "timestamp": timestamp,
            "pos": position,
            "vel": velocity,
            "near_vel_setpoint": not moving,
            "is_moving": moving,
            "is_moving_filtered": moving,
            "is_mg_moving": moving,
        }


class Head:
    POSES = {
        "ahead": (0.0, 0.0),
        "tool": (-1.57, -0.79),
        "wheels": (0.0, -1.50),
        "back": (-3.14, 0.0),
    }

    def __init__(self, robot: "Robot") -> None:
        self.head_pan = RevoluteJoint(
            robot,
            Actuators.head_pan,
            default_velocity=2.0,
            max_velocity=2.0,
        )
        self.head_tilt = RevoluteJoint(
            robot,
            Actuators.head_tilt,
            default_velocity=2.0,
            max_velocity=2.0,
        )
        self.joints = ["head_pan", "head_tilt"]

    def get_joint(self, name: str) -> RevoluteJoint:
        if name not in self.joints:
            raise KeyError(f"Unknown head joint: {name}")
        return getattr(self, name)

    def move_to(self, name: str, position_rad: float, v_r: float | None = None) -> None:
        self.get_joint(name).move_to(position_rad, v_r)

    def move_by(self, name: str, angle_rad: float, v_r: float | None = None) -> None:
        self.get_joint(name).move_by(angle_rad, v_r)

    def pose(self, name: str) -> None:
        if name not in self.POSES:
            raise KeyError(f"Unknown head pose: {name}")
        pan, tilt = self.POSES[name]
        self.head_pan.move_to(pan)
        self.head_tilt.move_to(tilt)


class EndOfArm:
    def __init__(self, robot: "Robot") -> None:
        self.wrist_pitch = RevoluteJoint(
            robot,
            Actuators.wrist_pitch,
            default_velocity=0.8,
            max_velocity=0.8,
        )
        self.wrist_roll = RevoluteJoint(
            robot,
            Actuators.wrist_roll,
            default_velocity=1.2,
            max_velocity=1.2,
        )
        self.wrist_yaw = RevoluteJoint(
            robot,
            Actuators.wrist_yaw,
            default_velocity=1.2,
            max_velocity=1.2,
            correction_gain=2.0,
            max_correction=0.2,
        )
        self.stretch_gripper = GripperJoint(robot)
        self.joints = {
            "wrist_pitch": self.wrist_pitch,
            "wrist_roll": self.wrist_roll,
            "wrist_yaw": self.wrist_yaw,
            "stretch_gripper": self.stretch_gripper,
        }

    def get_joint(self, name: str) -> RevoluteJoint | GripperJoint:
        if name not in self.joints:
            raise KeyError(f"Unknown end-of-arm joint: {name}")
        return self.joints[name]

    def move_to(self, name: str, position_rad: float, v_r: float | None = None) -> None:
        self.get_joint(name).move_to(position_rad, v_r)

    def move_by(self, name: str, angle_rad: float, v_r: float | None = None) -> None:
        self.get_joint(name).move_by(angle_rad, v_r)


class Robot:
    def __init__(self) -> None:
        self._sim = StretchMujocoSimulator()
        self._waiting_for: set[Joint | Base] = set()
        self._motion_lock = threading.Lock()
        self._command_complete = threading.Event()
        self._stop_controller = threading.Event()
        self._controller_thread: threading.Thread | None = None
        self.base = Base(self)
        self.lift = PrismaticJoint(self, Actuators.lift, 0.11, 0.15)
        self.arm = PrismaticJoint(
            self,
            Actuators.arm,
            0.14,
            0.2,
            limits=(0.0, 0.52),
        )
        self.head = Head(self)
        self.end_of_arm = EndOfArm(self)
        self._joints = (
            self.lift,
            self.arm,
            self.head.head_pan,
            self.head.head_tilt,
            self.end_of_arm.wrist_pitch,
            self.end_of_arm.wrist_roll,
            self.end_of_arm.wrist_yaw,
            self.end_of_arm.stretch_gripper,
        )

    def startup(self) -> bool:
        self._sim.start(headless=False)
        if not self._sim.is_running():
            return False

        status = self._sim.pull_status()
        for joint in self._joints:
            joint._startup(status)
        self.base._startup()

        self._controller_thread = threading.Thread(
            target=self._run_joint_controller,
            daemon=True,
        )
        self._controller_thread.start()
        return True

    def is_homed(self) -> bool:
        """Compatibility no-op; the simulated robot is always homed."""
        return True

    def home(self) -> None:
        """Compatibility no-op; the simulated robot is always homed."""

    def stow(self) -> None:
        """Compatibility no-op; the simulated robot has no stow position."""
        # TODO: Once trajectories are implemented, this could move the robot to a stowed position.

    def enable_collision_mgmt(self) -> None:
        """Compatibility no-op; the simulated robot has no collision management."""

    def disable_collision_mgmt(self) -> None:
        """Compatibility no-op; the simulated robot has no collision management."""

    def get_status(self) -> dict[str, dict]:
        return {
            "base": self.base.status,
            "arm": self.arm.status,
            "lift": self.lift.status,
            "head": {
                "head_pan": self.head.head_pan.status,
                "head_tilt": self.head.head_tilt.status,
            },
            "end_of_arm": {
                name: joint.status
                for name, joint in self.end_of_arm.joints.items()
            },
        }

    def push_command(self) -> None:
        with self._motion_lock:
            commanded = {
                joint
                for joint in self._joints
                if joint.requires_push and joint._push_command()
            }
            if self.base._push_command():
                commanded.add(self.base)
            if not commanded:
                return
            self._waiting_for.update(commanded)
            self._command_complete.clear()

    def _activate_joint(self, joint: Joint) -> None:
        with self._motion_lock:
            if joint._push_command():
                self._waiting_for.add(joint)
                self._command_complete.clear()

    def _run_joint_controller(self) -> None:
        status = self._sim.pull_status()
        previous_time = status.time

        while self._sim.is_running() and not self._stop_controller.is_set():
            status = self._sim.pull_status()
            elapsed = max(status.time - previous_time, 0.0)
            previous_time = status.time

            with self._motion_lock:
                if self.base._update(status, elapsed):
                    self._waiting_for.discard(self.base)
                for joint in self._joints:
                    if joint._update(status, elapsed):
                        self._waiting_for.discard(joint)
                if not self._waiting_for:
                    self._command_complete.set()

            time.sleep(0.02)

    def wait_command(self) -> bool:
        while self._sim.is_running():
            if self._command_complete.wait(timeout=0.1):
                return True
        return False

    def stop(self) -> None:
        self._stop_controller.set()
        if self._controller_thread is not None:
            self._controller_thread.join()
        self._sim.stop()
