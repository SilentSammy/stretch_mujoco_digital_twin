"""A small ``stretch_body.robot``-compatible facade for MuJoCo."""

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
            correction_gain=0.0,
            max_correction=0.0,
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


class Robot:
    def __init__(self) -> None:
        self._sim = StretchMujocoSimulator()
        self._waiting_for: set[Joint] = set()
        self._motion_lock = threading.Lock()
        self._command_complete = threading.Event()
        self._stop_controller = threading.Event()
        self._controller_thread: threading.Thread | None = None
        self.lift = PrismaticJoint(self, Actuators.lift, 0.11, 0.15)
        self.arm = PrismaticJoint(
            self,
            Actuators.arm,
            0.14,
            0.2,
            limits=(0.0, 0.52),
        )
        self.head = Head(self)
        self._joints = (
            self.lift,
            self.arm,
            self.head.head_pan,
            self.head.head_tilt,
        )

    def startup(self) -> bool:
        self._sim.start(headless=False)
        if not self._sim.is_running():
            return False

        status = self._sim.pull_status()
        for joint in self._joints:
            joint._startup(status)

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
        # TODO: Add keys: base, end_of_arm
        return {
            "arm": self.arm.status,
            "lift": self.lift.status,
            "head": {
                "head_pan": self.head.head_pan.status,
                "head_tilt": self.head.head_tilt.status,
            },
        }

    def push_command(self) -> None:
        with self._motion_lock:
            commanded = {
                joint
                for joint in self._joints
                if joint.requires_push and joint._push_command()
            }
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
