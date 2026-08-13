"""A small ``stretch_body.robot``-compatible facade for MuJoCo."""

import threading
import time

from stretch_mujoco.enums.actuators import Actuators
from stretch_mujoco.stretch_mujoco_simulator import StretchMujocoSimulator


class PrismaticJoint:
    def __init__(
        self,
        robot: "Robot",
        actuator: Actuators,
        default_velocity: float,
        max_velocity: float,
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
        self._limits = limits
        self.position_tolerance = position_tolerance
        self.velocity_tolerance = velocity_tolerance
        self.correction_gain = correction_gain
        self.max_correction = max_correction
        self._pending_motion: tuple[float, float] | None = None
        self._motion: tuple[float, float] | None = None
        self._desired_position = 0.0

    @property
    def status(self) -> dict[str, float]:
        status = self._robot._sim.pull_status()
        return {
            "pos": self._actuator.get_position(status),
            "vel": self._actuator.get_velocity(status),
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

    def move_to(self, position_m: float) -> None:
        self._move(position_m, self.default_velocity)

    def move_by(self, distance_m: float) -> None:
        self._move(self.status["pos"] + distance_m, self.default_velocity)

    def set_velocity(self, velocity_m: float) -> None:
        lower, upper = self.limits
        destination = upper if velocity_m > 0 else lower
        if velocity_m == 0:
            destination = self.status["pos"]
        self._move(destination, velocity_m)

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


class Robot:
    def __init__(self) -> None:
        self._sim = StretchMujocoSimulator()
        self._waiting_for: set[PrismaticJoint] = set()
        self._motion_lock = threading.Lock()
        self._command_complete = threading.Event()
        self._stop_controller = threading.Event()
        self._controller_thread: threading.Thread | None = None
        self.lift = PrismaticJoint(self, Actuators.lift, 0.11, 0.15)
        self.arm = PrismaticJoint(self, Actuators.arm, 0.14, 0.2, limits=(0.0, 0.52))
        self._joints = (self.lift, self.arm)

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

    def enable_collision_mgmt(self) -> None:
        """Compatibility no-op; the simulated robot has no collision management."""

    def disable_collision_mgmt(self) -> None:
        """Compatibility no-op; the simulated robot has no collision management."""

    def push_command(self) -> None:
        with self._motion_lock:
            commanded = {joint for joint in self._joints if joint._push_command()}
            if not commanded:
                return
            self._waiting_for = commanded
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
