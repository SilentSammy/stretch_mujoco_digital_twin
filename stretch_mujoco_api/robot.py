"""A small ``stretch_body.robot``-compatible facade for MuJoCo."""

import threading
import time

from stretch_mujoco.enums.actuators import Actuators
from stretch_mujoco.stretch_mujoco_simulator import StretchMujocoSimulator


class PrismaticJoint:
    DEFAULT_VELOCITY = 0.1
    LIMITS: tuple[float, float] | None = None

    def __init__(self, robot: "Robot", actuator: Actuators) -> None:
        self._robot = robot
        self._actuator = actuator

    @property
    def status(self) -> dict[str, float]:
        status = self._robot._sim.pull_status()
        return {
            "pos": self._actuator.get_position(status),
            "vel": self._actuator.get_velocity(status),
        }

    @property
    def limits(self) -> tuple[float, float]:
        return self.LIMITS or self._robot._sim.pull_joint_limits()[self._actuator]

    @property
    def total_range(self) -> float:
        lower, upper = self.limits
        return float(upper - lower)

    def _move(self, to: float, at: float) -> None:
        lower, upper = self.limits
        destination = min(max(to, lower), upper)
        self._robot._pending_motions[self._actuator] = destination, abs(at)

    def move_to(self, position_m: float) -> None:
        self._move(position_m, self.DEFAULT_VELOCITY)

    def move_by(self, distance_m: float) -> None:
        self._move(self.status["pos"] + distance_m, self.DEFAULT_VELOCITY)

    def set_velocity(self, velocity_m: float) -> None:
        lower, upper = self.limits
        destination = upper if velocity_m > 0 else lower
        if velocity_m == 0:
            destination = self.status["pos"]
        self._move(destination, velocity_m)


class Lift(PrismaticJoint):
    def __init__(self, robot: "Robot") -> None:
        super().__init__(robot, Actuators.lift)


class Arm(PrismaticJoint):
    LIMITS = (0.0, 0.52)

    def __init__(self, robot: "Robot") -> None:
        super().__init__(robot, Actuators.arm)


class Robot:
    POSITION_TOLERANCE = 0.005
    CORRECTION_GAIN = 20.0
    MAX_CORRECTION = 0.015

    def __init__(self) -> None:
        self._sim = StretchMujocoSimulator()
        self._pending_motions: dict[Actuators, tuple[float, float]] = {}
        self._motions: dict[Actuators, tuple[float, float]] = {}
        self._desired_positions: dict[Actuators, float] = {}
        self._waiting_for: set[Actuators] = set()
        self._motion_lock = threading.Lock()
        self._command_complete = threading.Event()
        self._stop_controller = threading.Event()
        self._controller_thread: threading.Thread | None = None
        self.lift = Lift(self)
        self.arm = Arm(self)

    def startup(self) -> bool:
        self._sim.start(headless=False)
        if not self._sim.is_running():
            return False

        status = self._sim.pull_status()
        for joint in (self.lift, self.arm):
            position = joint._actuator.get_position(status)
            self._motions[joint._actuator] = position, 0.0
            self._desired_positions[joint._actuator] = position

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
        if not self._pending_motions:
            return

        with self._motion_lock:
            self._motions.update(self._pending_motions)
            self._waiting_for = set(self._pending_motions)
            self._pending_motions.clear()
            self._command_complete.clear()

    def _run_joint_controller(self) -> None:
        status = self._sim.pull_status()
        previous_time = status.time
        limits = {joint._actuator: joint.limits for joint in (self.lift, self.arm)}

        while self._sim.is_running() and not self._stop_controller.is_set():
            status = self._sim.pull_status()
            elapsed = max(status.time - previous_time, 0.0)
            previous_time = status.time

            with self._motion_lock:
                motions = self._motions.copy()

            for actuator, motion in motions.items():
                destination, velocity = motion
                desired_position = self._desired_positions[actuator]
                remaining = destination - desired_position

                if velocity == 0 or abs(remaining) <= velocity * elapsed:
                    desired_position = destination
                else:
                    desired_position += velocity * elapsed * (1 if remaining > 0 else -1)

                self._desired_positions[actuator] = desired_position
                actual_position = actuator.get_position(status)
                error = desired_position - actual_position
                correction = min(
                    max(self.CORRECTION_GAIN * error, -self.MAX_CORRECTION),
                    self.MAX_CORRECTION,
                )
                lower, upper = limits[actuator]
                actuator_target = min(max(desired_position + correction, lower), upper)
                self._sim.move_to(actuator, actuator_target)

                if (
                    desired_position == destination
                    and abs(destination - actual_position) <= self.POSITION_TOLERANCE
                ):
                    with self._motion_lock:
                        self._waiting_for.discard(actuator)
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
