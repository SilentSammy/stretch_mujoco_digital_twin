"""A small ``stretch_body.robot``-compatible facade for MuJoCo."""

import threading
import time

from stretch_mujoco.enums.actuators import Actuators
from stretch_mujoco.stretch_mujoco_simulator import StretchMujocoSimulator


class Lift:
    DEFAULT_VELOCITY = 0.1

    def __init__(self, robot: "Robot") -> None:
        self._robot = robot

    @property
    def status(self) -> dict[str, float]:
        status = self._robot._sim.pull_status().lift
        return {"pos": status.pos, "vel": status.vel}

    @property
    def limits(self) -> tuple[float, float]:
        return self._robot._sim.pull_joint_limits()[Actuators.lift]

    @property
    def total_range(self) -> float:
        lower, upper = self.limits
        return float(upper - lower)

    def _move(self, to: float, at: float) -> None:
        lower, upper = self.limits
        destination = min(max(to, lower), upper)
        self._robot._pending_motion = destination, abs(at)

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


class Robot:
    POSITION_TOLERANCE = 0.005
    CORRECTION_GAIN = 20.0
    MAX_CORRECTION = 0.015

    def __init__(self) -> None:
        self._sim = StretchMujocoSimulator()
        self._pending_motion: tuple[float, float] | None = None
        self._motion: tuple[float, float] | None = None
        self._motion_lock = threading.Lock()
        self._command_complete = threading.Event()
        self._stop_controller = threading.Event()
        self._controller_thread: threading.Thread | None = None
        self.lift = Lift(self)

    def startup(self) -> bool:
        self._sim.start(headless=False)
        if not self._sim.is_running():
            return False

        position = self.lift.status["pos"]
        self._motion = position, 0.0
        self._controller_thread = threading.Thread(
            target=self._run_lift_controller,
            daemon=True,
        )
        self._controller_thread.start()
        return True

    def enable_collision_mgmt(self) -> None:
        """Compatibility no-op; MuJoCo handles contacts directly."""

    def push_command(self) -> None:
        if self._pending_motion is None:
            return

        with self._motion_lock:
            self._motion = self._pending_motion
            self._pending_motion = None
            self._command_complete.clear()

    def _run_lift_controller(self) -> None:
        status = self._sim.pull_status()
        desired_position = status.lift.pos
        previous_time = status.time
        lower, upper = self.lift.limits

        while self._sim.is_running() and not self._stop_controller.is_set():
            status = self._sim.pull_status()
            elapsed = max(status.time - previous_time, 0.0)
            previous_time = status.time

            with self._motion_lock:
                motion = self._motion

            if motion is not None:
                destination, velocity = motion
                remaining = destination - desired_position

                if velocity == 0 or abs(remaining) <= velocity * elapsed:
                    desired_position = destination
                else:
                    desired_position += velocity * elapsed * (1 if remaining > 0 else -1)

                error = desired_position - status.lift.pos
                correction = min(
                    max(self.CORRECTION_GAIN * error, -self.MAX_CORRECTION),
                    self.MAX_CORRECTION,
                )
                actuator_target = min(max(desired_position + correction, lower), upper)
                self._sim.move_to(Actuators.lift, actuator_target)

                if (
                    desired_position == destination
                    and abs(destination - status.lift.pos) <= self.POSITION_TOLERANCE
                ):
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
