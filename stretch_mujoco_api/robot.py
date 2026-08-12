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
        self._robot._pending_motion = (destination, abs(at))

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
    def __init__(self) -> None:
        self._sim = StretchMujocoSimulator()
        self._pending_motion: tuple[float, float] | None = None
        self._motion_thread: threading.Thread | None = None
        self._cancel_motion = threading.Event()
        self._motion_succeeded = True
        self.lift = Lift(self)

    def startup(self) -> bool:
        self._sim.start(headless=False)
        return self._sim.is_running()

    def enable_collision_mgmt(self) -> None:
        """Compatibility no-op; MuJoCo handles contacts directly."""

    def push_command(self) -> None:
        if self._pending_motion is None:
            return

        self._cancel_active_motion()
        destination, velocity = self._pending_motion
        self._pending_motion = None
        self._cancel_motion.clear()
        self._motion_succeeded = False
        self._motion_thread = threading.Thread(
            target=self._run_lift_motion,
            args=(destination, velocity),
            daemon=True,
        )
        self._motion_thread.start()

    def _run_lift_motion(self, destination: float, velocity: float) -> None:
        status = self._sim.pull_status()
        target = status.lift.pos
        previous_time = status.time

        while self._sim.is_running() and not self._cancel_motion.is_set():
            status = self._sim.pull_status()
            elapsed = status.time - previous_time
            previous_time = status.time
            remaining = destination - target

            if velocity == 0 or abs(remaining) <= velocity * elapsed:
                target = destination
            else:
                target += (velocity * elapsed) if remaining > 0 else -(velocity * elapsed)

            self._sim.move_to(Actuators.lift, target)

            if target == destination:
                self._motion_succeeded = self._sim.wait_until_at_setpoint(Actuators.lift)
                return

            time.sleep(0.02)

    def wait_command(self) -> bool:
        if self._motion_thread is not None:
            self._motion_thread.join()
        return self._motion_succeeded

    def _cancel_active_motion(self) -> None:
        if self._motion_thread is None or not self._motion_thread.is_alive():
            return
        self._cancel_motion.set()
        self._motion_thread.join()

    def stop(self) -> None:
        self._cancel_active_motion()
        self._sim.stop()
