"""A small ``stretch_body.robot``-compatible facade for the MuJoCo simulator."""

from collections.abc import Callable

from stretch_mujoco.enums.actuators import Actuators
from stretch_mujoco.stretch_mujoco_simulator import StretchMujocoSimulator


class Lift:
    """Lift API shaped like ``stretch_body.robot.Robot.lift``."""

    def __init__(self, robot: "Robot") -> None:
        self._robot = robot

    @property
    def status(self) -> dict[str, float]:
        status = self._robot._sim.pull_status().lift
        return {"pos": status.pos, "vel": status.vel}

    @property
    def total_range(self) -> float:
        lower, upper = self._robot._sim.pull_joint_limits()[Actuators.lift]
        return float(upper - lower)

    def move_to(self, position_m: float) -> None:
        self._robot._queue_command(
            lambda: self._robot._sim.move_to(Actuators.lift, position_m),
            lambda: self._robot._sim.wait_until_at_setpoint(Actuators.lift),
        )

    def move_by(self, distance_m: float) -> None:
        self._robot._queue_command(
            lambda: self._robot._sim.move_by(Actuators.lift, distance_m),
            lambda: self._robot._sim.wait_while_is_moving(Actuators.lift),
        )


class Robot:
    """Minimal simulated counterpart of ``stretch_body.robot.Robot``."""

    def __init__(self) -> None:
        self._sim = StretchMujocoSimulator()
        self._pending_commands: list[tuple[Callable[[], None], Callable[[], bool]]] = []
        self._command_waiters: list[Callable[[], bool]] = []
        self.lift = Lift(self)

    def startup(self) -> bool:
        self._sim.start(headless=False)
        return self._sim.is_running()

    def enable_collision_mgmt(self) -> None:
        """Compatibility no-op; MuJoCo handles physical contacts directly."""

    def _queue_command(self, command: Callable[[], None], waiter: Callable[[], bool]) -> None:
        self._pending_commands.append((command, waiter))

    def push_command(self) -> None:
        pending_commands = self._pending_commands
        self._pending_commands = []
        self._command_waiters = []

        for command, waiter in pending_commands:
            command()
            self._command_waiters.append(waiter)

    def wait_command(self) -> bool:
        waiters = self._command_waiters
        self._command_waiters = []
        results = [waiter() for waiter in waiters]
        return all(results)

    def stop(self) -> None:
        self._sim.stop()
