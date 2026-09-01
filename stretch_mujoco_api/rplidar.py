"""Small ``rplidar``-compatible adapter for the simulated base lidar."""

import time

import numpy as np

from stretch_mujoco.enums.stretch_sensors import StretchSensors


_sim = None


def _set_simulator(simulator):
    global _sim
    _sim = simulator


def _clear_simulator(simulator):
    global _sim
    if _sim is simulator:
        _sim = None


class RPLidar:
    def __init__(self, _port, *_, **__):
        self._connected = True
        self._scanning = True

    def iter_scans(self, scan_type="normal", max_buf_meas=500, min_len=5):
        del scan_type, max_buf_meas
        previous_time = None

        while self._connected and self._scanning and _sim is not None and _sim.is_running():
            sensor_data = _sim.pull_sensor_data()
            if sensor_data.time == previous_time:
                time.sleep(0.001)
                continue
            previous_time = sensor_data.time

            try:
                distances = sensor_data.get_data(StretchSensors.base_lidar)
            except ValueError:
                time.sleep(0.01)
                continue

            scan = [
                (15, float((180 - angle) % 360), float(distance * 1000))
                for angle, distance in enumerate(np.asarray(distances))
                if np.isfinite(distance) and distance > 0
            ]
            if len(scan) >= min_len:
                yield scan

    def stop(self):
        self._scanning = False

    def stop_motor(self):
        pass

    def disconnect(self):
        self._connected = False
