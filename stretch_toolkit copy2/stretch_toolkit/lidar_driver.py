"""Pure-Python RPLidar reader for the physical Stretch robot.

The Stretch mobile base carries a SLAMTEC RPLidar A1 exposed by the Hello Robot
udev rules as ``/dev/hello-lidar``.  This module talks to it directly over the
serial protocol (no ROS required) and continuously maintains the latest full
360-degree scan in a background thread.

The output format is intentionally identical to the simulator's
``get_lidar_ranges()``:

* a 360-element ``numpy`` array of distances in metres,
* indexed so that ray ``0`` points along the robot's +X axis (front) and the
  index increases **counter-clockwise** (ray ``90`` -> +Y / left),
* no-hit / out-of-range rays are ``numpy.inf``.

This matches the MuJoCo scene (360 sites replicated at 1-degree CCW increments
starting along +X) so the same scripts and ``LidarPlotter`` work unchanged on
both backends.
"""

import json
import os
import threading
import time
from pathlib import Path

import numpy as np

# Hello Robot udev symlink for the base RPLidar A1.
DEFAULT_LIDAR_PORT = "/dev/hello-lidar"
DEFAULT_BAUDRATE = 115200

N_RAYS = 360
# RPLidar A1 physical range limits (metres).
DEFAULT_RANGE_MIN = 0.15
DEFAULT_RANGE_MAX = 12.0

# Calibration knobs.  The A1 reports angle increasing clockwise from its own
# zero mark; the simulator/plotter use a counter-clockwise convention with
# 0 at the robot front, so by default we flip direction.  ``angle_offset_deg``
# rotates the whole scan to align ray 0 with the robot's physical front.
DEFAULT_CLOCKWISE = True
DEFAULT_ANGLE_OFFSET_DEG = 0.0

_CONFIG_FILE = Path(__file__).parent / "lidar_config.json"


def _load_or_create_config():
    """Load LiDAR calibration from JSON, creating defaults on first run."""
    defaults = {
        "port": DEFAULT_LIDAR_PORT,
        "baudrate": DEFAULT_BAUDRATE,
        "range_min": DEFAULT_RANGE_MIN,
        "range_max": DEFAULT_RANGE_MAX,
        "clockwise": DEFAULT_CLOCKWISE,
        "angle_offset_deg": DEFAULT_ANGLE_OFFSET_DEG,
        "_comment": (
            "Calibration for the physical RPLidar. 'clockwise' flips the spin "
            "direction; 'angle_offset_deg' rotates the scan so ray 0 points at "
            "the robot's physical front. Adjust these while watching the "
            "LidarPlotter until the scan lines up with reality."
        ),
    }
    if not _CONFIG_FILE.exists():
        try:
            with open(_CONFIG_FILE, "w") as f:
                json.dump(defaults, f, indent=2)
            print(f"[LiDAR] Created default lidar config: {_CONFIG_FILE}")
        except Exception as e:
            print(f"[LiDAR] Could not write default config ({e}); using built-in defaults.")
        return defaults

    try:
        with open(_CONFIG_FILE, "r") as f:
            loaded = json.load(f)
        defaults.update({k: v for k, v in loaded.items() if not k.startswith("_")})
    except Exception as e:
        print(f"[LiDAR] Failed to read {_CONFIG_FILE} ({e}); using built-in defaults.")
    return defaults


class RPLidarReader:
    """Background reader that keeps the latest 360-ray scan available.

    Usage::

        reader = RPLidarReader()
        reader.start()
        ranges = reader.get_ranges()   # 360-element np.ndarray, metres
        reader.stop()
    """

    def __init__(self, port=None, baudrate=None, angle_offset_deg=None,
                 clockwise=None, range_min=None, range_max=None):
        cfg = _load_or_create_config()
        self._port = port if port is not None else cfg["port"]
        self._baudrate = baudrate if baudrate is not None else cfg["baudrate"]
        self._range_min = range_min if range_min is not None else cfg["range_min"]
        self._range_max = range_max if range_max is not None else cfg["range_max"]
        self._clockwise = clockwise if clockwise is not None else cfg["clockwise"]
        self._angle_offset_deg = (
            angle_offset_deg if angle_offset_deg is not None else cfg["angle_offset_deg"]
        )

        self._ranges = np.full(N_RAYS, np.inf, dtype=float)
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._lidar = None
        self._got_scan = threading.Event()
        self.last_error = None

    # ── Lifecycle ─────────────────────────────────────────────────────
    def start(self):
        """Open the serial port and start the background scan thread."""
        if self._running:
            return
        try:
            from rplidar import RPLidar
        except ImportError as e:
            raise ImportError(
                "The 'rplidar' package is required for physical LiDAR support. "
                "Install it with:  pip install rplidar-roboticia"
            ) from e

        port = self._resolve_port()
        if port is None:
            raise IOError(
                f"Could not find the RPLidar serial port. Tried '{self._port}' and "
                "auto-detection of CP210x USB-serial devices found none. "
                "Plug in the LiDAR, or set the correct path in lidar_config.json "
                "(e.g. /dev/ttyUSB0). Available ports can be listed with: "
                "python -m serial.tools.list_ports -v"
            )
        if port != self._port:
            print(f"[LiDAR] '{self._port}' not found; using auto-detected port '{port}'.")
        self._port = port

        self._lidar = RPLidar(self._port, baudrate=self._baudrate)
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, name="RPLidarReader", daemon=True
        )
        self._thread.start()

    def _resolve_port(self):
        """Return a usable serial port, falling back to auto-detection.

        Prefers the configured port if it exists, then any Silicon Labs CP210x
        USB-serial bridge (the chip used by the RPLidar A1), then common
        ``/dev/ttyUSB*`` paths.
        """
        if os.path.exists(self._port):
            return self._port

        try:
            from serial.tools import list_ports
        except ImportError:
            return self._port if os.path.exists(self._port) else None

        ports = list(list_ports.comports())

        # 1. Match the CP210x USB-to-UART bridge by VID:PID or description.
        for p in ports:
            vidpid = f"{p.vid:04x}:{p.pid:04x}" if p.vid and p.pid else ""
            desc = (p.description or "").lower()
            if vidpid == "10c4:ea60" or "cp210" in desc or "silicon labs" in desc:
                return p.device

        # 2. Fall back to the first ttyUSB-style device.
        for p in ports:
            if "ttyusb" in (p.device or "").lower():
                return p.device

        return None

    def wait_for_first_scan(self, timeout: float = 5.0) -> bool:
        """Block until at least one full revolution has been ingested."""
        return self._got_scan.wait(timeout=timeout)

    def stop(self):
        """Stop the thread and release the serial port and motor."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._lidar is not None:
            try:
                self._lidar.stop()
                self._lidar.stop_motor()
                self._lidar.disconnect()
            except Exception:
                pass
            self._lidar = None

    # ── Data access ───────────────────────────────────────────────────
    def get_ranges(self):
        """Return a copy of the latest 360-ray scan (metres, inf = no hit)."""
        with self._lock:
            return self._ranges.copy()

    # ── Internal ──────────────────────────────────────────────────────
    def _loop(self):
        """Continuously read scans, recovering from transient device errors."""
        while self._running:
            try:
                for scan in self._lidar.iter_scans(max_buf_meas=500, min_len=5):
                    if not self._running:
                        break
                    self._ingest(scan)
            except Exception as e:
                self.last_error = str(e)
                # Attempt to recover the device and keep going.
                try:
                    self._lidar.stop()
                    self._lidar.clean_input()
                except Exception:
                    pass
                time.sleep(0.5)

    def _ingest(self, scan):
        """Convert one raw revolution into the robot-frame range array."""
        ranges = np.full(N_RAYS, np.inf, dtype=float)
        for measurement in scan:
            quality, angle, distance = measurement
            if quality <= 0 or distance <= 0:
                continue
            d = distance / 1000.0  # mm -> m
            if d < self._range_min or d > self._range_max:
                continue
            a = -angle if self._clockwise else angle
            a = (a + self._angle_offset_deg) % 360.0
            idx = int(round(a)) % N_RAYS
            # If several measurements fall in the same 1-degree bin, keep the nearest.
            if d < ranges[idx]:
                ranges[idx] = d

        with self._lock:
            self._ranges = ranges
        self._got_scan.set()
