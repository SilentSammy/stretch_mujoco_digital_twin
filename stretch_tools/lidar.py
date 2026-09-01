"""Helpers for processing Stretch lidar scans."""

import cv2
import numpy as np


SIM_MAST_PROFILE = [
    (250, 0.152),
    (249, 0.145),
    (248, 0.139),
    (247, 0.133),
    (246, 0.130),
    (245, 0.129),
    (244, 0.129),
    (243, 0.130),
    (242, 0.131),
    (241, 0.133),
    (240, 0.134),
    (239, 0.135),
    (238, 0.137),
    (237, 0.138),
    (236, 0.140),
    (235, 0.142),
    (234, 0.143),
]

REAL_MAST_PROFILE = [
    (233, 0.135),
    (234, 0.135),
    (235, 0.134),
    (236, 0.133),
    (237, 0.132),
    (238, 0.130),
    (239, 0.129),
    (240, 0.128),
    (241, 0.127),
    (242, 0.127),
    (243, 0.126),
    (244, 0.126),
]

MAST_PROFILE = SIM_MAST_PROFILE + REAL_MAST_PROFILE


def filter_mast_points(scan, angle_margin=2.0, distance_margin=0.03):
    """Remove lidar returns matching the lift mast profile."""

    def is_mast_point(angle, distance_mm):
        distance = distance_mm / 1000
        near_mast_angle = any(
            abs((angle - mast_angle + 180) % 360 - 180) <= angle_margin
            for mast_angle, _ in MAST_PROFILE
        )
        mast_limit = max(distance for _, distance in MAST_PROFILE) + distance_margin
        return near_mast_angle and distance <= mast_limit

    return [
        measurement
        for measurement in scan
        if not is_mast_point(measurement[1], measurement[2])
    ]


class LidarPlotter:
    IMG_SIZE = 800
    PIXELS_PER_METER = 55
    MAX_RANGE = 10.0

    def __init__(self, window_name="LiDAR 2D View"):
        self.window_name = window_name
        self._center = (self.IMG_SIZE // 2, self.IMG_SIZE // 2)

    def update(self, scan):
        if not scan:
            return

        angles = np.radians([-angle for _, angle, _ in scan])
        ranges = np.array([distance for _, _, distance in scan]) / 1000
        cv2.imshow(self.window_name, self._render(angles, ranges))
        cv2.waitKey(1)

    def close(self):
        cv2.destroyWindow(self.window_name)

    def _render(self, angles, ranges):
        size = self.IMG_SIZE
        cx, cy = self._center
        ppm = self.PIXELS_PER_METER
        canvas = np.full((size, size, 3), 18, dtype=np.uint8)

        for metres in range(-int(self.MAX_RANGE), int(self.MAX_RANGE) + 1):
            offset = int(metres * ppm)
            x = cx + offset
            y = cy + offset
            if 0 <= x < size:
                cv2.line(canvas, (x, 0), (x, size), (45, 45, 45), 1)
            if 0 <= y < size:
                cv2.line(canvas, (0, y), (size, y), (45, 45, 45), 1)

        cv2.line(canvas, (cx, 0), (cx, size), (120, 120, 120), 1)
        cv2.line(canvas, (0, cy), (size, cy), (120, 120, 120), 1)

        for metres in range(1, int(self.MAX_RANGE) + 1):
            radius = int(metres * ppm)
            cv2.circle(canvas, (cx, cy), radius, (35, 35, 35), 1)
            cv2.putText(
                canvas,
                f"{metres}m",
                (cx + radius + 4, cy - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (110, 110, 110),
                1,
            )

        px = (cx - ranges * np.sin(angles) * ppm).astype(int)
        py = (cy - ranges * np.cos(angles) * ppm).astype(int)
        normalized = np.clip(ranges / self.MAX_RANGE, 0, 1)
        red = (255 * (1 - normalized)).astype(np.uint8)
        green = (180 * normalized).astype(np.uint8)
        blue = (255 * normalized).astype(np.uint8)

        for x, y, b, g, r in zip(px, py, blue, green, red):
            if 0 <= x < size and 0 <= y < size:
                cv2.circle(canvas, (int(x), int(y)), 2, (int(b), int(g), int(r)), -1)

        cv2.circle(canvas, (cx, cy), 7, (230, 230, 230), -1)
        front = (cx, cy - ppm)
        cv2.arrowedLine(canvas, (cx, cy), front, (255, 255, 255), 2, tipLength=0.25)
        cv2.putText(
            canvas,
            "+X",
            (front[0] + 8, front[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )
        return canvas
