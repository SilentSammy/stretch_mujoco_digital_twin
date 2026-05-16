"""
Simple test: robot control with configurable camera displays,
object control, LiDAR readings, and LiDAR 2D ray visualization.
"""

from stretch_toolkit import (
    controller,
    teleop,
    BACKEND_NAME,
    HEAD_CAMERA,
    WRIST_CAMERA,
    NAVIGATION_CAMERA,
    HEAD_RGB_CAMERA,
    HEAD_DEPTH_CAMERA,
    WRIST_RGB_CAMERA,
    WRIST_DEPTH_CAMERA,
)

from stretch_mujoco.enums.stretch_sensors import StretchSensors

import stretch_toolkit.input as inp
import time
import cv2
import numpy as np


print(f"\n=== Running on {BACKEND_NAME} backend ===\n")


# Configure which cameras to display
# Set first value to True/False to enable/disable each feed
CAMERA_DISPLAYS = [
    (False, "Head RGB", HEAD_RGB_CAMERA),
    (False, "Head Depth", HEAD_DEPTH_CAMERA),
    (False, "Wrist RGB", WRIST_RGB_CAMERA),
    (False, "Wrist Depth", WRIST_DEPTH_CAMERA),
    (False, "Navigation", NAVIGATION_CAMERA),
]


def update_camera_windows(open_windows):
    """
    Updates all enabled camera windows and closes disabled ones.

    Args:
        open_windows: set of currently opened OpenCV window names.

    Returns:
        Updated set of opened window names.
    """
    active_windows = set()

    for enabled, window_name, camera in CAMERA_DISPLAYS:
        if enabled and camera is not None:
            active_windows.add(window_name)

            try:
                frame = camera.get_frame()

                # Colorize depth frames for visualization
                if "Depth" in window_name and frame is not None:
                    frame_vis = cv2.normalize(
                        frame,
                        None,
                        0,
                        255,
                        cv2.NORM_MINMAX,
                        dtype=cv2.CV_8U,
                    )
                    frame = cv2.applyColorMap(frame_vis, cv2.COLORMAP_JET)

                if frame is not None:
                    cv2.imshow(window_name, frame)
                    open_windows.add(window_name)

            except Exception:
                # Silently skip camera errors
                pass

    # Close windows that should no longer be displayed
    windows_to_close = open_windows - active_windows

    for window_name in windows_to_close:
        try:
            cv2.destroyWindow(window_name)
        except Exception:
            pass

    return active_windows


def get_lidar_ranges():
    """
    Reads and sanitizes LiDAR ranges from the MuJoCo simulator.

    Returns:
        np.ndarray with LiDAR distances in meters.
        Invalid or out-of-range readings are converted to np.inf.
    """
    try:
        sensor_data = controller.sim.pull_sensor_data()
        ranges = sensor_data.get_data(StretchSensors.base_lidar)

        ranges = np.asarray(ranges, dtype=float).reshape(-1)

        # Expected LiDAR limits
        range_min = 0.02
        range_max = 10.0

        # Clean bad values:
        # - NaN
        # - negative values
        # - values greater than the XML cutoff
        invalid_mask = (
            np.isnan(ranges)
            | (ranges < range_min)
            | (ranges > range_max)
        )

        ranges[invalid_mask] = np.inf

        return ranges

    except Exception as e:
        print(f"[LiDAR ERROR] Could not read LiDAR data: {e}")
        return None


def print_lidar_summary(ranges):
    """
    Prints a compact summary of the sanitized LiDAR scan.
    """
    if ranges is None or len(ranges) == 0:
        print("[LiDAR] No data.")
        return

    finite_ranges = ranges[np.isfinite(ranges)]

    if len(finite_ranges) == 0:
        print(f"[LiDAR] rays={len(ranges)} | all readings are inf/no-hit")
        return

    min_distance = float(np.min(finite_ranges))
    max_distance = float(np.max(finite_ranges))
    mean_distance = float(np.mean(finite_ranges))

    # Print representative ray samples
    sample_count = min(12, len(ranges))
    sample_indices = np.linspace(0, len(ranges) - 1, sample_count, dtype=int)

    sample_values = []
    for idx in sample_indices:
        value = ranges[idx]

        if np.isinf(value):
            sample_values.append("inf")
        else:
            sample_values.append(f"{value:.2f}")

    print(
        "[LiDAR] "
        f"rays={len(ranges)} | "
        f"valid={len(finite_ranges)} | "
        f"min={min_distance:.3f} m | "
        f"mean={mean_distance:.3f} m | "
        f"max={max_distance:.3f} m | "
        f"samples={sample_values}"
    )


def show_lidar_view(ranges, window_name="LiDAR 2D View"):
    """
    RViz-like top-down 2D visualization for LiDAR/LaserScan.

    Convention used in the viewer:
        +X/front  -> upward on screen
        +Y/left   -> left on screen

    Args:
        ranges: np.ndarray with LiDAR distances in meters.
        window_name: OpenCV window name.
    """
    if ranges is None or len(ranges) == 0:
        return

    # ----------------------------
    # Viewer configuration
    # ----------------------------
    img_size = 800
    center = (img_size // 2, img_size // 2)

    max_range = 10.0
    pixels_per_meter = 55

    # Change this if the scan appears rotated.
    # Good values to test:
    # 0.0
    # np.pi / 2
    # -np.pi / 2
    # np.pi
    angle_offset = 0.0

    # If the scan looks mirrored, change this to True.
    reverse_scan_order = False

    # If True, draw faint rays from robot to hit points.
    draw_hit_rays = False

    # ----------------------------
    # Prepare canvas
    # ----------------------------
    canvas = np.zeros((img_size, img_size, 3), dtype=np.uint8)

    # Background color similar to RViz dark theme
    canvas[:, :] = (18, 18, 18)

    # ----------------------------
    # Draw metric grid
    # ----------------------------
    grid_color = (45, 45, 45)
    axis_color = (120, 120, 120)

    meters_visible = int(max_range)

    for meter in range(-meters_visible, meters_visible + 1):
        offset = int(meter * pixels_per_meter)

        # Vertical grid lines
        x = center[0] + offset
        if 0 <= x < img_size:
            cv2.line(canvas, (x, 0), (x, img_size), grid_color, 1)

        # Horizontal grid lines
        y = center[1] + offset
        if 0 <= y < img_size:
            cv2.line(canvas, (0, y), (img_size, y), grid_color, 1)

    # Main axes
    cv2.line(canvas, (center[0], 0), (center[0], img_size), axis_color, 1)
    cv2.line(canvas, (0, center[1]), (img_size, center[1]), axis_color, 1)

    # Range circles
    for meter in range(1, int(max_range) + 1):
        radius = int(meter * pixels_per_meter)
        cv2.circle(canvas, center, radius, (35, 35, 35), 1)

        cv2.putText(
            canvas,
            f"{meter}m",
            (center[0] + radius + 5, center[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (110, 110, 110),
            1,
        )

    # ----------------------------
    # Draw robot origin
    # ----------------------------
    cv2.circle(canvas, center, 7, (230, 230, 230), -1)

    cv2.putText(
        canvas,
        "base_lidar",
        (center[0] + 10, center[1] + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (230, 230, 230),
        1,
    )

    # +X/front arrow upward
    front_len = int(1.0 * pixels_per_meter)
    front_end = (center[0], center[1] - front_len)

    cv2.arrowedLine(
        canvas,
        center,
        front_end,
        (255, 255, 255),
        2,
        tipLength=0.25,
    )

    cv2.putText(
        canvas,
        "+X/front",
        (front_end[0] + 8, front_end[1] - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
    )

    # +Y/left arrow
    y_len = int(0.8 * pixels_per_meter)
    y_end = (center[0] - y_len, center[1])

    cv2.arrowedLine(
        canvas,
        center,
        y_end,
        (180, 180, 180),
        1,
        tipLength=0.25,
    )

    cv2.putText(
        canvas,
        "+Y",
        (y_end[0] - 25, y_end[1] - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (180, 180, 180),
        1,
    )

    # ----------------------------
    # Draw LaserScan points
    # ----------------------------
    ranges_to_draw = ranges.copy()

    if reverse_scan_order:
        ranges_to_draw = ranges_to_draw[::-1]

    n = len(ranges_to_draw)

    valid_count = 0

    for i, raw_range in enumerate(ranges_to_draw):
        if not np.isfinite(raw_range):
            continue

        if raw_range <= 0.02 or raw_range > max_range:
            continue

        r = float(raw_range)

        # MuJoCo replicated sensor order:
        # 360 samples around Z.
        #
        # We map it to a LaserScan-style polar angle.
        # angle = 0 means +X/front.
        angle = (2.0 * np.pi * i / n) + angle_offset

        # Robotics convention:
        # x_robot = forward
        # y_robot = left
        x_robot = r * np.cos(angle)
        y_robot = r * np.sin(angle)

        # Screen convention:
        # screen x grows right
        # screen y grows down
        #
        # We want:
        # +X/front -> up, so screen_y -= x_robot
        # +Y/left  -> left, so screen_x -= y_robot
        x_screen = int(center[0] - y_robot * pixels_per_meter)
        y_screen = int(center[1] - x_robot * pixels_per_meter)

        if not (0 <= x_screen < img_size and 0 <= y_screen < img_size):
            continue

        valid_count += 1

        # Optional faint ray only to actual hit
        if draw_hit_rays:
            cv2.line(
                canvas,
                center,
                (x_screen, y_screen),
                (35, 65, 35),
                1,
            )

        # RViz-like LaserScan hit point
        cv2.circle(
            canvas,
            (x_screen, y_screen),
            2,
            (0, 255, 80),
            -1,
        )

    # ----------------------------
    # Text overlay
    # ----------------------------
    finite_ranges = ranges[np.isfinite(ranges)]
    finite_ranges = finite_ranges[
        (finite_ranges > 0.02) & (finite_ranges <= max_range)
    ]

    if len(finite_ranges) > 0:
        min_r = float(np.min(finite_ranges))
        mean_r = float(np.mean(finite_ranges))
        max_r = float(np.max(finite_ranges))

        status = (
            f"rays={len(ranges)} valid={valid_count} "
            f"min={min_r:.2f}m mean={mean_r:.2f}m max={max_r:.2f}m"
        )
    else:
        status = f"rays={len(ranges)} valid=0"

    cv2.putText(
        canvas,
        status,
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 80),
        1,
    )

    cv2.putText(
        canvas,
        "RViz-like LaserScan view | V to toggle",
        (15, img_size - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (180, 180, 180),
        1,
    )

    cv2.imshow(window_name, canvas)

def teleop_demo():
    """
    Run teleoperation loop with configurable camera displays,
    object control, LiDAR readings and LiDAR ray visualization.
    """

    from stretch_toolkit.input import rising_edge

    print("Teleop with camera views. Use gamepad/keyboard to control.")

    active_cameras = [
        name
        for enabled, name, cam in CAMERA_DISPLAYS
        if enabled and cam is not None
    ]

    print(f"Displaying {len(active_cameras)} camera feeds: {', '.join(active_cameras)}")
    print("Press Ctrl+C to stop\n")

    print("Controls:")
    print("  0             -> Toggle ROBOT CONTROL / OBJECT CONTROL")
    print("  1-5           -> Toggle camera windows")
    print("  L             -> Toggle continuous LiDAR print")
    print("  P             -> Print one LiDAR measurement now")
    print("  V             -> Toggle LiDAR 2D ray visualization")
    print("  Arrow Right   -> Move object continuously in +X")
    print("  Arrow Left    -> Move object continuously in -X")
    print("  Arrow Up      -> Move object continuously in +Z")
    print("  Arrow Down    -> Move object continuously in -Z")
    print("")

    # Track which OpenCV windows are currently open
    open_windows = set()

    # ---------------------------------------------------------
    # OBJECT CONTROL CONFIGURATION
    # ---------------------------------------------------------

    object_control_mode = False

    # This must match the body name in scene.xml:
    # <body name="android_lego" ...>
    object_name = "android_lego"

    # Movement speed in meters per second.
    # Example: 0.12 means 12 cm/s while holding the key.
    object_speed_x = 0.12
    object_speed_z = 0.12

    # Minimum Z limit to avoid pushing the object below the table/floor.
    object_z_min = 0.45

    # ---------------------------------------------------------
    # LiDAR CONFIGURATION
    # ---------------------------------------------------------

    lidar_print_enabled = False
    lidar_view_enabled = False

    # Print LiDAR at 5 Hz when enabled
    lidar_print_period = 0.20
    last_lidar_print_time = 0.0

    # Time tracking for smooth continuous movement
    last_time = time.perf_counter()

    try:
        while True:
            # ---------------------------------------------------------
            # CAMERA TOGGLE CONTROL
            # ---------------------------------------------------------

            for i in range(1, len(CAMERA_DISPLAYS) + 1):
                if rising_edge(str(i)):
                    enabled, name, cam = CAMERA_DISPLAYS[i - 1]
                    CAMERA_DISPLAYS[i - 1] = (not enabled, name, cam)

                    state = "ENABLED" if not enabled else "DISABLED"
                    print(f"\n[{i}. {name}] {state}")

            # ---------------------------------------------------------
            # TIME DELTA
            # ---------------------------------------------------------

            current_time = time.perf_counter()
            dt = current_time - last_time
            last_time = current_time

            # Avoid large jumps if the program freezes momentarily
            dt = min(dt, 0.05)

            # ---------------------------------------------------------
            # MODE TOGGLE
            # ---------------------------------------------------------

            if rising_edge("0"):
                object_control_mode = not object_control_mode

                mode = "OBJECT CONTROL" if object_control_mode else "ROBOT CONTROL"
                print(f"\nMode changed to: {mode}")

                # Stop robot immediately when changing modes
                controller.set_velocities({})

            # ---------------------------------------------------------
            # LiDAR CONTROLS
            # ---------------------------------------------------------

            if rising_edge("l") or rising_edge("L"):
                lidar_print_enabled = not lidar_print_enabled

                state = "ENABLED" if lidar_print_enabled else "DISABLED"
                print(f"\nLiDAR print: {state}")

            if rising_edge("p") or rising_edge("P"):
                ranges = get_lidar_ranges()
                print_lidar_summary(ranges)

            if rising_edge("v") or rising_edge("V"):
                lidar_view_enabled = not lidar_view_enabled

                state = "ENABLED" if lidar_view_enabled else "DISABLED"
                print(f"\nLiDAR 2D view: {state}")

                if not lidar_view_enabled:
                    try:
                        cv2.destroyWindow("LiDAR 2D View")
                    except Exception:
                        pass

            # Continuous LiDAR print
            if lidar_print_enabled:
                if current_time - last_lidar_print_time >= lidar_print_period:
                    ranges = get_lidar_ranges()
                    print_lidar_summary(ranges)
                    last_lidar_print_time = current_time

            # LiDAR 2D visualization
            if lidar_view_enabled:
                ranges = get_lidar_ranges()
                show_lidar_view(ranges)

            # ---------------------------------------------------------
            # OBJECT CONTROL MODE
            # ---------------------------------------------------------

            if object_control_mode:
                # Stop robot while controlling the object
                controller.set_velocities({})

                dx = 0.0
                dz = 0.0

                # Hold arrow keys to move continuously in X
                if inp.is_pressed("Key.right"):
                    dx += object_speed_x * dt

                if inp.is_pressed("Key.left"):
                    dx -= object_speed_x * dt

                # Hold arrow keys to move continuously in Z
                if inp.is_pressed("Key.up"):
                    dz += object_speed_z * dt

                if inp.is_pressed("Key.down"):
                    dz -= object_speed_z * dt

                moved = abs(dx) > 1e-8 or abs(dz) > 1e-8

                if moved:
                    controller.sim.move_object_by(
                        object_name,
                        delta=(dx, 0.0, dz),
                        z_min=object_z_min,
                    )

                    print(
                        f"{object_name}: "
                        f"dx={dx:.4f} m, "
                        f"dz={dz:.4f} m"
                    )

                # Keep camera windows alive while in object-control mode
                open_windows = update_camera_windows(open_windows)

                cv2.waitKey(1)
                time.sleep(1 / 30)
                continue

            # ---------------------------------------------------------
            # NORMAL ROBOT CONTROL MODE
            # ---------------------------------------------------------

            # Get normalized velocities from input devices
            velocities = teleop.get_normalized_velocities()

            # Send to robot, physical or simulated
            controller.set_velocities(velocities)

            # Update camera windows
            open_windows = update_camera_windows(open_windows)

            cv2.waitKey(1)
            time.sleep(1 / 30)

    except KeyboardInterrupt:
        print("\n\nStopping...")

    finally:
        # Stop all motion
        controller.set_velocities({})

        try:
            controller.stop()
        except Exception:
            pass

        cv2.destroyAllWindows()
        print("Demo complete!")


if __name__ == "__main__":
    teleop_demo()