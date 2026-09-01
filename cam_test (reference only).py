import pyrealsense2 as rs
import numpy as np
import cv2

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(
    rs.stream.depth,
    848, 480,
    rs.format.z16,
    10
)

pipeline.start(config)

# Shared filters
spatial = rs.spatial_filter()
temporal = rs.temporal_filter()

# Hole filling filters, one per mode
hole_left = rs.hole_filling_filter()
hole_left.set_option(rs.option.holes_fill, 0)

hole_far = rs.hole_filling_filter()
hole_far.set_option(rs.option.holes_fill, 1)

hole_near = rs.hole_filling_filter()
hole_near.set_option(rs.option.holes_fill, 2)


def colorize(depth_frame):
    img = np.asanyarray(depth_frame.get_data())

    return cv2.applyColorMap(
        cv2.convertScaleAbs(img, alpha=0.03),
        cv2.COLORMAP_JET
    )


try:
    while True:
        frames = pipeline.wait_for_frames()
        depth = frames.get_depth_frame()

        if not depth:
            continue

        # RAW
        raw_vis = colorize(depth)

        # First do Spatial + Temporal
        base = spatial.process(depth)
        base = temporal.process(base)

        # Then try each hole-filling method
        left = hole_left.process(base)
        far = hole_far.process(base)
        near = hole_near.process(base)

        left_vis = colorize(left)
        far_vis = colorize(far)
        near_vis = colorize(near)

        # Labels
        cv2.putText(
            raw_vis, "RAW",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        cv2.putText(
            left_vis, "FILL FROM LEFT",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        cv2.putText(
            far_vis, "FARTHEST",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        cv2.putText(
            near_vis, "NEAREST",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        # 2x2 layout
        top = np.hstack((raw_vis, left_vis))
        bottom = np.hstack((far_vis, near_vis))

        display = np.vstack((top, bottom))

        # Resize if it doesn't fit on screen
        display = cv2.resize(
            display,
            None,
            fx=0.75,
            fy=0.75
        )

        cv2.imshow("D405 Hole Filling Comparison", display)

        if cv2.waitKey(1) == 27:  # ESC
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()