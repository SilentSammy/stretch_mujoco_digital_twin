import time
import sys
import numpy as np
from stretch_mujoco_api.robot import Robot
from stretch_mujoco_api.sim_config import SimConfig, ObjectControlsConfig
from stretch_tools import HEAD_CAMERA, WRIST_CAMERA, NAVIGATION_CAMERA, close_cameras
from stretch_mujoco_api import cameras

def main():
    robot = Robot(SimConfig(object_controls=ObjectControlsConfig(enabled=False)))
    robot.startup()
    checked = 0
    try:
        for cycle in range(6):
            order = (WRIST_CAMERA, HEAD_CAMERA) if 'reverse' in sys.argv else (HEAD_CAMERA, WRIST_CAMERA)
            for camera in order:
                for i in range(8):
                    ok, rgb, depth = camera.get_frames()
                    assert ok and np.any(rgb) and np.any(depth), (cycle, camera.name, i)
                    checked += 1
                    if 'all' in sys.argv:
                        other = HEAD_CAMERA if camera is WRIST_CAMERA else WRIST_CAMERA
                        other_ok, other_rgb, other_depth = other.get_frames()
                        nav_ok, nav_rgb = NAVIGATION_CAMERA.get_frame()
                        assert other_ok and np.any(other_rgb) and np.any(other_depth)
                        assert nav_ok and np.any(nav_rgb)
                        checked += 2
                    raw = robot._sim.pull_camera_data()
                    raw_depth = raw.cam_d405_depth if camera is WRIST_CAMERA else raw.cam_d435i_depth
                    print('CHECK', cycle, camera.name, i, ok,
                          (int(depth.min()), int(depth.max()), np.count_nonzero(depth)) if ok else None,
                          (float(raw_depth.min()),float(raw_depth.max())) if raw_depth is not None else None, flush=True)
                    time.sleep(0.15)
            time.sleep(1.5)
        print('PASSED', checked, 'frame reads', flush=True)
    finally:
        close_cameras()
        robot.stop()

if __name__ == '__main__':
    main()
