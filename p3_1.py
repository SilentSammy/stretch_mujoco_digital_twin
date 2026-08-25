import cv2
from stretch_tools import Cameras, HEAD_CAMERA

from stretch_tools import NormVelController, TeleopProvider

try:
    import stretch_body.robot as robot
except ImportError:
    import stretch_mujoco_api.robot as robot


def main():
    stretch = robot.Robot()
    stretch.startup()
    stretch.enable_collision_mgmt()
    
    controller = NormVelController(stretch)
    teleop = TeleopProvider(robot=stretch)

    cameras = Cameras ( head_info = HEAD_CAMERA )

    try:
        while True:
            success , frame , depth_frame = cameras.read_head()
            if success:
                cv2.imshow("Head RGB", frame)
                color_depth_frame = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_frame, alpha=0.03),
                    cv2.COLORMAP_JET,
                )
                cv2.imshow("Head Depth", color_depth_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            command = {}
            
            # Send velocity command with manual override
            command = teleop.get_manual_override(command)
            controller.set_command(command)

    except KeyboardInterrupt:
        pass
    finally:
        stretch.stop()

if __name__ == "__main__":
    main()
