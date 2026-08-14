import time

try:
    import stretch_body.robot as robot
except ImportError:
    import stretch_mujoco_api.robot as robot


def main():
    stretch = robot.Robot()
    stretch.startup()
    stretch.enable_collision_mgmt()

    try:
        print("joints", stretch.head.joints, type(stretch.head.joints).__name__)

        for name in stretch.head.joints:
            status = stretch.head.get_joint(name).status
            print(name, "idle keys", list(status))
            print(name, "idle", status)

        stretch.head.move_to("head_pan", 0.5)
        time.sleep(0.3)

        status = stretch.head.get_joint("head_pan").status
        print("head_pan moving keys", list(status))
        print("head_pan moving", status)

        stretch.wait_command()
        print("head_pan settled", stretch.head.get_joint("head_pan").status)

    finally:
        stretch.stop()


if __name__ == "__main__":
    main()
