try:
    import stretch_body.robot as robot
except ImportError:
    import stretch_mujoco_api.robot as robot


OFFSETS = {
    "head_pan": 0.25,
    "head_tilt": -0.20,
    "wrist_pitch": -0.20,
    "wrist_roll": 0.25,
    "wrist_yaw": 0.25,
}


def main():
    stretch = robot.Robot()
    stretch.startup()

    try:
        joints = {
            "head_pan": stretch.head.get_joint("head_pan"),
            "head_tilt": stretch.head.get_joint("head_tilt"),
            "wrist_pitch": stretch.end_of_arm.get_joint("wrist_pitch"),
            "wrist_roll": stretch.end_of_arm.get_joint("wrist_roll"),
            "wrist_yaw": stretch.end_of_arm.get_joint("wrist_yaw"),
        }
        initial = {name: joint.status["pos"] for name, joint in joints.items()}

        for name, offset in OFFSETS.items():
            if name.startswith("head_"):
                stretch.head.move_by(name, offset)
            else:
                stretch.end_of_arm.move_by(name, offset)

        stretch.wait_command()
        for name, joint in joints.items():
            status = joint.status
            target = initial[name] + OFFSETS[name]
            print(
                name,
                "moved",
                f"pos={status['pos']:.4f}",
                f"vel={status['vel']:.4f}",
                f"error={target - status['pos']:.4f}",
            )

        for name, position in initial.items():
            if name.startswith("head_"):
                stretch.head.move_to(name, position)
            else:
                stretch.end_of_arm.move_to(name, position)

        stretch.wait_command()
        for name, joint in joints.items():
            status = joint.status
            print(
                name,
                "returned",
                f"pos={status['pos']:.4f}",
                f"vel={status['vel']:.4f}",
                f"error={initial[name] - status['pos']:.4f}",
            )
    finally:
        stretch.stop()


if __name__ == "__main__":
    main()
