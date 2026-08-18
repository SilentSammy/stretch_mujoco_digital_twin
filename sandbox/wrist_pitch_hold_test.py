import time

import stretch_mujoco_api.robot as robot


def sample(stretch, joint, duration, command_zero):
    positions = []
    end = time.monotonic() + duration
    while time.monotonic() < end:
        if command_zero:
            joint.set_velocity(0.0)
            stretch.push_command()
        positions.append(joint.status["pos"])
        time.sleep(0.1)
    return positions


if __name__ == "__main__":
    stretch = robot.Robot()
    stretch.startup()
    stretch.enable_collision_mgmt()
    wrist_pitch = stretch.end_of_arm.get_joint("wrist_pitch")

    try:
        wrist_pitch.move_to(-0.4)
        stretch.push_command()
        stretch.wait_command()

        idle = sample(stretch, wrist_pitch, 2.0, command_zero=False)
        commanded = sample(stretch, wrist_pitch, 2.0, command_zero=True)
        print("idle drift:", idle[-1] - idle[0])
        print("zero-command drift:", commanded[-1] - commanded[0])
        assert abs(commanded[-1] - commanded[0]) < 0.01
    finally:
        stretch.stop()
