import threading
import time

from stretch_tools import IS_STRETCH_ENV

if IS_STRETCH_ENV:
    import stretch_body.robot as robot
else:
    import stretch_mujoco_api.robot as robot


SAMPLE_PERIOD = 0.1


def joint_positions(stretch):
    status = stretch.get_status()
    return {
        'lift': round(status['lift']['pos'], 4),
        'arm': round(status['arm']['pos'], 4),
        'head_pan': round(status['head']['head_pan']['pos'], 4),
        'head_tilt': round(status['head']['head_tilt']['pos'], 4),
        **{
            name: round(joint['pos'], 4)
            for name, joint in status['end_of_arm'].items()
        },
    }


def main():
    stretch = robot.Robot()
    if not stretch.startup():
        return

    stretch.enable_collision_mgmt()
    errors = []

    def stow():
        try:
            stretch.stow()
        except Exception as error:
            errors.append(error)

    print('initial', joint_positions(stretch))
    started = time.monotonic()
    worker = threading.Thread(target=stow)
    worker.start()

    try:
        while worker.is_alive():
            elapsed = time.monotonic() - started
            print(f'{elapsed:.2f}', joint_positions(stretch))
            time.sleep(SAMPLE_PERIOD)

        worker.join()
        print('final', joint_positions(stretch))
        if errors:
            raise errors[0]
    finally:
        stretch.stop()


if __name__ == '__main__':
    main()
