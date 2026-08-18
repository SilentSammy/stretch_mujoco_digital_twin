import math
import statistics
import time

try:
    import stretch_body.robot as robot
except ImportError:
    import stretch_mujoco_api.robot as robot


SAMPLE_PERIOD = 0.1
COMMAND_TIME = 4.0
STOP_TIMEOUT = 6.0
SETTLED_VELOCITY = 0.005
SETTLED_SAMPLES = 3

TRANSLATION_REQUEST = 0.5
ROTATION_REQUEST = 3.0
MAX_FORWARD_DISTANCE = 1.25
MAX_BACKWARD_DISTANCE = 0.5


class PoseTracker:
    def __init__(self, status):
        self.start_x = status["x"]
        self.start_y = status["y"]
        self.start_theta = status["theta"]
        self.last_theta = status["theta"]
        self.rotation = 0.0

    def update(self, status):
        delta = (status["theta"] - self.last_theta + math.pi) % (2 * math.pi) - math.pi
        self.rotation += delta
        self.last_theta = status["theta"]

    def translation(self, status):
        dx = status["x"] - self.start_x
        dy = status["y"] - self.start_y
        return dx * math.cos(self.start_theta) + dy * math.sin(self.start_theta)


def send_velocity(stretch, linear, angular):
    stretch.base.set_velocity(linear, angular)
    stretch.push_command()


def take_sample(stretch, tracker, started, position_kind):
    status = stretch.base.status
    tracker.update(status)
    position = (
        tracker.translation(status)
        if position_kind == "translation"
        else tracker.rotation
    )
    velocity = (
        status["x_vel"]
        if position_kind == "translation"
        else status["theta_vel"]
    )
    return {
        "time": status["pose_time_s"] - started,
        "position": position,
        "velocity": velocity,
    }


def print_sample(label, sample):
    print(
        label,
        f"t={sample['time']:.3f}",
        f"pos={sample['position']:.4f}",
        f"vel={sample['velocity']:.4f}",
    )


def run_for_time(stretch, tracker, label, linear, angular, position_kind):
    started = stretch.base.status["pose_time_s"]
    samples = []

    while True:
        send_velocity(stretch, linear, angular)
        sample = take_sample(stretch, tracker, started, position_kind)
        samples.append(sample)
        print_sample(label, sample)

        if sample["time"] >= COMMAND_TIME:
            break
        if position_kind == "translation":
            if sample["position"] >= MAX_FORWARD_DISTANCE:
                print("Forward-distance safety limit reached")
                break
            if sample["position"] <= -MAX_BACKWARD_DISTANCE:
                print("Backward-distance safety limit reached")
                break

        time.sleep(SAMPLE_PERIOD)

    return samples


def run_until_position(
    stretch,
    tracker,
    label,
    linear,
    angular,
    position_kind,
    target,
):
    started = stretch.base.status["pose_time_s"]
    samples = []

    while True:
        send_velocity(stretch, linear, angular)
        sample = take_sample(stretch, tracker, started, position_kind)
        samples.append(sample)
        print_sample(label, sample)

        reached = (
            sample["position"] <= target
            if linear < 0 or angular < 0
            else sample["position"] >= target
        )
        if reached:
            break
        if sample["time"] >= 2 * COMMAND_TIME + STOP_TIMEOUT:
            print(label, "position timeout")
            break
        if position_kind == "translation" and sample["position"] <= -MAX_BACKWARD_DISTANCE:
            print("Backward-distance safety limit reached")
            break

        time.sleep(SAMPLE_PERIOD)

    return samples


def stop_and_sample(stretch, tracker, label, position_kind):
    started = stretch.base.status["pose_time_s"]
    samples = []
    settled = 0

    while True:
        send_velocity(stretch, 0.0, 0.0)
        sample = take_sample(stretch, tracker, started, position_kind)
        samples.append(sample)
        print_sample(label, sample)

        if abs(sample["velocity"]) <= SETTLED_VELOCITY:
            settled += 1
        else:
            settled = 0

        if settled >= SETTLED_SAMPLES:
            break
        if sample["time"] >= STOP_TIMEOUT:
            print(label, "stop timeout")
            break

        time.sleep(SAMPLE_PERIOD)

    return samples


def first_crossing(samples, threshold, increasing):
    for sample in samples:
        speed = abs(sample["velocity"])
        if (increasing and speed >= threshold) or (not increasing and speed <= threshold):
            return sample
    return None


def print_statistics(label, moving, stopping):
    speeds = [abs(sample["velocity"]) for sample in moving]
    peak = max(speeds)
    saturated = [speed for speed in speeds if speed >= 0.9 * peak]

    low = first_crossing(moving, 0.1 * peak, increasing=True)
    high = first_crossing(moving, 0.9 * peak, increasing=True)
    accel_time = None
    acceleration = None
    if low is not None and high is not None and high["time"] > low["time"]:
        accel_time = high["time"] - low["time"]
        acceleration = (0.8 * peak) / accel_time

    stop_high = first_crossing(stopping, 0.9 * peak, increasing=False)
    stop_low = first_crossing(stopping, 0.1 * peak, increasing=False)
    decel_time = None
    deceleration = None
    if (
        stop_high is not None
        and stop_low is not None
        and stop_low["time"] > stop_high["time"]
    ):
        decel_time = stop_low["time"] - stop_high["time"]
        deceleration = (0.8 * peak) / decel_time

    print(f"\n{label} statistics")
    print(f"peak_speed={peak:.4f}")
    print(f"saturated_mean={statistics.mean(saturated):.4f}")
    print(f"saturated_std={statistics.pstdev(saturated):.4f}")
    print(f"time_10_to_90={accel_time}")
    print(f"acceleration_10_to_90={acceleration}")
    print(f"deceleration_time_90_to_10={decel_time}")
    print(f"deceleration_90_to_10={deceleration}\n")


def correct_translation(stretch, tracker):
    status = stretch.base.status
    tracker.update(status)
    residual = tracker.translation(status)
    print(f"translation residual before correction={residual:.4f}")
    if abs(residual) > 0.01:
        stretch.base.translate_by(-residual, v_m=0.05)
        stretch.push_command()
        stretch.wait_command()
    status = stretch.base.status
    tracker.update(status)
    print(f"translation residual final={tracker.translation(status):.4f}")


def correct_rotation(stretch, tracker, origin):
    status = stretch.base.status
    tracker.update(status)
    residual = tracker.rotation - origin
    print(f"rotation residual before correction={residual:.4f}")
    if abs(residual) > 0.01:
        stretch.base.rotate_by(-residual, v_r=0.2)
        stretch.push_command()
        stretch.wait_command()
    status = stretch.base.status
    tracker.update(status)
    print(f"rotation residual final={tracker.rotation - origin:.4f}")


def test_translation(stretch, tracker):
    print("\n=== TRANSLATION SATURATION ===")
    forward = run_for_time(
        stretch,
        tracker,
        "translation +",
        TRANSLATION_REQUEST,
        0.0,
        "translation",
    )
    forward_release = forward[-1]["position"]
    forward_stop = stop_and_sample(
        stretch, tracker, "translation + stop", "translation"
    )
    stopping_distance = max(forward_stop[-1]["position"] - forward_release, 0.0)

    reverse = run_until_position(
        stretch,
        tracker,
        "translation -",
        -TRANSLATION_REQUEST,
        0.0,
        "translation",
        stopping_distance,
    )
    reverse_stop = stop_and_sample(
        stretch, tracker, "translation - stop", "translation"
    )

    print_statistics("translation +", forward, forward_stop)
    print_statistics("translation -", reverse, reverse_stop)
    correct_translation(stretch, tracker)


def test_rotation(stretch, tracker):
    print("\n=== ROTATION SATURATION ===")
    origin = tracker.rotation
    positive = run_for_time(
        stretch,
        tracker,
        "rotation +",
        0.0,
        ROTATION_REQUEST,
        "rotation",
    )
    positive_release = positive[-1]["position"]
    positive_stop = stop_and_sample(stretch, tracker, "rotation + stop", "rotation")
    stopping_angle = max(positive_stop[-1]["position"] - positive_release, 0.0)

    negative = run_until_position(
        stretch,
        tracker,
        "rotation -",
        0.0,
        -ROTATION_REQUEST,
        "rotation",
        origin + stopping_angle,
    )
    negative_stop = stop_and_sample(stretch, tracker, "rotation - stop", "rotation")

    print_statistics("rotation +", positive, positive_stop)
    print_statistics("rotation -", negative, negative_stop)
    correct_rotation(stretch, tracker, origin)


def main():
    stretch = robot.Robot()
    if not stretch.startup():
        return

    stretch.enable_collision_mgmt()
    tracker = PoseTracker(stretch.base.status)

    try:
        test_translation(stretch, tracker)
        test_rotation(stretch, tracker)
    finally:
        for _ in range(10):
            send_velocity(stretch, 0.0, 0.0)
            time.sleep(SAMPLE_PERIOD)
        stretch.stop()


if __name__ == "__main__":
    main()
