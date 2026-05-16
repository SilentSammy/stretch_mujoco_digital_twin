"""
Direct MuJoCo LiDAR reader.
This loads the XML directly and prints rangefinder readings.
"""

import mujoco
import time
from pathlib import Path
import numpy as np


def main():
    scene_path = Path("stretch_mujoco/models/scene.xml").resolve()

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)

    rangefinder_ids = []

    for sensor_id in range(model.nsensor):
        if model.sensor_type[sensor_id] == mujoco.mjtSensor.mjSENS_RANGEFINDER:
            rangefinder_ids.append(sensor_id)

    print(f"Found {len(rangefinder_ids)} rangefinder sensors.")

    try:
        while True:
            mujoco.mj_step(model, data)

            ranges = []

            for sensor_id in rangefinder_ids:
                adr = model.sensor_adr[sensor_id]
                dim = model.sensor_dim[sensor_id]

                value = float(data.sensordata[adr:adr + dim][0])

                # MuJoCo returns -1 when the ray does not hit anything within cutoff
                if value < 0:
                    value = np.inf

                ranges.append(value)

            print(
                "LiDAR:",
                ["inf" if np.isinf(r) else f"{r:.2f}" for r in ranges[:20]],
                f"... total={len(ranges)}"
            )

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping.")


if __name__ == "__main__":
    main()