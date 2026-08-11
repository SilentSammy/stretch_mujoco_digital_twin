import time
from stretch_mujoco.stretch_mujoco_simulator import StretchMujocoSimulator

if __name__ == "__main__":
    sim = StretchMujocoSimulator()
    sim.start(headless=True)
    start_time = time.time()
    try:
        while sim.is_running():
            status = sim.pull_status()
            real_time = time.time() - start_time
            sim_time = status.time
            ratio = status.sim_to_real_time_ratio_msg
            ratio2 = sim_time / real_time if real_time > 0 else float('inf')

            print(f"rt: {real_time:.2f}, st: {sim_time:.2f}, r: {ratio}, r2: {ratio2:.2f}")
    finally:
        sim.stop()
