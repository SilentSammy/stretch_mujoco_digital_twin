"""
Example script showing cross-platform robot control.
Works in both simulation and physical environments!
"""
from stretch_toolkit import controller, teleop, BACKEND_NAME
import time

print(f"\n=== Running on {BACKEND_NAME} backend ===\n")

def teleop_demo():
    """Run teleoperation loop."""
    print("Teleop demo started. Use gamepad/keyboard to control.")
    print("Press Ctrl+C to stop\n")
    
    # State tracking for printing changes
    last_state = None
    last_print_time = time.time()
    
    # Thresholds for significant change
    POSITION_THRESHOLD = 0.01  # meters or radians
    PRINT_INTERVAL = 0.5  # seconds
    
    try:
        while True:
            # Get normalized velocities from input devices
            velocities = teleop.get_normalized_velocities()
            
            # Send to robot (physical or simulated)
            controller.set_velocities(velocities)
            
            # Periodically print joint changes
            current_time = time.time()
            if current_time - last_print_time >= PRINT_INTERVAL:
                current_state = controller.get_state()
                
                if last_state is not None:
                    # Find joints that changed significantly
                    changed_joints = {}
                    for joint, value in current_state.items():
                        if joint in last_state:
                            delta = abs(value - last_state[joint])
                            if delta >= POSITION_THRESHOLD:
                                changed_joints[joint] = {
                                    'from': last_state[joint],
                                    'to': value,
                                    'delta': value - last_state[joint]
                                }
                    
                    # Print changes if any
                    if changed_joints:
                        print(f"\n[{current_time - last_print_time:.1f}s] Joint changes:")
                        for joint, change in changed_joints.items():
                            print(f"  {joint:30s}: {change['from']:+7.4f} → {change['to']:+7.4f} (Δ {change['delta']:+7.4f})")
                
                last_state = current_state
                last_print_time = current_time
            
            time.sleep(1/30)  # 30 Hz update rate
            
    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        # Stop all motion
        controller.set_velocities({})
        controller.stop()
        print("Demo complete!")

if __name__ == "__main__":
    teleop_demo()
