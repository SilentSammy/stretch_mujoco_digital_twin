"""
Example script showing cross-platform robot control.
Works in both simulation and physical environments!
"""
from stretch_toolkit import controller, teleop, BACKEND_NAME
import time

print(f"\n=== Running on {BACKEND_NAME} backend ===\n")

def merge_proportional(cmd_primary, cmd_secondary):
    # Primary command overrides secondary based on its magnitude
    cmd_final = {}
    
    # Handle all joints from both commands
    all_joints = set(cmd_primary.keys()) | set(cmd_secondary.keys())
    
    for joint in all_joints:
        primary_input = cmd_primary.get(joint, 0.0)    # Default to 0 if missing
        secondary_input = cmd_secondary.get(joint, 0.0) # Default to 0 if missing
        
        if abs(primary_input) < 0.05:  # No primary input
            cmd_final[joint] = secondary_input
        else:
            # Primary input interpolates between secondary and desired value
            # abs(primary_input) determines how much override (0 to 1)
            # sign(primary_input) determines direction
            override_strength = abs(primary_input)
            desired_value = 1.0 if primary_input > 0 else -1.0
            cmd_final[joint] = (1 - override_strength) * secondary_input + override_strength * desired_value
    
    return cmd_final

def teleop_demo():
    """Run teleoperation loop with automatic lift control."""
    print("Teleop demo started. Use gamepad/keyboard to control.")
    print("Lift will automatically move between 0.2m and 1.0m")
    print("Press Ctrl+C to stop\n")
    
    # State tracking for printing changes
    last_state = None
    last_print_time = time.time()
    
    # Thresholds for significant change
    POSITION_THRESHOLD = 0.01  # meters or radians
    PRINT_INTERVAL = 0.5  # seconds
    
    # Automatic lift control state
    lift_target = 1.0  # Start by moving to 1.0m
    LIFT_LOW = 0.2
    LIFT_HIGH = 1.0
    LIFT_TOLERANCE = 0.05  # Switch target when within 5cm
    LIFT_KP = 5.0  # Proportional gain for P controller
    
    try:
        while True:
            # Get normalized velocities from input devices (primary)
            teleop_cmd = teleop.get_normalized_velocities()
            
            # Get current state
            current_state = controller.get_state()
            lift_pos = current_state['lift_up']
            
            # Automatic lift controller (secondary)
            lift_error = lift_target - lift_pos
            
            # Switch target when close enough
            if abs(lift_error) < LIFT_TOLERANCE:
                lift_target = LIFT_LOW if lift_target == LIFT_HIGH else LIFT_HIGH
                print(f"\n[AUTO] Lift target switched to {lift_target}m")
                lift_error = lift_target - lift_pos
            
            # P controller: velocity proportional to error
            lift_vel = max(-1.0, min(1.0, lift_error * LIFT_KP))  # Clamp to [-1, 1]
            
            # Create automatic control command (secondary)
            auto_cmd = {
                'lift_up': lift_vel
            }
            
            # Merge commands: teleop (primary) can override auto (secondary)
            final_cmd = merge_proportional(teleop_cmd, auto_cmd)
            
            # Send to robot (physical or simulated)
            controller.set_velocities(final_cmd)
            
            # Periodically print joint changes
            current_time = time.time()
            if current_time - last_print_time >= PRINT_INTERVAL:
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
                            status = ""
                            if joint == 'lift_up':
                                status = f" [Target: {lift_target:.2f}m, Error: {lift_error:+.3f}m]"
                            print(f"  {joint:30s}: {change['from']:+7.4f} → {change['to']:+7.4f} (Δ {change['delta']:+7.4f}){status}")
                
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
