import gymnasium as gym
from stable_baselines3 import PPO
import time
import math
import numpy as np

# Import the custom environment
from balancing_robot_env import BalancingRobotEnv

def main():
    print("--- Testing Model Recovery from Large Angles ---")
    
    # 1. Instantiate the environment with Matplotlib rendering enabled
    env = BalancingRobotEnv(render_mode="human")
    
    # 2. Load the trained model
    try:
        model = PPO.load("balancing_robot_model")
        print("Model loaded successfully!")
    except Exception as e:
        print("Could not load the model. Did you finish training?")
        return
        
    # 3. Define the extreme angles we want to drop the robot at (in degrees)
    # We stop at 18/19 because 20 is the exact failure threshold in the env!
    test_angles_deg = [0.0, 2.0, 5.0, 8.0, 10.0, 12.0, 15.0, 18.0]
    
    for angle_deg in test_angles_deg:
        print(f"\n--- Starting Test: Dropping robot at {angle_deg} degrees ---")
        obs, _ = env.reset()
        
        # --- HIJACK THE INITIAL STATE ---
        # Convert degrees to radians
        angle_rad = angle_deg * math.pi / 180.0
        
        # Define our custom starting state: [x, x_dot, theta, theta_dot]
        # We start it at position 0, velocity 0, with a massive tilt, and 0 angular velocity
        custom_state = (0.0, 0.0, angle_rad, 0.0)
        
        # env.unwrapped allows us to bypass Gymnasium's safety wrappers and alter the core variables
        env.unwrapped.state = custom_state
        
        # We must also update the 'obs' variable so the Neural Network 
        # sees this massive tilt on its very first frame!
        obs = np.array(custom_state, dtype=np.float32)
        # --------------------------------
        
        done = False
        step_count = 0
        
        while not done:
            # The AI predicts the best action based on the overwritten observation
            action, _states = model.predict(obs, deterministic=True)
            
            # Step the environment forward
            obs, reward, terminated, truncated, _ = env.step(action)
            
            # Update the Matplotlib graphical window
            env.render()
            
            done = terminated or truncated
            step_count += 1
            
            # Limit the test to ~4 seconds (200 steps * 0.02s) so it automatically moves to the next angle
            if step_count > 200:
                print("Robot successfully survived for 4 seconds!")
                break
                
        if terminated:
            print(f"Robot fell! It couldn't recover from {angle_deg} degrees.")
            
        time.sleep(1.5) # Pause for a moment before dropping it at the next angle
        
    env.close()

if __name__ == "__main__":
    main()