import gymnasium as gym
from stable_baselines3 import PPO
import time
import math
import numpy as np
import matplotlib
matplotlib.use('TkAgg') # Force GUI on Linux
import matplotlib.pyplot as plt

# Import the custom environment
from balancing_robot_env import BalancingRobotEnv

def main():
    print("--- Comparing 300k vs 500k Side-by-Side ---")
    
    # Instantiate TWO separate environments
    env1 = BalancingRobotEnv(render_mode="human")
    env2 = BalancingRobotEnv(render_mode="human")
    
    # Load both models
    try:
        model_300k = PPO.load("model_300k", device="cpu")
        model_500k = PPO.load("model_500k", device="cpu")
        print("Both models loaded successfully!")
    except Exception as e:
        print(f"Could not load the models. Error: {e}")
        return

    # Setup the Side-by-Side Matplotlib window
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.canvas.manager.set_window_title("Training Progress Comparison")
    plt.show(block=False)
        
    test_angles_deg = [0.0, 5.0, 10.0, 15.0, 18.0]
    
    for angle_deg in test_angles_deg:
        print(f"\nDropping robots at {angle_deg} degrees...")
        obs1, _ = env1.reset()
        obs2, _ = env2.reset()
        
        # Hijack the states so they start identically
        angle_rad = angle_deg * math.pi / 180.0
        custom_state = (0.0, 0.0, angle_rad, 0.0)
        
        env1.unwrapped.state = custom_state
        env2.unwrapped.state = custom_state
        
        obs1 = np.array(custom_state, dtype=np.float32)
        obs2 = np.array(custom_state, dtype=np.float32)
        
        done1, done2 = False, False
        step_count = 0
        
        # Run until BOTH robots have fallen or 5 seconds pass
        while step_count < 250: # 5 seconds
            # Model 1 (300k) decides
            if not done1:
                action1, _ = model_300k.predict(obs1, deterministic=True)
                obs1, _, term1, trunc1, _ = env1.step(action1)
                done1 = term1 or trunc1
            
            # Model 2 (500k) decides
            if not done2:
                action2, _ = model_500k.predict(obs2, deterministic=True)
                obs2, _, term2, trunc2, _ = env2.step(action2)
                done2 = term2 or trunc2
            
            # Render them simultaneously on their respective axes
            env1.render(ax=ax1, title="Model A (300k Steps)")
            env2.render(ax=ax2, title="Model B (500k Steps)")
            
            # Update the main window once per frame
            plt.pause(0.001)
            step_count += 1
            
            # If both have fallen, move to the next test early
            if done1 and done2:
                break
                
        time.sleep(1.5)
        
    env1.close()
    env2.close()

if __name__ == "__main__":
    main()