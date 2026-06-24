import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

# Import the custom environment we built in Phase 1
from balancing_robot_env import BalancingRobotEnv

def main():
    print("--- Phase 2: Training the Balancing Robot ---")
    
    # 1. Instantiate the environment
    # We wrap it in a Monitor to log statistics like episode length and rewards
    env = Monitor(BalancingRobotEnv())
    
    # 2. Define the PPO Agent
    # MlpPolicy: We are using a Multi-Layer Perceptron (standard Neural Network)
    # verbose=1: Prints training progress to the console
    # tensorboard_log: Saves logs so we can visualize training later
    print("Initializing PPO Neural Network...")
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log="./ppo_balancing_tensorboard/",
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64
    )
    
    # 3. Train the Agent
    # 500,000 timesteps is a good starting point for a simple balancing task.
    # On a modern CPU, this should take a few minutes.
    print("Starting training... This may take a few minutes.")
    model.learn(total_timesteps=300000, progress_bar=True)
    
    # 4. Save the trained Neural Network
    print("Training complete! Saving the model to 'balancing_robot_model.zip'...")
    model.save("balancing_robot_model")
    
    # 5. Evaluate the trained policy
    # Let's test it for 10 episodes to see what average reward it achieves
    print("Evaluating the trained model...")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # 6. Quick Visual Test
    print("\nRunning a live test with the trained brain:")
    obs, _ = env.reset()
    for i in range(200): # Run for 200 simulation steps
        # The model predicts the best action based on the current IMU/Encoder observation
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, _, _ = env.step(action)
        
        env.render()
        
        if terminated:
            print("Robot fell! Resetting...")
            obs, _ = env.reset()

if __name__ == "__main__":
    main()