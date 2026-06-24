from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from balancing_robot_env import BalancingRobotEnv

def main():
    print("--- Multi-Stage Training ---")
    env = Monitor(BalancingRobotEnv())
    
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
    
    # 1. Train up to 300k and save
    print("\n--- Training Phase 1: 0 to 300,000 steps ---")
    model.learn(total_timesteps=300000, reset_num_timesteps=False, tb_log_name="PPO_300k")
    model.save("model_300k")
    print("Saved 'model_300k.zip'")

    # 2. Resume training the exact same model for 200k more steps (total 500k)
    print("\n--- Training Phase 2: 300,000 to 500,000 steps ---")
    model.learn(total_timesteps=200000, reset_num_timesteps=False, tb_log_name="PPO_500k")
    model.save("model_500k")
    print("Saved 'model_500k.zip'")
    
    print("Done! You now have both models ready for comparison.")

if __name__ == "__main__":
    main()