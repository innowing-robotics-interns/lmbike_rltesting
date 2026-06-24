import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math

class BalancingRobotEnv(gym.Env):
    """
    Two-wheeled balancing robot RL simulation environment with Matplotlib rendering.
    """
    def __init__(self, render_mode=None):
        super(BalancingRobotEnv, self).__init__()
        self.render_mode = render_mode
        self.fig = None
        self.ax = None
        
        # --- Physics parameters ---
        self.gravity = 9.8            
        self.masscart = 0.5           
        self.masspole = 0.5           
        self.total_mass = self.masspole + self.masscart
        self.length = 0.15            
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0         
        self.tau = 0.02               
        
        # --- Hardware Realism: Motor Delay ---
        self.motor_alpha = 0.3        # Motor responsiveness (1.0 = instant, 0.1 = very sluggish)
        self.current_action = 0.0     # Tracks the actual physical state of the motor
        
        self.theta_threshold_radians = 20 * 2 * math.pi / 360  
        self.x_threshold = 2.4        
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max
        ], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        
        self.state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_action = 0.0     # Reset the motor state
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        return np.array(self.state, dtype=np.float32), {}

    def step(self, action):
        state = self.state
        x, x_dot, theta, theta_dot = state
        
        target_action = np.clip(action, -1.0, 1.0)[0]
        
        # --- Simulate Motor Delay (First-Order Lag) ---
        self.current_action = (self.motor_alpha * target_action) + ((1.0 - self.motor_alpha) * self.current_action)
        self._last_action = self.current_action  # Save actual output for the renderer
        
        force = self.current_action * self.force_mag
        
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        
        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        
        self.state = (x, x_dot, theta, theta_dot)
        
        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        
        if not terminated:
            reward = 1.0 
            # --- Advanced Reward Shaping (Quadratic Penalties) ---
            reward -= (theta / self.theta_threshold_radians)**2 * 3.0  
            reward -= (x / self.x_threshold)**2 * 2.0                  
            reward -= abs(x_dot) * 0.1                                 
            reward -= abs(target_action) * 0.05                        
        else:
            reward = -10.0
            
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

    def close(self):
        if self.fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self.fig)
            self.fig = None
            self.ax = None

    # UPDATED: Added ax and title arguments for subplot support
    def render(self, ax=None, title="Balancing Robot"):
        x, x_dot, theta, theta_dot = self.state
        angle_deg = theta * 180 / math.pi
        
        if self.render_mode == "human":
            import matplotlib
            # Only try to set TkAgg if we aren't already in a rendering loop
            if ax is None and self.fig is None:
                try:
                    matplotlib.use('TkAgg')
                except:
                    pass
                    
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches

            # Use the provided axis (for side-by-side), OR create a new one
            current_ax = ax
            if current_ax is None:
                if self.fig is None:
                    plt.ion()  
                    self.fig, self.ax = plt.subplots(figsize=(6, 4))
                    self.fig.canvas.manager.set_window_title("Balancing Robot - Matplotlib")
                    plt.show(block=False) 
                current_ax = self.ax
            
            # Clear the previous frame
            current_ax.clear()
            
            # Set physical limits for the axes
            current_ax.set_xlim(-self.x_threshold, self.x_threshold)
            current_ax.set_ylim(-0.5, 1.5)
            current_ax.set_aspect('equal')
            current_ax.axis('off')  
            
            # 1. Draw the Ground
            current_ax.axhline(0, color='black', linewidth=2)
            
            # 2. Draw the Cart (Chassis)
            cart_width, cart_height = 0.4, 0.2
            cart = patches.Rectangle(
                (x - cart_width/2, -cart_height/2), 
                cart_width, 
                cart_height, 
                color='royalblue',
                zorder=2
            )
            current_ax.add_patch(cart)
            
            # 3. Draw the Pole (Robot Body)
            pole_length = 0.8
            pole_end_x = x + pole_length * math.sin(theta)
            pole_end_y = pole_length * math.cos(theta)
            current_ax.plot([x, pole_end_x], [0, pole_end_y], color='crimson', linewidth=5, zorder=1)
            
            # 4. Add telemetry text
            info_text = f"Pos: {x:.2f}m | Angle: {angle_deg:.2f}° | Action: {self._last_action if hasattr(self, '_last_action') else 0:.2f}"
            current_ax.text(-self.x_threshold + 0.1, 1.3, info_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
            current_ax.set_title(title)

            # Only pause if the environment is managing its own window
            if ax is None:
                plt.pause(0.001) 
        else:
            print(f"[{title}] Pos: {x:.2f}m | Vel: {x_dot:.2f}m/s | Angle: {angle_deg:.2f}° | Action: {self._last_action if hasattr(self, '_last_action') else 0:.2f}")

if __name__ == "__main__":
    env = BalancingRobotEnv(render_mode="human")
    obs, _ = env.reset()
    for i in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        env.render()
        if terminated:
            obs, _ = env.reset()
            break