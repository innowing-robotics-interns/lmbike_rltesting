import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math

class _PID:
    """
    Python version of the PID controller, perfectly replicating the C logic.
    Includes integral anti-windup and output clamping.
    """
    def __init__(self, integral_max, out_max):
        self.err = 0.0
        self.err_last = 0.0
        self.expect = 0.0
        self.feedback = 0.0
        
        # Parameters to be dynamically tuned by the RL agent
        self.kp = 0.0
        self.ki = 0.0
        self.kd = 0.0
        
        self.integral = 0.0
        self.integral_max = integral_max
        
        self.out = 0.0
        self.out_max = out_max

    def compute(self, expect, feedback):
        self.err_last = self.err
        self.err = expect - feedback
        self.integral += self.ki * self.err
        
        # Integral Anti-windup
        if self.integral > self.integral_max:
            self.integral = self.integral_max
        elif self.integral < -self.integral_max:
            self.integral = -self.integral_max
            
        # PID Calculation (Matches C code formula)
        self.out = self.kp * self.err + self.integral + self.kd * (self.err - self.err_last)
        
        # Output Limiting
        if self.out > self.out_max:
            self.out = self.out_max
        elif self.out < -self.out_max:
            self.out = -self.out_max
            
        return self.out

    def clear_integral(self):
        self.integral = 0.0


class PIDBalancingRobotEnv(gym.Env):
    """
    Two-wheeled balancing robot RL simulation environment based on Cascaded PID.
    The RL agent's job is to dynamically tune the PID parameters.
    """
    def __init__(self, render_mode=None):
        super(PIDBalancingRobotEnv, self).__init__()
        self.render_mode = render_mode
        self.fig = None
        self.ax = None
        
        # --- Physics parameters ---
        self.gravity = 9.8            
        self.masscart = 2.5           
        self.masspole = 9.5          
        self.total_mass = self.masspole + self.masscart
        self.length = 0.6            
        self.polemass_length = self.masspole * self.length
        self.force_mag = 50.0         
        self.tau = 0.02               # Frequency = 1/tau = 50 Hz              
        
        self.theta_threshold_radians = 70 * 2 * math.pi / 360  
        self.x_threshold = 5.0        
        
        # --- Initialize 3-loop Cascaded PID Controllers ---
        # Corresponds to all.vel_encoder, all.rol_angle, all.rol_gyro in C
        self.pid_vel = _PID(integral_max=500.0, out_max=2000.0)
        self.pid_angle = _PID(integral_max=550.0, out_max=2000.0)
        self.pid_gyro = _PID(integral_max=500.0, out_max=2000.0)
        
        # --- Action Space ---
        # RL outputs 5 values [-1.0, 1.0] to tune the 5 non-zero PID gains:
        # [Vel_Kp, Vel_Ki, Vel_Kd, Angle_Kp, Gyro_Kp]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
        
        # --- Observation Space ---
        high = np.array([
            self.x_threshold * 2,           
            np.finfo(np.float32).max,       
            self.theta_threshold_radians * 2, 
            np.finfo(np.float32).max        
        ], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        
        self.state = None
        self._last_motor_out = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Clear all PID integrals (similar to C code reset after falling)
        self.pid_vel.clear_integral()
        self.pid_angle.clear_integral()
        self.pid_gyro.clear_integral()
        
        # FIX: Decoupled Randomization for robust training (prevents catastrophic forgetting)
        x = self.np_random.uniform(low=-1.0, high=1.0)              
        x_dot = self.np_random.uniform(low=-0.2, high=0.2)          
        theta = self.np_random.uniform(low=-0.25, high=0.25)        # ~14.3 degrees spawn
        theta_dot = self.np_random.uniform(low=-0.1, high=0.1)      
        
        self.state = (x, x_dot, theta, theta_dot)
        self._last_motor_out = 0.0
        return np.array(self.state, dtype=np.float32), {}

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        
        # --- 1. RL Action to PID Tuning ---
        # Map the [-1, 1] action into safe, stable physical bounds.
        self.pid_vel.kp = max(0.0, 1.0 + action[0] * 1.0)       # Range: 0.0 ~ 2.0 
        self.pid_vel.ki = max(0.0, 0.01 + action[1] * 0.01)     # Range: 0.0 ~ 0.02
        
        # FIX 1: Massively reduced the Velocity Kd (Derivative/Dampener) range.
        # It was previously allowed to go up to 1.0, acting as a massive brake.
        self.pid_vel.kd = max(0.0, 0.1 + action[2] * 0.1)       # Range: 0.0 ~ 0.2 
        
        self.pid_angle.kp = max(0.0, 10.0 + action[3] * 5.0)    # Range: 5.0 ~ 15.0 
        self.pid_gyro.kp = max(0.0, 20.0 + action[4] * 10.0)    # Range: 10.0 ~ 30.0 
        
        # Keep consistent with C code, set other unused parameters to 0
        self.pid_angle.ki, self.pid_angle.kd = 0.0, 0.0
        self.pid_gyro.ki, self.pid_gyro.kd = 0.0, 0.0

        # --- 2. Run Cascaded PID Logic ---
        # (1) Velocity Controller (Outer Loop)
        vel_out = self.pid_vel.compute(expect=0.0, feedback=x_dot)
        
        # (2) Angle Controller (Middle Loop)
        angle_out = self.pid_angle.compute(expect=vel_out, feedback=theta)
        
        # (3) Gyro Controller (Inner Loop)
        gyro_out = self.pid_gyro.compute(expect=angle_out, feedback=theta_dot)

        # Map the final PID PWM output to physical engine force
        motor_force = (gyro_out / 2000.0) * self.force_mag
        self._last_motor_out = gyro_out

        # --- 3. Physics Simulation Engine ---
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        
        temp = (motor_force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        
        self.state = (x, x_dot, theta, theta_dot)
        
        # --- 4. Reward and Termination Conditions ---
        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        
        if not terminated:
            # Base survival reward
            reward = 1.0 
            
            # FIX 2: Removed the Angle "Dampener" Penalty completely.
            # By removing `reward -= (theta / threshold)**2`, the AI no longer panics 
            # about being slightly tilted. It is free to swing and use momentum.
            # We only keep a very light position penalty to stop it from driving away.
            reward -= (x / self.x_threshold)**2 * 0.5                  
            
            # Keep a very tiny penalty for wildly fluctuating RL actions 
            # so the agent provides smooth PID tuning rather than jittering the knobs.
            reward -= sum([abs(a) for a in action]) * 0.005
        else:
            reward = -10.0
            
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

    def close(self):
        if self.fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self.fig)
            self.fig = None
            self.ax = None

    def render(self, ax=None, title="RL-Tuned Cascaded PID Robot"):
        x, x_dot, theta, theta_dot = self.state
        angle_deg = theta * 180 / math.pi
        
        if self.render_mode == "human":
            import matplotlib
            if ax is None and self.fig is None:
                try:
                    matplotlib.use('TkAgg')
                except:
                    pass
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches

            current_ax = ax
            if current_ax is None:
                if self.fig is None:
                    plt.ion()  
                    self.fig, self.ax = plt.subplots(figsize=(6, 4))
                    self.fig.canvas.manager.set_window_title("Balancing Robot - RL tuned PID")
                    plt.show(block=False) 
                current_ax = self.ax
            
            current_ax.clear()
            current_ax.set_xlim(-self.x_threshold, self.x_threshold)
            current_ax.set_ylim(-0.5, 1.5)
            current_ax.set_aspect('equal')
            current_ax.axis('off')  
            
            # Ground
            current_ax.axhline(0, color='black', linewidth=2)
            
            # Cart Body
            cart_width, cart_height = 0.4, 0.2
            cart = patches.Rectangle(
                (x - cart_width/2, -cart_height/2), 
                cart_width, 
                cart_height, 
                color='mediumseagreen',
                zorder=2
            )
            current_ax.add_patch(cart)
            
            # Pole
            pole_length = 0.8
            pole_end_x = x + pole_length * math.sin(theta)
            pole_end_y = pole_length * math.cos(theta)
            current_ax.plot([x, pole_end_x], [0, pole_end_y], color='darkorange', linewidth=5, zorder=1)
            
            # Telemetry text
            info_text = f"Pos: {x:.2f}m | Angle: {angle_deg:.2f}°\nPID Out(PWM): {self._last_motor_out:.0f}"
            current_ax.text(-self.x_threshold + 0.1, 1.2, info_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
            current_ax.set_title(title)

            if ax is None:
                plt.pause(0.001) 
        else:
            print(f"[{title}] Pos: {x:.2f}m | Vel: {x_dot:.2f}m/s | Angle: {angle_deg:.2f}° | PID Out: {self._last_motor_out:.0f}")

if __name__ == "__main__":
    env = PIDBalancingRobotEnv(render_mode="human")
    obs, _ = env.reset()
    for i in range(500):
        # Sample random PID tunings for testing
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        env.render()
        if terminated:
            obs, _ = env.reset()
            break