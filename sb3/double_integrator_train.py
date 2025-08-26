"""
sb3_ppo_double_integrator_torch_vectorized.py

Requirements:
    pip install stable-baselines3 gym torch

Run:
    python sb3_ppo_double_integrator_torch_vectorized.py
"""

import gymnasium
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv
from gymnasium.spaces import Box
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor

# -------------------------
# Vectorized double integrator implemented in torch
# -------------------------
class VectorizedDoubleIntegrator(VecEnv):
    """
    Vectorized double integrator environment:
      state: [position, velocity]  shape (2,)
      action: acceleration scalar (continuous)
      dynamics: x' = v, v' = a  (simple Euler integration)
    The environment is batched over num_envs internally with PyTorch tensors.
    """

    metadata = {"render.modes": []}

    def __init__(
        self,
        num_envs: int = 256,
        dt: float = 0.05,
        max_steps: int = 200,
        device: str = "cpu",  # set to "cpu" if no GPU
        pos_range: tuple = (-2.0, 2.0),  # (min, max) for initial position
        vel_range: tuple = (-1.0, 1.0),   # (min, max) for initial velocity
        pos_bounds: tuple = (-5.0, 5.0)   # (min, max) termination boundaries for position
    ):
        self.num_envs = num_envs
        self.device = device
        self.dt = dt
        self.max_steps = max_steps
        
        # Store randomization ranges
        self.pos_range = pos_range
        self.vel_range = vel_range
        
        # Store position termination boundaries
        self.pos_bounds = pos_bounds

        # state = [position, velocity]
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        # action is continuous acceleration scalar
        self.action_space = Box(low=-3.0, high=3.0, shape=(1,), dtype=np.float32)

        # Add these lines after existing initialization
        super().__init__(num_envs, self.observation_space, self.action_space)

        # Batched tensors for states and counters
        self.pos = torch.zeros((num_envs, 1), dtype=torch.float32, device=self.device)
        self.vel = torch.zeros((num_envs, 1), dtype=torch.float32, device=self.device)
        self.steps = torch.zeros((num_envs,), dtype=torch.int32, device=self.device)

        # Target is always 0 for all environments
        self.targets = torch.zeros((num_envs, 1), dtype=torch.float32, device=self.device)

        # For compatibility with VecEnv step_async/step_wait
        self._actions = None
        self._last_obs = None

        # Initialize the environment and set _last_obs
        initial_obs = self.reset_torch()
        self._last_obs = initial_obs.cpu().numpy()

    def reset_torch(self):
        # Randomize initial pos/vel within specified ranges
        pos_min, pos_max = self.pos_range
        vel_min, vel_max = self.vel_range
        
        self.pos = torch.rand((self.num_envs, 1), device=self.device) * (pos_max - pos_min) + pos_min
        self.vel = torch.rand((self.num_envs, 1), device=self.device) * (vel_max - vel_min) + vel_min
        self.steps = torch.zeros((self.num_envs,), dtype=torch.int32, device=self.device)
        
        # Target is always 0
        self.targets = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)
        
        obs = torch.cat([self.pos, self.vel], dim=1)
        return obs

    def reset(self):
        obs = self.reset_torch()
        self._last_obs = obs.cpu().numpy()  # Store as numpy array
        # return only numpy array for SB3 compatibility
        return obs.cpu().numpy()

    def compute_reward(self, actions_t):
        """Compute reward given current states and actions."""
        dist = torch.abs(self.pos - self.targets)
        
        # Hardcoded weights
        distance_weight = 1.0      # Weight for distance-based reward
        action_penalty_weight = 0.1  # Weight for action penalty
        
        # Reward: 1/(1+d) for getting closer to target, minus action penalty
        distance_reward = distance_weight * (1.0 / (1.0 + dist[:, 0]))
        action_penalty = action_penalty_weight * (actions_t ** 2)[:, 0]
        
        reward = distance_reward - action_penalty
        return reward, dist

    def check_termination(self, dist):
        """Check termination conditions and return done flags and indices."""
        # Check if position is within bounds
        pos_min, pos_max = self.pos_bounds
        done_bounds = (self.pos[:, 0] < pos_min) | (self.pos[:, 0] > pos_max)
        
        # Check if close to target
        done_close = (dist[:, 0] < 0.05)
        
        # Check if max steps reached
        done_max = (self.steps >= self.max_steps)
        
        # Combine all termination conditions
        dones = (done_close | done_max | done_bounds).cpu().numpy()
        
        if dones.any():
            done_idx = np.nonzero(dones)[0]
            return dones, done_idx
        else:
            return dones, np.array([], dtype=np.int64)

    def reset_environments(self, env_ids):
        """Reset specified environments and return updated observations."""
        if len(env_ids) == 0:
            return None
        
        # Convert to tensor indices
        idx_t = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        
        # Sample new initial conditions within specified ranges
        pos_min, pos_max = self.pos_range
        vel_min, vel_max = self.vel_range
        
        new_pos = torch.rand((len(env_ids), 1), device=self.device) * (pos_max - pos_min) + pos_min
        new_vel = torch.rand((len(env_ids), 1), device=self.device) * (vel_max - vel_min) + vel_min
        new_target = torch.zeros((len(env_ids), 1), dtype=torch.float32, device=self.device)  # Always 0
        
        # Reset state variables
        self.pos.index_copy_(0, idx_t, new_pos)
        self.vel.index_copy_(0, idx_t, new_vel)
        self.targets.index_copy_(0, idx_t, new_target)
        self.steps.index_copy_(0, idx_t, torch.zeros((len(env_ids),), dtype=torch.int32, device=self.device))
        
        # Return updated observation
        obs = torch.cat([self.pos, self.vel], dim=1)
        return obs

    def create_info_dict(self, env_ids, dist):
        """Create info dictionaries for environments that are done."""
        infos = [{} for _ in range(self.num_envs)]
        
        pos_min, pos_max = self.pos_bounds
        
        for i in env_ids:
            infos[i]["final_distance"] = float(dist[i, 0].cpu().item())
            infos[i]["steps"] = int(self.steps[i].cpu().item())
            
            # Add termination reason
            pos = float(self.pos[i, 0].cpu().item())
            if pos < pos_min or pos > pos_max:
                infos[i]["termination_reason"] = "out_of_bounds"
            elif dist[i, 0] < 0.05:
                infos[i]["termination_reason"] = "reached_target"
            elif self.steps[i] >= self.max_steps:
                infos[i]["termination_reason"] = "max_steps"
            else:
                infos[i]["termination_reason"] = "unknown"
        
        return infos

    # --- VecEnv required methods ---
    def step_async(self, actions):
        # actions will be a numpy array of shape (num_envs, action_dim)
        # store and execute in step_wait
        self._actions = actions

    def step_wait(self):
        """Execute the step and return observations, rewards, dones, and infos."""
        # Convert actions batch to torch tensor once
        actions_np = self._actions
        actions_t = torch.as_tensor(actions_np, device=self.device, dtype=torch.float32).reshape(self.num_envs, 1)

        # Simple Euler integration: v += a*dt ; x += v*dt
        self.vel = self.vel + actions_t * self.dt
        self.pos = self.pos + self.vel * self.dt

        # Increment steps
        self.steps = self.steps + 1

        # Compute reward
        reward, dist = self.compute_reward(actions_t)

        # Check termination conditions
        dones, done_env_ids = self.check_termination(dist)

        # Prepare observations
        obs = torch.cat([self.pos, self.vel], dim=1)

        # Create info dictionaries for done environments
        infos = self.create_info_dict(done_env_ids, dist)

        # Reset environments that are done
        if len(done_env_ids) > 0:
            obs = self.reset_environments(done_env_ids)

        # Convert obs and reward to numpy arrays (batched)
        obs_np = obs.cpu().numpy()
        reward_np = reward.cpu().numpy()
        
        # Update _last_obs as numpy array
        self._last_obs = obs_np

        return obs_np, reward_np, dones, infos

    def step(self, actions):
        # convenience: synchronous step
        self.step_async(actions)
        return self.step_wait()

    def render(self, mode="human"):
        raise NotImplementedError

    def close(self):
        # nothing special to clean up
        return

    def seed(self, seed=None):
        # optional
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

    def env_is_wrapped(self, wrapper_class, indices=None):
        """Check if environments are wrapped with a given wrapper."""
        return [False] * self.num_envs

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """Call instance methods of vectorized environments."""
        # For this simple environment, we don't have per-env methods to call
        return [None] * self.num_envs

    def get_attr(self, attr_name, indices=None):
        """Get a property from each (sub-)environment."""
        if indices is None:
            indices = range(self.num_envs)
        elif isinstance(indices, int):
            indices = [indices]
        
        # Return the attribute for each requested environment
        # For this environment, most attributes are shared
        if attr_name == "dt":
            return [self.dt] * len(indices)
        elif attr_name == "max_steps":
            return [self.max_steps] * len(indices)
        elif attr_name == "pos_range":
            return [self.pos_range] * len(indices)
        elif attr_name == "vel_range":
            return [self.vel_range] * len(indices)
        elif attr_name == "pos_bounds":
            return [self.pos_bounds] * len(indices)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr_name}'")

    def set_attr(self, attr_name, value, indices=None):
        """Set a property in each (sub-)environment."""
        if indices is None:
            indices = range(self.num_envs)
        elif isinstance(indices, int):
            indices = [indices]
        
        # Set the attribute for the environment
        # For this environment, most attributes are shared across all envs
        if attr_name == "dt":
            self.dt = value
        elif attr_name == "max_steps":
            self.max_steps = value
        elif attr_name == "pos_range":
            self.pos_range = value
        elif attr_name == "vel_range":
            self.vel_range = value
        elif attr_name == "pos_bounds":
            self.pos_bounds = value
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr_name}'")

# -------------------------
# Example training script with SB3 PPO
# -------------------------
def main():
    # Choose device (will use GPU if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    num_envs = 1024  # large vectorized envs
    
    # Create environment with custom ranges and boundaries
    env = VectorizedDoubleIntegrator(
        num_envs=num_envs, 
        device=device,
        pos_range=(-9.0, 9.0),    # Initial position range
        vel_range=(-1.5, 1.5),    # Initial velocity range
        pos_bounds=(-10.0, 10.0)  # Position termination boundaries
    )
    
    # Wrap environment with VecMonitor to track episode rewards
    env = VecMonitor(env)

    # Create PPO model - note: SB3 expects a VecEnv, which our class implements
    # n_steps is # of steps to run per environment per update (choose moderate)
    # For stability: batch_size must divide n_steps * num_envs
    n_steps = 32
    batch_size = 4096  # PPO minibatch size (must be <= n_steps * num_envs)
    # Ensure batch_size <= n_steps * num_envs
    assert batch_size <= n_steps * num_envs

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        device=device,
        n_steps=n_steps,
        batch_size=batch_size,
        tensorboard_log="./ppo_double_integrator_tb/",
        learning_rate=3e-4,
        ent_coef=0.0,
        clip_range=0.2,
        gae_lambda=0.95,
    )

    # Train for a while
    total_timesteps = 2_000_000
    model.learn(total_timesteps=total_timesteps)

    # Save model
    model.save("ppo_double_integrator_torchvec")

    # Close env
    env.close()
    print("Training complete and model saved.")

if __name__ == "__main__":
    main()
