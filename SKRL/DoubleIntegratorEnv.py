import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from skrl.models.torch import Model, GaussianMixin

# -------------------------
# Vectorized double integrator implemented in torch
# -------------------------
class VectorizedDoubleIntegrator(gym.vector.VectorEnv):
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
        super().__init__()

        # Define observation and action spaces
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-3.0, high=3.0, shape=(1,), dtype=np.float32)
        
        self.num_envs = num_envs
        self.device = device
        self.dt = dt
        self.max_steps = max_steps
        self.num_agents = 1
        
        # Store randomization ranges
        self.pos_range = pos_range
        self.vel_range = vel_range
        
        # Store position termination boundaries
        self.pos_bounds = pos_bounds

        # Batched tensors for states and counters
        self.pos = torch.zeros((num_envs, 1), dtype=torch.float32, device=self.device)
        self.vel = torch.zeros((num_envs, 1), dtype=torch.float32, device=self.device)
        self.steps = torch.zeros((num_envs,), dtype=torch.int32, device=self.device)

        # Target is always 0 for all environments
        self.targets = torch.zeros((num_envs, 1), dtype=torch.float32, device=self.device)

    # Add missing methods required by SKRL
    def seed(self, seed=None):
        """Set the seed for reproducibility"""
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        return [seed] * self.num_envs

    def close(self):
        """Close the environment"""
        # Nothing special to clean up for this environment
        pass

    def render(self, mode="human"):
        """Render the environment (not implemented for this example)"""
        pass

    # Add method to reset specific environments (required by some SKRL wrappers)
    def reset_at(self, indices):
        """Reset environments at specific indices"""
        if not isinstance(indices, (list, tuple, np.ndarray)):
            indices = [indices]
        
        self.reset_environments(indices)
        obs = torch.cat([self.pos, self.vel], dim=1)
        return obs[indices], [{} for _ in indices]

    def reset(self):
        """Reset all environments and return initial observations"""
        # Reset all environments
        obs = self.reset_torch()
        
        # For compatibility with gymnasium - return observations and empty info dict
        info = [{} for _ in range(self.num_envs)]
        return obs, info

    def reset_torch(self):
        """Reset all environments and return initial torch observations"""
        # Randomize initial pos/vel within specified ranges
        pos_min, pos_max = self.pos_range
        vel_min, vel_max = self.vel_range
        
        self.pos = torch.rand((self.num_envs, 1), device=self.device) * (pos_max - pos_min) + pos_min
        self.vel = torch.rand((self.num_envs, 1), device=self.device) * (vel_max - vel_min) + vel_min
        self.steps = torch.zeros((self.num_envs,), dtype=torch.int32, device=self.device)
        
        # Target is always 0
        self.targets = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)
        
        # Concatenate position and velocity for observation
        obs = torch.cat([self.pos, self.vel], dim=1)
        return obs

    def step(self, actions):
        """Execute one step in all environments and return results"""
        # Convert actions to torch tensor if needed
        if isinstance(actions, np.ndarray):
            actions_t = torch.as_tensor(actions, device=self.device, dtype=torch.float32).reshape(self.num_envs, 1)
        else:
            actions_t = actions.reshape(self.num_envs, 1)
        
        # Check for NaN and large values in actions
        if torch.isnan(actions_t).any():
            print("WARNING: NaN detected in actions!")
            print(f"NaN actions count: {torch.isnan(actions_t).sum().item()}/{actions_t.numel()}")
            actions_t = torch.nan_to_num(actions_t, nan=0.0)
        
        # Check for large action values
        large_action_threshold = 100.0
        if torch.abs(actions_t).max() > large_action_threshold:
            print(f"WARNING: Large actions detected! Max: {torch.abs(actions_t).max().item():.6f}")
            print(f"Actions above threshold: {(torch.abs(actions_t) > large_action_threshold).sum().item()}")
            large_action_indices = torch.where(torch.abs(actions_t) > large_action_threshold)[0]
            print(f"Large action values: {actions_t[large_action_indices].flatten()}")
        
        # Update with bounded dynamics
        self.vel = self.vel + actions_t * self.dt
        
        self.pos = self.pos + self.vel * self.dt
        

        # Increment steps
        self.steps = self.steps + 1

        # Compute reward
        reward = self.compute_reward(actions_t)
        
        # Check termination conditions
        dones, terminated, truncated, done_env_ids = self.check_termination()

        # Create info dictionaries
        infos = self.create_info_dict(done_env_ids)

        # Prepare observations
        obs = torch.cat([self.pos, self.vel], dim=1)

        # Reset environments that are done
        if len(done_env_ids) > 0:
            self.reset_environments(done_env_ids)

        # Return everything as torch tensors
        return obs, reward.unsqueeze(-1), terminated.unsqueeze(-1), truncated.unsqueeze(-1), infos

    def compute_reward(self, actions_t):
        """Compute reward given current states and actions."""
        dist = torch.abs(self.pos - self.targets).squeeze(-1)
        
        # Hardcoded weights
        distance_weight = 1.0      # Weight for distance-based reward
        action_penalty_weight = 0.1  # Weight for action penalty
        
        # Reward: 1/(1+d) for getting closer to target, minus action penalty
        # Add small epsilon for numerical stability
        epsilon = 1e-6
        distance_reward = 1.0 / (1.0 + dist + epsilon)
        action_penalty = action_penalty_weight * torch.sum(torch.square(actions_t), dim=1)
        
        reward = distance_reward - action_penalty
        
        return reward

    def check_termination(self):
        """Check termination conditions and return done flags and indices."""
        dist = torch.abs(self.pos - self.targets).squeeze(-1)

        # Check if position is within bounds
        pos_min, pos_max = self.pos_bounds
        done_bounds = (self.pos < pos_min) | (self.pos > pos_max)
        done_bounds = done_bounds.squeeze(-1)
        
        # Check if close to target
        # done_close = (dist < 0.05)
        
        # Check if max steps reached
        done_max = (self.steps >= self.max_steps)
        
        # Combine all termination conditions
        terminated = (done_bounds)  # True termination
        truncated = done_max                     # Truncated due to max steps
        dones = terminated | truncated           # Combined done flag
        
        done_env_ids = []
        if dones.any():
            done_env_ids = torch.nonzero(dones).squeeze(-1).cpu().tolist()
            if not isinstance(done_env_ids, list):
                done_env_ids = [done_env_ids]
        
        return dones, terminated, truncated, done_env_ids

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

    def create_info_dict(self, env_ids):
        """Create info dictionaries for all environments."""
        infos = {}  # SKRL expects a dict of dicts format

        dist = torch.abs(self.pos - self.targets)
        
        # Add episode information for done environments
        pos_min, pos_max = self.pos_bounds
        
        # Add special tracking for terminated environments
        if env_ids:
            infos["final_distance"] = torch.zeros(self.num_envs, device=self.device)
            infos["steps"] = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
            infos["termination_reason"] = ["" for _ in range(self.num_envs)]
            
            for i in env_ids:
                infos["final_distance"][i] = dist[i, 0]
                infos["steps"][i] = self.steps[i]
                
                # Add termination reason
                pos = float(self.pos[i, 0].item())
                if pos < pos_min or pos > pos_max:
                    infos["termination_reason"][i] = "out_of_bounds"
                elif dist[i, 0] < 0.05:
                    infos["termination_reason"][i] = "reached_target"
                elif self.steps[i] >= self.max_steps:
                    infos["termination_reason"][i] = "max_steps"
                else:
                    infos["termination_reason"][i] = "unknown"
        
        return infos
    

# -------------------------
# Define policy and value networks for SKRL
# -------------------------
class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, output_range=[-3.0, 3.0], clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions)
        
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        ).to(device)  # Explicitly move to device
        
        self.mean_layer = nn.Linear(64, self.num_actions).to(device)  # Move to device
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions, device=device))  # Create on device
        self.output_range = output_range

    def compute(self, inputs, role):
        states = inputs["states"]
        x = self.net(states)
        mean = self.mean_layer(x)
        
        # Clip mean values to prevent extreme actions
        mean = torch.clamp(mean, self.output_range[0], self.output_range[1])
        
        log_std = torch.clamp(self.log_std_parameter, -20.0, 2.0)
        log_std = log_std.expand_as(mean)
        
        return mean, log_std, {}

    def act(self, inputs, role):
        """Generate actions for the policy"""
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        return self.compute(inputs, role)


class Value(Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        ).to(device)  # Explicitly move to device

    def compute(self, inputs, role):
        states = inputs["states"]
        values = self.net(states)
        return values, {}, {}

    def act(self, inputs, role):
        """Generate value estimates"""
        return self.compute(inputs, role)