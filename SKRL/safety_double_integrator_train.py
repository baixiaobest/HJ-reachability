import torch
import datetime
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path

# SKRL imports
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from skrl.models.torch import Model

from DoubleIntegratorEnv import VectorizedDoubleIntegrator, Policy
from algorithms.SafetyDDPG import SafetyDDPG, SAFETY_DDPG_DEFAULT_CONFIG

# Safety critic network
class SafetyValue(Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.num_observations, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        ).to(device)  # Explicitly move to device

    def compute(self, inputs, role):
        states = inputs["states"]
        values = self.net(states)
        return values, {}, {}

    def act(self, inputs, role):
        """Generate value estimates"""
        return self.compute(inputs, role)

# Modified double integrator environment with safety-oriented rewards
class SafetyDoubleIntegrator(VectorizedDoubleIntegrator):
    def __init__(self, num_envs: int = 256,
                 dt: float = 0.05,
                 max_steps: int = 200,
                 device: str = "cpu",  # set to "cpu" if no GPU
                 pos_range: tuple = (-2.0, 2.0),  # (min, max) for initial position
                 vel_range: tuple = (-1.0, 1.0),   # (min, max) for initial velocity
                 pos_bounds: tuple = (-5.0, 5.0),   # (min, max) termination boundaries for position
                 penalty_barrier_bounds=(-5.0, 5.0)):
        super().__init__(
            dt=dt, max_steps=max_steps, device=device, pos_range=pos_range, vel_range=vel_range, pos_bounds=pos_bounds)
        self.penalty_barrier_bounds = penalty_barrier_bounds

    def compute_reward(self, actions_t):
        """
        Compute safety reward based on signed distance to boundary
        Positive when inside bounds, negative when outside
        """
        pos_min, pos_max = self.penalty_barrier_bounds
        
        # Calculate distance to nearest boundary
        dist_to_min = self.pos - pos_min
        dist_to_max = pos_max - self.pos
        
        # Signed distance (positive inside bounds, negative outside)
        signed_dist = torch.min(dist_to_min, dist_to_max)
        
        # # Squeeze to match expected dimensions
        # return signed_dist.squeeze(-1)

        reward = torch.tanh(signed_dist)
        return reward.squeeze(-1)


def plot_safety_critic_heatmap(critic_model, device, pos_range=(-6, 6), vel_range=(-10, 10), 
                              resolution=100, save_path=None, show_plot=True, safety_threshold=0.0):
    """
    Plot safety critic values as a heatmap
    
    Args:
        critic_model: Trained safety critic model
        device: torch device
        pos_range: tuple of (min_pos, max_pos) for plotting
        vel_range: tuple of (min_vel, max_vel) for plotting
        resolution: number of grid points per axis
        save_path: path to save the plot (optional)
        show_plot: whether to display the plot
        safety_threshold: threshold value for safety (default: 0.0)
    """
    critic_model.eval()
    
    # Create grid of positions and velocities
    pos_vals = np.linspace(pos_range[0], pos_range[1], resolution)
    vel_vals = np.linspace(vel_range[0], vel_range[1], resolution)
    
    pos_grid, vel_grid = np.meshgrid(pos_vals, vel_vals)
    
    # Flatten grids and create state tensor
    pos_flat = pos_grid.flatten()
    vel_flat = vel_grid.flatten()
    states = np.column_stack([pos_flat, vel_flat])
    states_tensor = torch.FloatTensor(states).to(device)
    
    # Get critic values
    with torch.no_grad():
        values, _, _ = critic_model.act({"states": states_tensor}, role="critic")
        values_np = values.cpu().numpy().reshape(resolution, resolution)
    
    # Create the heatmap
    plt.figure(figsize=(12, 8))
    
    # Find the range for better color scaling
    vmin, vmax = values_np.min(), values_np.max()
    
    # Handle edge cases for TwoSlopeNorm - ensure proper ordering around safety_threshold
    if vmin >= safety_threshold:
        # All values are above threshold (safe), adjust vmin to include threshold
        vmin = min(safety_threshold - 0.1, vmin - 0.1)
    elif vmax <= safety_threshold:
        # All values are below threshold (unsafe), adjust vmax to include threshold
        vmax = max(safety_threshold + 0.1, vmax + 0.1)
    
    # Create a normalization that centers at safety_threshold
    norm = TwoSlopeNorm(vmin=vmin, vcenter=safety_threshold, vmax=vmax)
    
    # Plot heatmap with custom colormap (red for unsafe, blue for safe)
    im = plt.imshow(values_np, extent=[pos_range[0], pos_range[1], vel_range[0], vel_range[1]], 
                    origin='lower', aspect='auto', cmap='RdBu', norm=norm)
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Safety Value', rotation=270, labelpad=20)
    
    # Add a horizontal line on colorbar to mark the safety threshold
    cbar.ax.axhline(y=safety_threshold, color='black', linewidth=2, linestyle='-')
    cbar.ax.text(0.5, safety_threshold, f'{safety_threshold} (Safety Threshold)', 
                 transform=cbar.ax.transData, ha='left', va='center', fontweight='bold')
    
    # Add boundary lines (assuming pos_bounds = (-5, 5))
    plt.axvline(x=-5, color='black', linestyle='--', linewidth=3, label='Position Boundaries')
    plt.axvline(x=5, color='black', linestyle='--', linewidth=3)
    
    # Labels and title
    plt.xlabel('Position', fontsize=12)
    plt.ylabel('Velocity', fontsize=12)
    plt.title(f'Safety Critic Heatmap\n(Blue: Safe regions (value > {safety_threshold}), Red: Unsafe regions (value ≤ {safety_threshold}))', 
              fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add text annotations for interpretation
    plt.text(0.02, 0.98, f'Blue: Safe regions (value > {safety_threshold})', transform=plt.gca().transAxes, 
             verticalalignment='top', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    plt.text(0.02, 0.92, f'Red: Unsafe regions (value ≤ {safety_threshold})', transform=plt.gca().transAxes, 
             verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    # Add statistics based on safety threshold
    safe_percentage = (values_np > safety_threshold).mean() * 100
    plt.text(0.02, 0.86, f'Safe region: {safe_percentage:.1f}%', transform=plt.gca().transAxes, 
             verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # Add value range information
    plt.text(0.02, 0.80, f'Value range: [{vmin:.2f}, {vmax:.2f}]', transform=plt.gca().transAxes, 
             verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # Add safety threshold information
    plt.text(0.02, 0.74, f'Safety threshold: {safety_threshold}', transform=plt.gca().transAxes, 
             verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    if save_path:
        heatmap_save_path = save_path.replace('.png', '_value.png')
        plt.savefig(heatmap_save_path, dpi=300, bbox_inches='tight')
        print(f"Safety critic heatmap saved to {heatmap_save_path}")
    
    # Second plot - Binary safety set visualization
    plt.figure(figsize=(12, 8))
    
    # Create binary safety mask
    safety_mask = values_np > safety_threshold
    
    # Plot binary heatmap (red for unsafe, blue for safe)
    im2 = plt.imshow(safety_mask, extent=[pos_range[0], pos_range[1], vel_range[0], vel_range[1]], 
                     origin='lower', aspect='auto', cmap='RdBu')
    
    # Add colorbar for binary visualization
    cbar2 = plt.colorbar(im2)
    cbar2.set_label('Safety Set (Binary)', rotation=270, labelpad=20)
    cbar2.set_ticks([0, 1])
    cbar2.set_ticklabels(['Unsafe', 'Safe'])
    
    # Add boundary lines (assuming pos_bounds = (-5, 5))
    plt.axvline(x=-5, color='black', linestyle='--', linewidth=3, label='Position Boundaries')
    plt.axvline(x=5, color='black', linestyle='--', linewidth=3)
    
    # Labels and title for second plot
    plt.xlabel('Position', fontsize=12)
    plt.ylabel('Velocity', fontsize=12)
    plt.title(f'Binary Safety Set\n(Blue: Safe regions (value > {safety_threshold}), Red: Unsafe regions (value ≤ {safety_threshold}))', 
              fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add text annotations for binary interpretation
    plt.text(0.02, 0.98, f'Blue: Safe regions (value > {safety_threshold})', transform=plt.gca().transAxes, 
             verticalalignment='top', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    plt.text(0.02, 0.92, f'Red: Unsafe regions (value ≤ {safety_threshold})', transform=plt.gca().transAxes, 
             verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    # Add statistics for binary visualization
    safe_percentage_binary = safety_mask.mean() * 100
    plt.text(0.02, 0.86, f'Safe region: {safe_percentage_binary:.1f}%', transform=plt.gca().transAxes, 
             verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # Add safety threshold information
    plt.text(0.02, 0.80, f'Safety threshold: {safety_threshold}', transform=plt.gca().transAxes, 
             verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # Save binary plot if save_path is provided
    if save_path:
        binary_save_path = save_path.replace('.png', '_binary.png')
        plt.savefig(binary_save_path, dpi=300, bbox_inches='tight')
        print(f"Binary safety set heatmap saved to {binary_save_path}")
    
    if show_plot:
        plt.show()

def plot_policy_action_heatmap(policy_model, device, pos_range=(-6, 6), vel_range=(-10, 10), 
                             resolution=100, save_path=None, show_plot=True):
    """
    Plot policy actions as a heatmap
    
    Args:
        policy_model: Trained policy model
        device: torch device
        pos_range: tuple of (min_pos, max_pos) for plotting
        vel_range: tuple of (min_vel, max_vel) for plotting
        resolution: number of grid points per axis
        save_path: path to save the plot (optional)
        show_plot: whether to display the plot
    """
    policy_model.eval()
    
    # Create grid of positions and velocities
    pos_vals = np.linspace(pos_range[0], pos_range[1], resolution)
    vel_vals = np.linspace(vel_range[0], vel_range[1], resolution)
    
    pos_grid, vel_grid = np.meshgrid(pos_vals, vel_vals)
    
    # Flatten grids and create state tensor
    pos_flat = pos_grid.flatten()
    vel_flat = vel_grid.flatten()
    states = np.column_stack([pos_flat, vel_flat])
    states_tensor = torch.FloatTensor(states).to(device)
    
    # Get policy actions
    with torch.no_grad():
        actions, _, _ = policy_model.act({"states": states_tensor}, role="evaluation")
        actions_np = actions.cpu().numpy().reshape(resolution, resolution)
    
    # Create the heatmap
    plt.figure(figsize=(12, 8))
    
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0, vmax=1.0)
    
    # Plot heatmap with custom colormap (red for unsafe, blue for safe)
    im = plt.imshow(actions_np, extent=[pos_range[0], pos_range[1], vel_range[0], vel_range[1]], 
                    origin='lower', aspect='auto', cmap='RdBu', norm=norm)
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Action (Acceleration)', rotation=270, labelpad=20)
    
    # Add boundary lines (assuming pos_bounds = (-5, 5))
    plt.axvline(x=-5, color='black', linestyle='--', linewidth=3, label='Position Boundaries')
    plt.axvline(x=5, color='black', linestyle='--', linewidth=3)
    
    # Labels and title
    plt.xlabel('Position', fontsize=12)
    plt.ylabel('Velocity', fontsize=12)
    plt.title('Policy Action Heatmap', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add action range information
    action_min, action_max = actions_np.min(), actions_np.max()
    plt.text(0.02, 0.98, f'Action range: [{action_min:.2f}, {action_max:.2f}]', 
             transform=plt.gca().transAxes, verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # Add color interpretation
    plt.text(0.02, 0.92, 'Blue: Positive acceleration (right)', 
             transform=plt.gca().transAxes, verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    plt.text(0.02, 0.86, 'Red: Negative acceleration (left)', 
             transform=plt.gca().transAxes, verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Action heatmap saved to {save_path}")
    
    if show_plot:
        plt.show()

def load_and_plot_critic(critic_path, policy_path=None, device="cpu", safety_threshold=0.0):
    """
    Load a saved critic model and policy model (if provided) and plot their heatmaps
    
    Args:
        critic_path: path to the saved critic state dict
        policy_path: optional path to the saved policy state dict
        device: torch device
        safety_threshold: threshold value for safety (default: 0.0)
    """
    # Create critic model (same architecture as in training)
    from gymnasium.spaces import Box
    observation_space = Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
    action_space = Box(low=-3.0, high=3.0, shape=(1,), dtype=np.float32)
    
    # Load and plot critic
    critic = SafetyValue(observation_space, action_space, device)
    critic.load_state_dict(torch.load(critic_path, map_location=device))
    
    critic_plot_path = critic_path.replace('.pt', f'_heatmap_thresh{safety_threshold}.png')
    plot_safety_critic_heatmap(critic, device, save_path=critic_plot_path, resolution=200,
                             safety_threshold=safety_threshold, show_plot=False)
    
    # Load and plot policy if provided
    if policy_path is not None:
        policy = Policy(observation_space, action_space, device, output_range=[-3.0, 3.0])
        policy.load_state_dict(torch.load(policy_path, map_location=device))
        
        policy_plot_path = policy_path.replace('.pt', '_action_heatmap.png')
        plot_policy_action_heatmap(policy, device, save_path=policy_plot_path, show_plot=False)
    
    plt.show()

def max_next_value(sampled_states, sampled_next_states, next_state_q_values, target_critic):
    """
    Compute the maximum next value for the safety critic
    This is only an approximation since we cannot solve the maximization exactly.
        max_u V(x + f(x, u) * dt)
    """
    max_next_val = torch.ones_like(next_state_q_values)
    in_goal = (torch.norm(sampled_states, dim=1) < 0.01).unsqueeze(-1)
    max_next_val[~in_goal] = next_state_q_values[~in_goal]

    return max_next_val

def main(policy_path, critic_path=None):
    """
    Train a safety critic for the double integrator environment
    
    Args:
        critic_path: Optional path to a pre-trained critic model to continue training
    """
    # Set random seed for reproducibility
    set_seed(42)
    
    # Choose device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    num_envs = 1024
    
    # Create timestamp for unique experiment name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create safety-focused environment
    env = SafetyDoubleIntegrator(
        num_envs=num_envs, 
        device=device,
        pos_range=(-6.0, 6.0),     # Initial position range
        vel_range=(-10.0, 10.0),     # Initial velocity range
        pos_bounds=(-6.0, 6.0),     # Position termination boundaries
        penalty_barrier_bounds=(-5.0, 5.0)  # Bounds for safety reward calculation
    )
    
    # Configuration for memory
    memory = RandomMemory(
        memory_size=100000,
        num_envs=num_envs,
        device=device
    )
    
    # Create behavior policy (could be pre-trained or random)
    behavior_policy = Policy(env.observation_space, env.action_space, device, 
                        output_range=[-3.0, 3.0], clip_actions=True)
    
    # Create safety critic and target critic
    critic = SafetyValue(env.observation_space, env.action_space, device)
    target_critic = SafetyValue(env.observation_space, env.action_space, device)

    # Load behavior policy if available
    policy_path = Path(policy_path)
    if policy_path.exists():
        behavior_policy.load_state_dict(torch.load(policy_path, map_location=device))
        print(f"Loaded policy from {policy_path}")
    else:
        raise FileNotFoundError(f"Policy file not found: {policy_path}")
    
    # Load critic if path is provided
    if critic_path is not None:
        critic_path = Path(critic_path)
        if critic_path.exists():
            critic.load_state_dict(torch.load(critic_path, map_location=device))
            # Copy weights to target critic as well
            target_critic.load_state_dict(critic.state_dict())
            print(f"Loaded critic from {critic_path}")
            print("Continuing training from the loaded critic...")
        else:
            print(f"Warning: Critic file not found: {critic_path}")
            print("Starting with a new critic...")
    else:
        print("Starting with a new critic...")
    
    # Configure SafetyDDPG
    total_timesteps = 20000
    safety_ddpg_config = copy.deepcopy(SAFETY_DDPG_DEFAULT_CONFIG)
    safety_ddpg_config.update({
        "learning_starts": 1000,
        "gradient_steps": 1,
        "batch_size": 4096,
        "discount_factor": 0.99,
        "discount_factor_scheduler": {
                "final_gamma": 0.9999, 
                "total_timesteps": total_timesteps, 
                "schedule_type": "linear"
        },
        "critic_learning_rate": 3e-4,
        "polyak": 0.05,
        "experiment": {
            "directory": "data/skrl_logs",            # experiment's parent directory
            "experiment_name": f"safety_ddpg_double_integrator_{timestamp}",      # experiment name
            "write_interval": 100,   # TensorBoard writing interval (timesteps)
            "checkpoint_interval": 1000,      # interval for checkpoints (timesteps)
        },
    })
    
    # Create SafetyDDPG agent
    agent = SafetyDDPG(
        models={"behavior_policy": behavior_policy, 
                "critic": critic, 
                "target_critic": target_critic},
        memory=memory,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        cfg=safety_ddpg_config,
        max_next_value_func=max_next_value
    )
    
    # Configure trainer
    trainer_config = {
        "timesteps": total_timesteps,
        "headless": True
    }
    
    # Create trainer
    trainer = SequentialTrainer(
        cfg=trainer_config,
        env=env,
        agents=agent
    )
    
    # Train the safety critic
    trainer.train()
    
    # Save safety critic model
    save_path = f"./data/models/safety_ddpg_double_integrator_{timestamp}"
    os.makedirs(save_path, exist_ok=True)
    
    target_critic_path = os.path.join(save_path, "safety_critic.pt")
    torch.save(agent.models["target_critic"].state_dict(), target_critic_path)
    print(f"Training complete. Safety critic saved to {target_critic_path}")
    
    # Plot and save heatmap with customizable safety threshold
    heatmap_path = os.path.join(save_path, "safety_critic_heatmap.png")
    safety_threshold = 0.0  # You can change this value as needed
    plot_safety_critic_heatmap(agent.models["critic"], device, save_path=heatmap_path, 
                              show_plot=True, safety_threshold=safety_threshold)
    
    print("Safety critic training complete. You can now use the safety critic to evaluate states.")
    print("Lower values indicate less safe states (closer to or beyond boundaries).")
    print("Higher values indicate safer states (further from boundaries).")
    print(f"Heatmap visualization saved to {heatmap_path}")

if __name__ == "__main__":
    # Examples of how to load and plot an existing model with different thresholds:
    load = True
    # load = False
    policy_path="data/models/ppo_double_integrator_skrl_20250827_170020/policy.pt"
    critic_path="data/models/safety_ddpg_double_integrator_20250827_173450/safety_critic.pt"
    
    if load:
        # Load and plot both critic and policy
        load_and_plot_critic(critic_path=critic_path, 
                           policy_path=policy_path, 
                           device="cpu", 
                           safety_threshold=0.40)
    else:
        main(policy_path=policy_path, critic_path=None)