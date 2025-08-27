"""
Double Integrator Inference Script

Load a trained PPO model and perform inference on the double integrator environment.
User can choose between deterministic and stochastic inference.

Hardcoded arguments for IDE execution.
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import datetime

from DoubleIntegratorEnv import VectorizedDoubleIntegrator, Policy, Value
from skrl.utils import set_seed


def get_hardcoded_args():
    """Return hardcoded arguments for IDE execution"""
    class Args:
        def __init__(self):
            # MODIFY THESE SETTINGS AS NEEDED
            self.model_path = "data/models/ppo_double_integrator_skrl_20250827_170020"  # Change to your model path
            self.deterministic = True    # Set to True for deterministic, False for stochastic
            self.stochastic = False      # Set to True for stochastic, False for deterministic
            self.num_episodes = 10       # Number of episodes to run
            self.max_steps = 1000         # Maximum steps per episode
            self.render = True           # Set to True to show plots
            self.save_results = False     # Set to True to save results
            self.device = "cuda"         # "auto", "cuda", or "cpu"
            self.seed = 845               # Random seed
    
    return Args()


def parse_arguments():
    """Parse command line arguments (fallback for command line usage)"""
    parser = argparse.ArgumentParser(description="Double Integrator Inference")
    
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the trained model directory")
    
    parser.add_argument("--deterministic", action="store_true",
                       help="Use deterministic inference (no exploration noise)")
    
    parser.add_argument("--stochastic", action="store_true", 
                       help="Use stochastic inference (with exploration noise)")
    
    parser.add_argument("--num_episodes", type=int, default=10,
                       help="Number of episodes to run")
    
    parser.add_argument("--max_steps", type=int, default=200,
                       help="Maximum steps per episode")
    
    parser.add_argument("--render", action="store_true",
                       help="Render the episodes (plot trajectories)")
    
    parser.add_argument("--save_results", action="store_true",
                       help="Save inference results to file")
    
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "cpu"],
                       help="Device to run inference on")
    
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    return parser.parse_args()


def load_trained_model(model_path, observation_space, action_space, output_range, device):
    """Load the trained policy and value models"""
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model path {model_path} does not exist")
    
    # Initialize models with same architecture as training
    policy = Policy(observation_space, action_space, device, output_range=output_range, clip_actions=True)
    value = Value(observation_space, action_space, device)
    
    # Load policy model
    policy_file = model_path / "policy.pt"
    if policy_file.exists():
        policy.load_state_dict(torch.load(policy_file, map_location=device))
        print(f"Loaded policy from {policy_file}")
    else:
        raise FileNotFoundError(f"Policy file not found: {policy_file}")
    
    # Load value model (optional, not needed for inference)
    value_file = model_path / "value.pt"
    if value_file.exists():
        value.load_state_dict(torch.load(value_file, map_location=device))
        print(f"Loaded value function from {value_file}")
    else:
        print("Value function file not found, proceeding without it")
    
    # Set models to evaluation mode
    policy.eval()
    value.eval()
    
    return policy, value


def run_inference_episode(env, policy, deterministic=True, max_steps=200):
    """Run a single inference episode"""
    # Reset environment for single episode
    obs, _ = env.reset()
    obs = obs[0:1]  # Take only first environment
    
    episode_data = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'positions': [],
        'velocities': [],
        'step': 0,
        'total_reward': 0.0,
        'final_distance': 0.0,
        'success': False
    }
    
    for step in range(max_steps):
        episode_data['observations'].append(obs.cpu().numpy().copy())
        episode_data['positions'].append(float(obs[0, 0].item()))
        episode_data['velocities'].append(float(obs[0, 1].item()))
        
        # Get action from policy
        with torch.no_grad():
            if deterministic:
                # Deterministic inference - use mean of policy distribution
                mean, log_std, _ = policy.compute({"states": obs}, role="evaluation")
                action = mean
            else:
                # Stochastic inference - sample from policy distribution
                action, _, _ = policy.act({"states": obs}, role="policy")
        
        episode_data['actions'].append(action.cpu().numpy().copy())
        
        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        episode_data['rewards'].append(float(reward[0].item()))
        episode_data['total_reward'] += float(reward[0].item())
        episode_data['step'] = step + 1
        
        # Check if episode is done
        done = terminated[0] or truncated[0]
        if done:
            episode_data['final_distance'] = float(abs(next_obs[0, 0].item()))
            episode_data['success'] = episode_data['final_distance'] < 0.05
            break
        
        obs = next_obs
    
    return episode_data


def plot_episode_trajectory(episode_data, episode_num, save_plot=False):
    """Plot the trajectory of a single episode"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Episode {episode_num} - Total Reward: {episode_data["total_reward"]:.3f}')
    
    steps = range(len(episode_data['positions']))
    
    # Position over time
    axes[0, 0].plot(steps, episode_data['positions'], 'b-', linewidth=2)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.7, label='Target')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Position')
    axes[0, 0].set_title('Position Trajectory')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Velocity over time
    axes[0, 1].plot(steps, episode_data['velocities'], 'g-', linewidth=2)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.7)
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Velocity')
    axes[0, 1].set_title('Velocity Trajectory')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Actions over time
    if len(episode_data['actions']) > 0:
        actions = [a[0, 0] for a in episode_data['actions']]
        # Use the same number of steps as actions (actions are taken at each step)
        action_steps = range(len(actions))
        axes[1, 0].plot(action_steps, actions, 'orange', linewidth=2)
        axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.7)
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Action (Acceleration)')
        axes[1, 0].set_title('Actions Taken')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Phase plot (position vs velocity)
    axes[1, 1].plot(episode_data['positions'], episode_data['velocities'], 'purple', linewidth=2)
    axes[1, 1].plot(episode_data['positions'][0], episode_data['velocities'][0], 'go', markersize=8, label='Start')
    axes[1, 1].plot(episode_data['positions'][-1], episode_data['velocities'][-1], 'ro', markersize=8, label='End')
    axes[1, 1].plot(0, 0, 'r*', markersize=12, label='Target')
    axes[1, 1].set_xlabel('Position')
    axes[1, 1].set_ylabel('Velocity')
    axes[1, 1].set_title('Phase Plot')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    if save_plot:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'inference_episode_{episode_num}_{timestamp}.png', dpi=150, bbox_inches='tight')
    
    plt.show()


def main():
    # Use hardcoded arguments for IDE execution
    # Comment out the next line and uncomment the one after if you want command line args
    args = get_hardcoded_args()
    # args = parse_arguments()
    
    print("=== HARDCODED CONFIGURATION ===")
    print(f"Model path: {args.model_path}")
    print(f"Deterministic: {args.deterministic}")
    print(f"Stochastic: {args.stochastic}")
    print(f"Num episodes: {args.num_episodes}")
    print(f"Max steps: {args.max_steps}")
    print(f"Render: {args.render}")
    print(f"Save results: {args.save_results}")
    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")
    print("=" * 35)
    
    # Validate arguments
    if not (args.deterministic or args.stochastic):
        print("Warning: Neither deterministic nor stochastic specified. Using deterministic by default.")
        args.deterministic = True
    
    if args.deterministic and args.stochastic:
        print("Warning: Both deterministic and stochastic specified. Using deterministic.")
        args.stochastic = False
    
    # Set random seed
    set_seed(args.seed)
    
    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    print(f"Inference mode: {'Deterministic' if args.deterministic else 'Stochastic'}")
    
    # Create environment (single environment for inference)
    env = VectorizedDoubleIntegrator(
        num_envs=1,  # Single environment for inference
        device=device,
        max_steps=args.max_steps,
        pos_range=(-9.0, 9.0),
        vel_range=(-3.0, 3.0),
        pos_bounds=(-10.0, 10.0)
    )
    
    print(f"Environment created with {env.num_envs} environment")
    
    # Load trained model
    try:
        output_range=[-1.0, 1.0]

        policy, value = load_trained_model(
            args.model_path, 
            env.observation_space, 
            env.action_space,
            output_range,
            device
        )
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Run inference episodes
    all_episodes = []
    success_count = 0
    total_rewards = []
    final_distances = []
    
    print(f"\nRunning {args.num_episodes} inference episodes...")
    
    for episode in range(args.num_episodes):
        episode_data = run_inference_episode(
            env, 
            policy, 
            deterministic=args.deterministic,
            max_steps=args.max_steps
        )
        
        all_episodes.append(episode_data)
        total_rewards.append(episode_data['total_reward'])
        final_distances.append(episode_data['final_distance'])
        
        if episode_data['success']:
            success_count += 1
        
        print(f"Episode {episode + 1:2d}: "
              f"Steps: {episode_data['step']:3d}, "
              f"Reward: {episode_data['total_reward']:7.3f}, "
              f"Final Dist: {episode_data['final_distance']:6.4f}, "
              f"Success: {'Yes' if episode_data['success'] else 'No'}")
        
        # Render individual episodes if requested
        if args.render and episode < 5:  # Limit to first 5 episodes to avoid too many plots
            plot_episode_trajectory(episode_data, episode + 1, args.save_results)
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("INFERENCE SUMMARY")
    print(f"{'='*60}")
    print(f"Total Episodes:        {args.num_episodes}")
    print(f"Successful Episodes:   {success_count} ({success_count/args.num_episodes*100:.1f}%)")
    print(f"Average Reward:        {np.mean(total_rewards):.3f} ± {np.std(total_rewards):.3f}")
    print(f"Average Final Distance: {np.mean(final_distances):.4f} ± {np.std(final_distances):.4f}")
    print(f"Min Final Distance:    {np.min(final_distances):.4f}")
    print(f"Max Final Distance:    {np.max(final_distances):.4f}")
    
    # Save results if requested
    if args.save_results:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"inference_results_{timestamp}.npz"
        
        np.savez(results_file,
                 episodes=all_episodes,
                 success_count=success_count,
                 total_rewards=total_rewards,
                 final_distances=final_distances,
                 args=vars(args))
        
        print(f"\nResults saved to: {results_file}")
    
    print(f"{'='*60}")


if __name__ == "__main__":
    main()