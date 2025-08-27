"""
skrl_ppo_double_integrator_torch_vectorized.py

Requirements:
    pip install skrl gymnasium torch
"""

import torch
import datetime

# SKRL imports
from PPOEnhanced import PPOWithEntropyAnnealing
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

from DoubleIntegratorEnv import VectorizedDoubleIntegrator, Policy, Value

# -------------------------
# Example training script with SKRL PPO
# -------------------------
def main():
    # Set random seed for reproducibility
    set_seed(42)
    
    # Choose device (will use GPU if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    num_envs = 2048
    
    # Create timestamp for unique experiment name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create environment with custom ranges and boundaries
    env = VectorizedDoubleIntegrator(
        num_envs=num_envs, 
        device=device,
        max_steps=1000,
        pos_range=(-9.0, 9.0),    # Initial position range
        vel_range=(-3.0, 3.0),    # Initial velocity range
        pos_bounds=(-10.0, 10.0)  # Position termination boundaries
    )
    
    # Don't wrap environment for now
    # env = Wrapper(env)

    n_steps = 32  # Keep rollout steps small
    
    # Define memory with correct configuration
    memory = RandomMemory(
        memory_size=n_steps,
        num_envs=num_envs,
        device=device
    )
    
    # Define models
    models_config = {
        "policy": {
            "output_range": [-1.0, 1.0],
            "clip_actions": True,
        },
        "value": {}
    }
    
    policy = Policy(env.observation_space, env.action_space, device, 
                    **models_config["policy"])
    value = Value(env.observation_space, env.action_space, device, **models_config["value"])
    
    # Configure PPO with unique experiment name
    ppo_config = PPO_DEFAULT_CONFIG.copy()
    ppo_config.update({
        "learning_rate": 1e-4,  # Lower learning rate
        "batch_size": 4096,     
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "entropy_coef": 0.0,
        "num_epochs": 10,       
        "discount_factor": 0.99,
        "learning_starts": 0,
        "rollouts": n_steps,  # Match rollout steps
        "grad_norm_clip": 0.5,  # Add gradient clipping
        "verbose": 1,
        "experiment": {
            "directory": "./data/skrl_logs",
            "experiment_name": f"ppo_double_integrator_{timestamp}",  # Unique name
            "write_interval": 10,  # Very frequent logging for debugging
        },

        # Simple entropy annealing configuration
        "entropy_annealing_enabled": True,    # Enable/disable entropy annealing
        "initial_entropy_coef": 0.02,         # Starting entropy coefficient
        "final_entropy_coef": 0.001,          # Final entropy coefficient
        "annealing_schedule": "exponential",  # "linear" or "exponential"
        "exponential_decay_rate": 0.01,       # Decay rate for exponential (smaller = faster decay)
    })
    
    # Create PPO agent
    agent = PPOWithEntropyAnnealing(
        models={"policy": policy, "value": value},
        memory=memory,
        cfg=ppo_config,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device
    )
    
    # Configure trainer with very small timesteps for testing
    trainer_config = {
        "timesteps": 20000,
        "headless": True
    }
    
    # Create trainer
    trainer = SequentialTrainer(
        cfg=trainer_config,
        env=env,
        agents=agent
    )
    
    # Train the agent
    trainer.train()
    
    # Save agent models
    save_path = f"./data/models/ppo_double_integrator_skrl_{timestamp}"
    # Fallback to manual saving
    import os
    os.makedirs(save_path, exist_ok=True)
    
    torch.save(agent.models["policy"].state_dict(), 
                os.path.join(save_path, "policy.pt"))
    torch.save(agent.models["value"].state_dict(), 
                os.path.join(save_path, "value.pt"))
    print("Training complete and model saved.")


if __name__ == "__main__":
    main()