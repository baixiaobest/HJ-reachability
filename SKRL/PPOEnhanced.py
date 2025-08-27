"""
skrl_ppo_double_integrator_torch_vectorized.py

Requirements:
    pip install skrl gymnasium torch
"""

import torch
import datetime
import math

# SKRL imports
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

from DoubleIntegratorEnv import VectorizedDoubleIntegrator, Policy, Value

# -------------------------
# Custom PPO Agent with Simple Entropy Coefficient Annealing
# -------------------------
class PPOWithEntropyAnnealing(PPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Extract annealing configuration
        self.entropy_annealing_enabled = self.cfg.get("entropy_annealing_enabled", False)
        self.initial_entropy_coef = self.cfg.get("initial_entropy_coef", 0.01)
        self.final_entropy_coef = self.cfg.get("final_entropy_coef", 0.0)
        self.annealing_schedule = self.cfg.get("annealing_schedule", "linear")  # "linear" or "exponential"
        self.exponential_decay_rate = self.cfg.get("exponential_decay_rate", 0.01)  # Only used for exponential
        
        # Set initial entropy coefficient
        if self.entropy_annealing_enabled:
            self.cfg["entropy_coef"] = self.initial_entropy_coef
            print(f"Entropy annealing enabled ({self.annealing_schedule}): {self.initial_entropy_coef} → {self.final_entropy_coef}")
        else:
            print(f"Entropy annealing disabled: fixed at {self.cfg['entropy_coef']}")
        
    def pre_interaction(self, timestep: int, timesteps: int):
        super().pre_interaction(timestep, timesteps)
        
        if self.entropy_annealing_enabled:
            # Calculate progress ratio (0 at start, 1 at end)
            progress_ratio = timestep / timesteps
            
            if self.annealing_schedule == "linear":
                # Linear annealing from initial to final
                current_entropy_coef = self.initial_entropy_coef + progress_ratio * (self.final_entropy_coef - self.initial_entropy_coef)
            elif self.annealing_schedule == "exponential":
                # Exponential decay: initial * (decay_rate ^ progress) + final
                current_entropy_coef = self.initial_entropy_coef * (self.exponential_decay_rate ** progress_ratio) + self.final_entropy_coef
            else:
                # Fallback to linear
                current_entropy_coef = self.initial_entropy_coef + progress_ratio * (self.final_entropy_coef - self.initial_entropy_coef)
            
            self.cfg["entropy_coef"] = current_entropy_coef
            
            # Log occasionally
            if timestep % 1000 == 0:
                print(f"Step {timestep}/{timesteps}: Entropy coef = {current_entropy_coef:.6f}")