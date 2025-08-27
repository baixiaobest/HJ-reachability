from typing import Any, Mapping, Optional, Tuple, Union

import copy
import gymnasium
from packaging import version

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config, logger
from skrl.agents.torch import Agent
from skrl.memories.torch import Memory
from skrl.models.torch import Model


# fmt: off
# [start-config-dict-torch]
SAFETY_DDPG_DEFAULT_CONFIG = {
    "gradient_steps": 1,            # gradient steps
    "batch_size": 64,               # training batch size

    "discount_factor": 0.99,        # initial discount factor (gamma)
    "discount_factor_scheduler": None,  # discount factor scheduler config
    "polyak": 0.005,                # soft update hyperparameter (tau)

    "actor_learning_rate": 1e-3,    # actor learning rate
    "critic_learning_rate": 1e-3,   # critic learning rate
    "learning_rate_scheduler": None,        # learning rate scheduler class (see torch.optim.lr_scheduler)
    "learning_rate_scheduler_kwargs": {},   # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})

    "state_preprocessor": None,             # state preprocessor class (see skrl.resources.preprocessors)
    "state_preprocessor_kwargs": {},        # state preprocessor's kwargs (e.g. {"size": env.observation_space})

    "learning_starts": 0,           # learning starts after this many steps

    "grad_norm_clip": 0,            # clipping coefficient for the norm of the gradients

    "mixed_precision": False,       # enable automatic mixed precision for higher performance

    "experiment": {
        "directory": "",            # experiment's parent directory
        "experiment_name": "",      # experiment name
        "write_interval": "auto",   # TensorBoard writing interval (timesteps)

        "checkpoint_interval": "auto",      # interval for checkpoints (timesteps)
        "store_separately": False,          # whether to store checkpoints separately

        "wandb": False,             # whether to use Weights & Biases
        "wandb_kwargs": {}          # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
    },
}
# [end-config-dict-torch]
# fmt: on


class DiscountFactorScheduler:
    """Custom scheduler for discount factor annealing"""
    
    def __init__(self, initial_gamma=0.99, final_gamma=0.999, total_timesteps=50000, 
                 schedule_type="linear"):
        self.initial_gamma = initial_gamma
        self.final_gamma = final_gamma
        self.total_timesteps = total_timesteps
        self.schedule_type = schedule_type
        self.current_gamma = initial_gamma
        
    def step(self, timestep):
        """Update discount factor based on current timestep"""
        if timestep >= self.total_timesteps:
            self.current_gamma = self.final_gamma
            return self.current_gamma
            
        progress = timestep / self.total_timesteps
        
        if self.schedule_type == "linear":
            self.current_gamma = self.initial_gamma + (self.final_gamma - self.initial_gamma) * progress
        elif self.schedule_type == "exponential":
            # Exponential decay
            decay_rate = (self.final_gamma / self.initial_gamma) ** (1 / self.total_timesteps)
            self.current_gamma = self.initial_gamma * (decay_rate ** timestep)
        elif self.schedule_type == "cosine":
            # Cosine annealing
            import math
            self.current_gamma = self.final_gamma + (self.initial_gamma - self.final_gamma) * \
                               (1 + math.cos(math.pi * progress)) / 2
        
        return self.current_gamma
    
    def get_gamma(self):
        """Get current discount factor"""
        return self.current_gamma


class SafetyDDPG(Agent):
    def __init__(
        self,
        models: Mapping[str, Model],
        memory: Optional[Union[Memory, Tuple[Memory]]] = None,
        observation_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        action_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[dict] = None,
        max_next_value_func: callable = None
    ) -> None:
        """Deep Deterministic Policy Gradient (DDPG)

        https://arxiv.org/abs/1509.02971

        :param models: Models used by the agent
        :type models: dictionary of skrl.models.torch.Model
        :param memory: Memory to storage the transitions.
                       If it is a tuple, the first element will be used for training and
                       for the rest only the environment transitions will be added
        :type memory: skrl.memory.torch.Memory, list of skrl.memory.torch.Memory or None
        :param observation_space: Observation/state space or shape (default: ``None``)
        :type observation_space: int, tuple or list of int, gymnasium.Space or None, optional
        :param action_space: Action space or shape (default: ``None``)
        :type action_space: int, tuple or list of int, gymnasium.Space or None, optional
        :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda"`` if available or ``"cpu"``
        :type device: str or torch.device, optional
        :param cfg: Configuration dictionary
        :type cfg: dict

        :raises KeyError: If the models dictionary is missing a required key
        """
        _cfg = copy.deepcopy(SAFETY_DDPG_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            cfg=_cfg,
        )

        # models
        self.behavior_policy = self.models.get("behavior_policy", None)
        self.critic = self.models.get("critic", None)
        self.target_critic = self.models.get("target_critic", None)

        # checkpoint models
        self.checkpoint_modules["critic"] = self.critic
        self.checkpoint_modules["target_critic"] = self.target_critic

        # broadcast models' parameters in distributed runs
        if config.torch.is_distributed:
            logger.info(f"Broadcasting models' parameters")
            if self.policy is not None:
                self.policy.broadcast_parameters()
            if self.critic is not None:
                self.critic.broadcast_parameters()

        if self.target_critic is not None:
            # freeze target networks with respect to optimizers (update via .update_parameters())
            self.target_critic.freeze_parameters(True)

            # update target networks (hard update)
            self.target_critic.update_parameters(self.critic, polyak=1)

        # configuration
        self._gradient_steps = self.cfg["gradient_steps"]
        self._batch_size = self.cfg["batch_size"]

        self._discount_factor = self.cfg["discount_factor"]
        self._polyak = self.cfg["polyak"]

        self._actor_learning_rate = self.cfg["actor_learning_rate"]
        self._critic_learning_rate = self.cfg["critic_learning_rate"]
        self._learning_rate_scheduler = self.cfg["learning_rate_scheduler"]

        self._state_preprocessor = self.cfg["state_preprocessor"]

        self._learning_starts = self.cfg["learning_starts"]

        self._grad_norm_clip = self.cfg["grad_norm_clip"]

        self._mixed_precision = self.cfg["mixed_precision"]

        # set up automatic mixed precision
        self._device_type = torch.device(device).type
        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.amp.GradScaler(device=self._device_type, enabled=self._mixed_precision)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self._mixed_precision)

        # set up optimizers and learning rate schedulers
        if self.critic is not None:
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self._critic_learning_rate)
            if self._learning_rate_scheduler is not None:
                self.critic_scheduler = self._learning_rate_scheduler(
                    self.critic_optimizer, **self.cfg["learning_rate_scheduler_kwargs"]
                )

            self.checkpoint_modules["critic_optimizer"] = self.critic_optimizer

        # set up preprocessors
        if self._state_preprocessor:
            self._state_preprocessor = self._state_preprocessor(**self.cfg["state_preprocessor_kwargs"])
            self.checkpoint_modules["state_preprocessor"] = self._state_preprocessor
        else:
            self._state_preprocessor = self._empty_preprocessor

        # Set up discount factor scheduler
        self._discount_factor_scheduler_cfg = self.cfg.get("discount_factor_scheduler", None)
        if self._discount_factor_scheduler_cfg:
            self._discount_scheduler = DiscountFactorScheduler(initial_gamma=self._discount_factor, **self._discount_factor_scheduler_cfg)
        else:
            self._discount_scheduler = None

        self._max_next_value_func = max_next_value_func

    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        """Initialize the agent"""
        super().init(trainer_cfg=trainer_cfg)
        self.set_mode("eval")

        # create tensors in memory
        if self.memory is not None:
            self.memory.create_tensor(name="states", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="next_states", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
            self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="truncated", size=1, dtype=torch.bool)

            self._tensors_names = ["states", "actions", "rewards", "next_states", "terminated", "truncated"]

        # clip noise bounds
        if self.action_space is not None:
            self.clip_actions_min = torch.tensor(self.action_space.low, device=self.device)
            self.clip_actions_max = torch.tensor(self.action_space.high, device=self.device)

    def act(self, states: torch.Tensor, timestep: int, timesteps: int) -> torch.Tensor:
        """Process the environment's states to make a decision (actions) using the main policy

        :param states: Environment's states
        :type states: torch.Tensor
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int

        :return: Actions
        :rtype: torch.Tensor
        """

        # sample deterministic actions
        with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            actions, _, outputs = self.behavior_policy.act({"states": self._state_preprocessor(states)}, role="evalutation")

        return actions, None, outputs

    def record_transition(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: Any,
        timestep: int,
        timesteps: int,
    ) -> None:
        """Record an environment transition in memory

        :param states: Observations/states of the environment used to make the decision
        :type states: torch.Tensor
        :param actions: Actions taken by the agent
        :type actions: torch.Tensor
        :param rewards: Instant rewards achieved by the current actions
        :type rewards: torch.Tensor
        :param next_states: Next observations/states of the environment
        :type next_states: torch.Tensor
        :param terminated: Signals to indicate that episodes have terminated
        :type terminated: torch.Tensor
        :param truncated: Signals to indicate that episodes have been truncated
        :type truncated: torch.Tensor
        :param infos: Additional information about the environment
        :type infos: Any type supported by the environment
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        super().record_transition(
            states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps
        )

        if self.memory is not None:
            # reward shaping
            # if self._rewards_shaper is not None:
            #     rewards = self._rewards_shaper(rewards, timestep, timesteps)

            # storage transition in memory
            self.memory.add_samples(
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                terminated=terminated,
                truncated=truncated,
            )
            for memory in self.secondary_memories:
                memory.add_samples(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                )

    def pre_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called before the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        pass

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called after the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        if timestep >= self._learning_starts:
            self.set_mode("train")
            self._update(timestep, timesteps)
            self.set_mode("eval")

        # write tracking data and checkpoints
        super().post_interaction(timestep, timesteps)

    def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step with discount factor scheduling"""

        # Update discount factor if scheduler is enabled
        if self._discount_scheduler:
            self._discount_factor = self._discount_scheduler.step(timestep)

        # gradient steps
        for gradient_step in range(self._gradient_steps):

            # sample a batch from memory
            (
                sampled_states,
                sampled_actions,
                sampled_rewards,
                sampled_next_states,
                sampled_terminated,
                sampled_truncated,
            ) = self.memory.sample(names=self._tensors_names, batch_size=self._batch_size)[0]

            with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):

                sampled_states = self._state_preprocessor(sampled_states, train=True)
                sampled_next_states = self._state_preprocessor(sampled_next_states, train=True)

                # compute target values
                with torch.no_grad():
                    # next_actions, _, _ = self.behavior_policy.act({"states": sampled_next_states}, role="target_policy")

                    next_state_q_values, _, _ = self.target_critic.act(
                        {"states": sampled_next_states}, role="target_critic"
                    )

                    
                    if self._max_next_value_func is None:
                        # Here, we make an assumption that max_u V(x + f(x,u)*dt) is approximated by
                        # V(x') where x' is the next state after taking action u from state x.
                        # This assumes that the control policy u is optimal or near-optimal.
                        # In practice, this may not hold, and can cause abnormality in the value function.
                        max_next_q_val = next_state_q_values
                    else:
                        max_next_q_val = self._max_next_value_func(
                            sampled_states,
                            sampled_next_states,
                            next_state_q_values,
                            self.target_critic)
                        
                    target_values = (
                        (1 - self._discount_factor) * sampled_rewards\
                        + self._discount_factor 
                            * torch.min(sampled_rewards, 
                                        max_next_q_val * (sampled_terminated | sampled_truncated).logical_not())
                    )

                # compute critic loss
                critic_values, _, _ = self.critic.act(
                    {"states": sampled_states}, role="critic"
                )

                critic_loss = F.mse_loss(critic_values, target_values)

            # optimization step (critic)
            self.critic_optimizer.zero_grad()
            self.scaler.scale(critic_loss).backward()

            if config.torch.is_distributed:
                self.critic.reduce_parameters()

            if self._grad_norm_clip > 0:
                self.scaler.unscale_(self.critic_optimizer)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self._grad_norm_clip)

            self.scaler.step(self.critic_optimizer)

            self.scaler.update()  # called once, after optimizers have been stepped

            # update target networks
            self.target_critic.update_parameters(self.critic, polyak=self._polyak)

            # update learning rate
            if self._learning_rate_scheduler:
                self.critic_scheduler.step()

            # record data
            self.track_data("Loss / Critic loss", critic_loss.item())

            self.track_data("Q-network / Q1 (max)", torch.max(critic_values).item())
            self.track_data("Q-network / Q1 (min)", torch.min(critic_values).item())
            self.track_data("Q-network / Q1 (mean)", torch.mean(critic_values).item())

            self.track_data("Target / Target (max)", torch.max(target_values).item())
            self.track_data("Target / Target (min)", torch.min(target_values).item())
            self.track_data("Target / Target (mean)", torch.mean(target_values).item())

            if self._learning_rate_scheduler:
                self.track_data("Learning / Policy learning rate", self.policy_scheduler.get_last_lr()[0])
                self.track_data("Learning / Critic learning rate", self.critic_scheduler.get_last_lr()[0])

            # Track discount factor
            if self._discount_scheduler:
                self.track_data("Learning / Discount factor", self._discount_factor)