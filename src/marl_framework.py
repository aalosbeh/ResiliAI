"""
Multi-Agent Reinforcement Learning (MARL) Framework for ResiliAI

This module defines the core components of the MARL environment, including agents
representing economic actors (households, industries, government) and the simulation
environment itself.

Author: Anas ALsobeh, Raneem Alkurdi
Date: July 2025
"""

import numpy as np
import pettingzoo
from pettingzoo.utils import agent_selector, wrappers
from gymnasium import spaces


class EconomicAgent(pettingzoo.utils.BaseParallelWrapper):
    """
    Represents a single economic actor in the simulation.
    """
    def __init__(self, agent_id, type, initial_capital, consumption_rate):
        self.id = agent_id
        self.type = type  # e.g., household, industry, government
        self.capital = initial_capital
        self.consumption_rate = consumption_rate
        self.policy = np.zeros(1)  # Placeholder for agent-specific policy

    def step(self, action):
        """
        Update agent state based on action.
        """
        # Simple economic model: capital decreases by consumption, increases by production (action)
        production = action[0]
        self.capital += production - self.consumption_rate * self.capital
        return self.capital, self.get_observation()

    def get_observation(self):
        """
        Return the agent's observation of the environment.
        """
        return np.array([self.capital, self.consumption_rate])


def env(render_mode=None):
    """
    The env function returns the raw environment without wrappers
    since the standard PettingZoo wrappers are not compatible with ParallelEnv.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)
    # No wrappers for ParallelEnv
    return env


class raw_env(pettingzoo.ParallelEnv):
    """
    The AEC API is simpler to implement and allows for more complex scenarios.
    For information on how to convert between AEC and Parallel, see
    https://pettingzoo.farama.org/api/parallel/#pettingzoo.utils.conversions.from_parallel
    """

    metadata = {
        "render_modes": ["human"],
        "name": "ResiliAI_v1",
    }

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.agents_config = {
            "household_0": {"type": "household", "initial_capital": 100, "consumption_rate": 0.1},
            "industry_0": {"type": "industry", "initial_capital": 1000, "consumption_rate": 0.05},
            "government_0": {"type": "government", "initial_capital": 10000, "consumption_rate": 0.01},
        }
        self.possible_agents = list(self.agents_config.keys())
        self.agents = {}

        # gymnasium spaces are defined here
        self.action_spaces = {agent: spaces.Box(low=0, high=10, shape=(1,)) for agent in self.possible_agents}
        self.observation_spaces = {agent: spaces.Box(low=0, high=np.inf, shape=(2,)) for agent in self.possible_agents}

    def reset(self, seed=None, options=None):
        self.agents = {}
        self.rewards = {agent: 0 for agent in self.possible_agents}
        self._cumulative_rewards = {agent: 0 for agent in self.possible_agents}
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self.infos = {agent: {} for agent in self.possible_agents}

        for agent_id, config in self.agents_config.items():
            self.agents[agent_id] = EconomicAgent(agent_id, **config)

        observations = {agent_id: agent.get_observation() for agent_id, agent in self.agents.items()}
        return observations, {agent: {} for agent in self.possible_agents}

    def step(self, actions):
        # economic shock (e.g., reduce capital of all agents)
        if np.random.rand() < 0.1:  # 10% chance of a shock
            for agent in self.agents.values():
                agent.capital *= (1 - np.random.uniform(0.1, 0.5)) # reduce capital by 10-50%

        observations = {}
        rewards = {}
        for agent_id, action in actions.items():
            capital, obs = self.agents[agent_id].step(action)
            observations[agent_id] = obs
            rewards[agent_id] = capital # reward is the agent's capital

        terminations = {agent: False for agent in self.possible_agents}
        truncations = {agent: False for agent in self.possible_agents}
        infos = {agent: {} for agent in self.possible_agents}

        return observations, rewards, terminations, truncations, infos

    def render(self):
        if self.render_mode is None:
            return

        for agent_id, agent in self.agents.items():
            print(f"{agent.type} {agent.id}: Capital = {agent.capital:.2f}")

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

