import numpy as np
from functools import wraps


def make_epsilon_greedy(agent, all_actions, exploration_max=1.0, exploration_min=0.1, exploration_decay=0.995):
    agent.exploration_rate = exploration_max
    agent.exploration_min = exploration_min
    agent.exploration_decay = exploration_decay

    agent_get_action = agent.get_action
    @wraps(agent.get_action)
    def get_action(state, available_actions=None):
        if np.random.rand() < agent.exploration_rate:
            if available_actions is None:
                return all_actions.sample()
            return np.random.choice(list(available_actions))
        return agent_get_action(state, available_actions=available_actions)

    agent.get_action = get_action

    agent_episode_end = agent.episode_end
    @wraps(agent.episode_end)
    def episode_end():
        agent_episode_end()
        agent.exploration_rate *= agent.exploration_decay
        agent.exploration_rate = max(agent.exploration_min, agent.exploration_rate)

    agent.episode_end = episode_end

    return agent
