import numpy as np
from functools import wraps


def make_epsilon_greedy(agent, exploration_max=1.0, exploration_min=0.1, exploration_decay=0.995):
    agent.exploration_rate = exploration_max
    agent.exploration_min = exploration_min
    agent.exploration_decay = exploration_decay

    @wraps(agent.get_action)
    def get_action(self, state, available_actions=None):
        if np.random.rand() < self.exploration_rate:
            if available_actions is None:
                return self.available_actions.sample()
            return np.random.choice(list(available_actions))
        return agent.get_action(state, available_actions=available_actions)

    agent.get_action = get_action

    @wraps(agent.episode_end)
    def episode_end(self):
        agent.episode_end()
        self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(self.exploration_min, self.exploration_rate)

    return agent
