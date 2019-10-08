from abc import abstractmethod


class Agent(object):
    @abstractmethod
    def get_action(self, state, available_actions=None):
        raise NotImplemented()


class LearningAgent(Agent):
    @abstractmethod
    def episode_end(self):
        raise NotImplemented()

    @abstractmethod
    def train(self, inputs, outputs, weights=None):
        raise NotImplemented()


class QLearningAgent(Agent):
    @abstractmethod
    def train(self, inputs, q_values, weights=None):
        raise NotImplemented()

    @abstractmethod
    def compute_q_target(self, state, action, reward, next_state, terminal):
        raise NotImplemented()
