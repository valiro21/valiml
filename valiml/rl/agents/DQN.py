import numpy as np

from valiml.rl.agents.base import QLearningAgent

try:
    from keras.models import clone_model
except ImportError as e:
    from copy import deepcopy as clone_model


def restricted_argmax(array, limited_indexes):
    limited_indexes = list(limited_indexes)
    best_idx = limited_indexes[0]
    if len(limited_indexes) > 1:
        for idx in limited_indexes[1:]:
            if array[0][idx] > array[0][best_idx]:
                best_idx = idx

    return best_idx


class DQN(QLearningAgent):
    def __init__(self, training_model, inference_model, gamma=0.95):
        self.training_model = training_model
        self.inference_model = inference_model
        self.gamma = gamma
        self.n_training_steps = 0

    def get_action(self, state, available_actions=None):
        prediction = self.get_q_values(state)

        if available_actions is None:
            return np.argmax(prediction)
        return restricted_argmax(prediction, available_actions)

    def train(self, inputs, q_values, weights=None):
        if weights is None:
            weights = np.ones((inputs[0].shape[0], 1))

        history = self.training_model.fit([*inputs, weights], q_values, verbose=0)

        self.n_training_steps += 1

        return history.history['loss'][-1]

    def compute_q_target(self, state, action, reward, next_state, terminal):
        return reward if terminal else reward + self.gamma * np.amax(self.inference_model.predict(next_state)[0])

    def get_q_values(self, states):
        return self.inference_model.predict(states)

    def clone(self):
        return clone_model(self.inference_model)
