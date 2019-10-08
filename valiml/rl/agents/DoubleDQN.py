import numpy as np

from valiml.rl.agents.DQN import DQN


class DoubleDQN(DQN):
    def __init__(self, training_model, inference_model, gamma=0.95, polyak_coef=0.01, shift_episodes=1):
        super(DoubleDQN, self).__init__(training_model, inference_model, gamma=gamma)
        self.target_model = self.clone()
        self.polyak_coef = polyak_coef
        self.shift_episodes = shift_episodes

    def train(self, inputs, q_values, weights=None):
        loss = super(DoubleDQN, self).train(inputs, q_values, weights=weights)

        if self.n_training_steps == self.shift_episodes:
            agent_weights = self.inference_model.get_weights()
            target_weights = self.target_model.get_weights()

            new_weights = []
            for agent_layer_weights, target_layer_weights in zip(agent_weights, target_weights):
                new_layer_weights = self.polyak_coef * agent_layer_weights + \
                                    (1 - self.polyak_coef) * target_layer_weights
                new_weights.append(new_layer_weights)

            self.target_model.set_weights(new_weights)

        return loss

    def compute_q_target(self, state, action, reward, next_state, terminal):
        target_action = np.argmax(self.target_model.predict(next_state)[0])
        return reward if terminal else reward + self.gamma * self.inference_model.predict(next_state)[0][target_action]
