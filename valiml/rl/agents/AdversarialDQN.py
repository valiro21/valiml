import numpy as np
from valiml.rl.agents.DoubleDQN import DoubleDQN


class AdversarialDQN(DoubleDQN):
    def __init__(self, training_model, inference_model, **kwargs):
        super(AdversarialDQN, self).__init__(training_model, inference_model, **kwargs)
        self.other_player = None

    def set_other_player(self, agent):
        self.other_player = agent

    def compute_q_target(self, state, action, reward, next_state, terminal):
        target_action = np.argmax(self.other_player.target_model.predict(next_state)[0])
        opponent_reward =  self.other_player.inference_model.predict(next_state)[0][target_action]
        return reward if terminal else reward - self.gamma * opponent_reward
