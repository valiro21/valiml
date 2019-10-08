import numpy as np


class PrioritizedExperienceReplay(object):
    def __init__(self, beta=0.5):
        self.beta = beta

    def train(self, agent, prioritized_memory, batch_size=20):
        if len(prioritized_memory) < batch_size:
            return None, None

        rewards = []
        inputs = []
        outputs = []
        weights = []
        for idx in prioritized_memory.sample(sample_size=batch_size):
            state, action, reward, state_next, terminal = prioritized_memory[idx]
            q_update = agent.compute_q_target(state, action, reward, state_next, terminal)
            q_values = agent.get_q_values(state)
            q_diff = q_update - q_values[0][action]
            q_values[0][action] = q_update

            prioritized_memory.priority_update(idx, np.abs(q_diff))

            priority = prioritized_memory.get_priority(idx)
            w = np.power((1 / len(prioritized_memory)) * (1 / priority), self.beta)

            rewards.append(reward)

            inputs.append(state)
            outputs.append(q_values)
            weights.append(w)

        loss = agent.train(
            [np.concatenate(inputs, axis=0)],
            np.concatenate(outputs, axis=0),
            weights=np.array(weights) / np.sum(weights)
        )

        return loss, np.mean(rewards)
