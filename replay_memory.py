import numpy as np

class ReplayMemory:
    def __init__(self, config, np_random):
        self.size = config['replay_memory_size']
        self.minibatch_size = m = config['minibatch_size']
        self.agent_history_length = h = config['agent_history_length']
        self.Q_network_input_size = s = config['Q_network_input_size']

        self.index = 0
        self.full = False
        self.states = np.zeros((self.size, s, s), dtype=np.uint8)
        self.actions = np.zeros((self.size), dtype=np.uint8)
        self.rewards = np.zeros((self.size), dtype=np.float32)
        self.terminals = np.zeros((self.size), dtype=np.uint8)
        self._np_random = np_random
        self._minibatch = {
            'states': np.zeros((m, s, s, h), dtype=np.float32),
            'actions': np.zeros(m, dtype=np.uint8),
            'rewards': np.zeros(m, dtype=np.float32),
            'next_states': np.zeros((m, s, s, h), dtype=np.float32),
            'terminals': np.zeros(m, dtype=np.uint8),
        }

    def store(self, state, action, reward, terminal):
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.terminals[self.index] = terminal
        self.index += 1
        if self.index == self.size:
            self.full = True
            self.index = 0

    def get_size(self):
        if self.full:
            current_size = self.size
        else:
            current_size = self.index
        return current_size

    def get_samples(self, sample_size):
        current_size = self.get_size()
        if current_size < sample_size:
            raise RuntimeError(
                'Number of states {} less than minibatch size {}'
                .format(current_size, sample_size)
            )
        samples = self._np_random.randint(
            low=(self.agent_history_length - 1),
            # XXX The following cannot sample self.states[self.size]
            # even when the next state is available as self.states[0].
            high=(current_size - 1),
            size=sample_size,
        )
        return samples

    def sample_minibatch(self):
        samples = self.get_samples(self.minibatch_size)

        self.fill_states(samples, self._minibatch['states'])
        self.fill_states(samples, self._minibatch['next_states'], offset=1)
        self._minibatch['actions'] = self.actions[samples]
        self._minibatch['rewards'] = self.rewards[samples]
        self._minibatch['terminals'] = self.terminals[samples]

        return self._minibatch

    def fill_states(self, samples, buf_states, offset=0):
        h = self.agent_history_length
        for i in range(len(samples)):
            j = samples[i]
            buf_states[i] = np.transpose(
                self.states[(j - h + 1 + offset):(j + 1 + offset),:,:],
                (1, 2, 0),
            )
