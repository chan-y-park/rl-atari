import random
import os

import numpy as np

import gym
import tensorflow as tf

#from collections import deque

from PIL import Image

# TODO
# * reimplement replay memory using uint8 and avoid saving 4 frames.
# * what to do with the stacking of states at the beginning of an episode.
# * rescaling rewards.
# * check if render() works fine with python/ipython cli.

#REPLAY_MEMORY_CAPACITY = 10 ** 6
#NUM_OF_FRAMES_TO_STACK = 4
#MINIBATCH_SIZE = 32
#GAMMA = 0.95 # TODO: check original code
#NUM_OF_STEPS_TO_MIN_EPSILON = 10 ** 6
#MIN_EPSILON = 0.1
#
#RMS_PROP_LEARNING_RATE = 0.0002 # TODO: check original code
#RMS_PROP_DECAY = 0.99 # TODO: check original code
#RMS_PROP_EPSILON = 1e-6 # TODO: check original code

DQN_Nature_configuration = {
    'Q_network': [
        ('input', {}),
        ('conv1', {'W_size': 8, 'stride': 4, 'in': 4, 'out': 16}),
        ('conv2', {'W_size': 4, 'stride': 2, 'in': 16, 'out': 32}),
        ('fc1', {'num_relus': 256}),
        ('output', {}),
    ],
    'Q_network_input_size': 84,
    'var_init_mean': 0.0,   # XXX
    'var_init_stddev': 0.01,    # XXX
    'minibatch_size': 32,
    'replay_memory_size': 10 ** 6,
#    'replay_memory_size': 10 ** 4,
    'agent_history_length': 4,
    'target_network_update_frequency': 10 ** 4,
#    'discount_factor': 0.99,
    'discount_factor': 0.95,
    'action_repeat': 4,
    'update_frequenct': 4,
#    'learning_rate': 0.00025,
    'learning_rate': 0.0002,
    'rms_prop_decay': 0.99,
#    'gradient_momentum': 0.95,
    'gradient_momentum': 0.0,
    'squared_gradient_momentum': 0.95,
#    'min_squared_gradient': 0.01,
    'min_squared_gradient': 1e-6,
    'initial_exploration': 1,
    'final_exploration': 0.1,
#    'final_exploration_frame': 10 ** 6,
    'final_exploration_frame': 10 ** 3,
#    'replay_start_size': 5 * (10 ** 4),
    'replay_start_size': 10 ** 2,
    'no-op_max': 30
}

class ReplayMemory:
    def __init__(self, config):
        self.size = config['replay_memory_size']
        self.minibatch_size = m = config['minibatch_size']
        self.agent_history_length = h = config['agent_history_length']
        self.index = 0
        self.full = False
        s = config['Q_network_input_size']
        self.states = np.zeros((self.size, s, s), dtype=np.uint8)
        self.actions = np.zeros((self.size), dtype=np.uint8)
        self.rewards = np.zeros((self.size), dtype=np.float32)
        self.terminals = np.zeros((self.size), dtype=np.uint8)
        self._minibatch = {
            'states': np.zeros((m, s, s, h), dtype=np.float32),
            'actions' : np.zeros(m, dtype=np.uint8),
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

    def get_stacked_states(self):
        h = self.agent_history_length
        i_start = self.index - h + 1 
        if i_start < 0:
            if not self.full:
                raise RuntimeError(
                    'Need at least 4 states in the replay memory.'
                )
            else:
                i_start += self.size
        indices = [
            i  % self.size
            for i in range(i_start, i_start + h)
        ]
        stacked_states = self.states[indices].transpose(1, 2, 0)
        return stacked_states[np.newaxis,:,:,:]

    def get_size(self):
        if self.full:
            current_size = self.size
        else:
            current_size = self.index + 1
        return current_size

    def sample_minibatch(self):
        current_size = self.get_size()
        if current_size < self.minibatch_size:
            raise RuntimeError(
                'Number of states {} less than minibatch size {}'
                .format(current_size, self.minibatch_size)
            )
        samples = np.random.randint(
            low=(self.agent_history_length - 1),
            # XXX The following cannot sample self.states[self.size]
            # even when the next state is available as self.states[0].
            high=(current_size - 1),
            size=self.minibatch_size,
        )
####
        self._samples = samples
####

        h = self.agent_history_length       
        for i in range(self.minibatch_size):
            j = samples[i]
            self._minibatch['states'][i] = np.transpose(
                self.states[(j - h + 1):(j + 1),:,:],
                (1, 2, 0),
            )
            self._minibatch['next_states'][i] = np.transpose(
                self.states[(j - h + 2):(j + 2),:,:],
                (1, 2, 0),
            )
        self._minibatch['actions'] = self.actions[samples]
        self._minibatch['rewards'] = self.rewards[samples]
        self._minibatch['terminals'] = self.terminals[samples]

        return self._minibatch

class AtariDQNAgent:
    def __init__(self, game_name='Breakout-v0'):
        # TODO: Check random seed usages.
        self.game_name = game_name
        log_dir = './log'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        checkpoints_dir = './checkpoints'
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)
        self._config = DQN_Nature_configuration
        self._random_seed = None
        self._env = gym.make(game_name)
        # XXX: Should we do skipping and stacking at the same time?
        # TODO: Remove the following hack
#        self._env.__init__(
#            game='breakout',
#            obs_type='image',
#            frameskip=1,
#            repeat_action_probability=1,
#        )
        self._replay_memory = ReplayMemory(config=self._config)

#        self.losses = deque(maxlen=1000)

        self._Q_network_layers = None
        self._train_op_input = {}
        self._loss_op = None
        self._tf_summary = {}
        self._tf_session = None

        self._graph = tf.Graph()
        with self._graph.as_default():
            self._build_graph()

    def _build_graph(self):
        self._Q_network_layers = []
        prev_layer = None
        for layer_name, layer_conf in self._config['Q_network']:
            with tf.variable_scope(layer_name):
                if 'input' in layer_name:
                    s = self._config['Q_network_input_size']
                    c = self._config['agent_history_length']

                    # The size of the minibatch can be either 32 when training
                    # or 1 when evaluating Q for a given state,
                    # therefore set to None.
                    new_layer = tf.placeholder(
                        tf.float32,
                        shape=(None, s, s, c),
                        name='input_layer',
                    )
                elif 'conv' in layer_name:
                    W_s = layer_conf['W_size']
                    s = layer_conf['stride']
                    i = layer_conf['in']
                    o = layer_conf['out']
                    W, b = self._get_filter_and_bias(
                        W_shape=(W_s, W_s, i, o),
                        b_shape=o,
                    )
                    a_tensor = tf.nn.conv2d(
                        prev_layer,
                        W,
                        strides=(1, s, s, 1),
                        # TODO: check original padding
                        padding='VALID',
                    )
                    a_tensor = tf.nn.bias_add(a_tensor, b)
                    new_layer = tf.nn.relu(a_tensor)
                elif 'fc' in layer_name:
                    n = layer_conf['num_relus']
                    _, h, w, c = prev_layer.shape.as_list()
                    in_dim = h * w * c
                    conv_out_flattened = tf.reshape(prev_layer, (-1, in_dim))
                    W, b = self._get_filter_and_bias(
                        W_shape=(in_dim, n),
                        b_shape=n,
                    )
                    new_layer = tf.nn.relu(
                        tf.nn.bias_add(
                            tf.matmul(conv_out_flattened, W), b
                        )
                    )
                elif 'output' in layer_name:
                    in_dim = prev_layer.shape.as_list()[-1]
                    num_actions = self.get_num_of_actions()
                    W, b = self._get_filter_and_bias(
                        W_shape=(in_dim, num_actions),
                        b_shape=(num_actions),
                    )
                    new_layer = tf.nn.bias_add(
                        tf.matmul(prev_layer, W), b
                    )


                self._Q_network_layers.append(new_layer)
                prev_layer = new_layer

        with tf.variable_scope('train_op'):

            ys = tf.placeholder(
                dtype=tf.float32,
                shape=(self._config['minibatch_size']),
                name='ys',
            )
            actions = tf.placeholder(
                #dtype=tf.float32,
                dtype=tf.uint8,
                shape=(self._config['minibatch_size']),
                name='actions',
            )
            Q_input = self._Q_network_layers[0]

            self._train_op_input = {
                'states': Q_input,
                'actions': actions,
                'ys': ys,
            }

            Q_output = self._Q_network_layers[-1]
            one_hot_actions = tf.one_hot(
                actions,
                self.get_num_of_actions(),
                #dtype=tf.float32,
            )
            Qs_of_action = tf.reduce_sum(
                tf.multiply(Q_output, one_hot_actions),
                axis=1,
            )
            self.Qs_of_action = Qs_of_action

            self._loss_op = tf.reduce_mean(tf.square(ys - Qs_of_action))
            # TODO Check original paremeters, especially momentum.
            self._train_op = tf.train.RMSPropOptimizer(
                learning_rate=self._config['learning_rate'],
                decay=self._config['rms_prop_decay'],
                momentum=self._config['gradient_momentum'],
                epsilon=self._config['min_squared_gradient'],
            ).minimize(self._loss_op)
            self._tf_summary['loss'] = tf.summary.scalar('loss', self._loss_op)

    def _get_filter_and_bias(self, W_shape, b_shape):
        W = tf.get_variable(
            'W',
            shape=W_shape,
            initializer=self._get_variable_initializer()
        )
        b = tf.get_variable(
            'b',
            shape=b_shape,
            initializer=self._get_variable_initializer()
        )
        return (W, b)

    def _get_variable_initializer(self):
        # TODO: Check the original initialization
        return tf.truncated_normal_initializer(
            mean=self._config['var_init_mean'],
            stddev=self._config['var_init_stddev'],
            seed=self._random_seed,
        )

    def get_num_of_actions(self):
        return self._env.action_space.n

    def preprocess_observation(
        self,
        observation,
        new_width=84,
        new_height=110,
        threshold=1,
    ):
        # TODO: remove resizing/grayscaling/thresholding,
        img = Image.fromarray(observation)
        img = img.convert('L')
        new_size = (new_width, new_height) 
        img = img.resize(new_size)
        obs = np.array(img, dtype=np.float32)

        # TODO: Use the following grayscaling without resizing.
#        obs = np.average(observation, weights=[0.299, 0.587, 0.114], axis=2)
        
        obs[obs < threshold] = 0
        obs[obs >= threshold] = 255
        obs = obs[-new_width:, :]
        return obs

#        if state is None:
#            # Prepare and return the initial state.
#            # TODO: check the original code.
#            return np.stack([obs] * 4, axis=2)
#        else:
#            return np.append(
#                obs.reshape(new_width, new_width, 1),
#                state[:, :, 1:],
#                axis=2,
#            )

    def train(self, max_num_of_steps, save_path=None):

        with self._graph.as_default():
            self._tf_session = tf.Session()

            self._tf_session.run(tf.global_variables_initializer())

            saver = tf.train.Saver()
            if save_path is not None:
                saver.restore(self._tf_session, save_path) 
                step = self._get_step_from_checkpoint(save_path)
            else:
                step = 0

            rewards = []    # reward per episode

            summary_writer = tf.summary.FileWriter('log')
            # TODO: check if add_graph is necessary.
            summary_writer.add_graph(self._graph, step)

            done = True
            while step < max_num_of_steps:
                if done:
                    initial_observation = self._env.reset()
                    state = self.preprocess_observation(initial_observation)
                    done = False
                    rewards.append(0)

                action = self._get_action(step)
                observation, reward, done, _ = self._env.step(action)
                rewards[-1] += reward
                step += 1
                state = self.preprocess_observation(observation)
                self._replay_memory.store(state, action, reward, int(done))

                # TODO: Wait until when?
                if step > self._config['replay_start_size']:
                    loss, loss_summary_str = self._optimize_Q()
                    summary_writer.add_summary(loss_summary_str, step)

                    if step % 1000 == 0:
                        print('step: {}, loss: {:g}, ave. reward: {:g}'
                              .format(step, loss, np.mean(rewards)))
#                        self.play_game()
#                        save_path = saver.save(
#                            self._tf_session,
#                            'checkpoints/{}'.format(self.game_name),
#                            global_step=step,
#                        )

            assert(step >= max_num_of_steps)
#            self.play_game()
#            save_path = saver.save(
#                tf_session,
#                'checkpoints/{}'.format(self.game_name),
#                global_step=step,
#            )
            summary_writer.close()
            return save_path


    def play_game(
        self,
        save_path=None,
        render=False,
    ):
        play_images = []
        if self._tf_session is not None:
            initial_observation = self._env.reset()
            state = self.preprocess_observation(initial_observation)
            done = False
            while not done:
                action = self._get_action_from_Q(state)
                observation, reward, done, _ = self._env.step(action)
                if render:
                    self._env.render()
                state = self.preprocess_observation(observation)
                play_images.append(Image.fromarray(observation))
        play_images[0].save(
            'play.gif',
            save_all=True,
            append_images=play_images[1:],
        )
    
    def _get_action(self, step):
        n_steps = self._config['final_exploration_frame'] 
        min_epsilon = self._config['final_exploration']
        if step < n_steps:
            epsilon = 1 - (1 - min_epsilon) / n_steps * step 
        else:
            epsilon = min_epsilon

        if (
            step < self._config['replay_start_size']
            or random.random() < epsilon
        ):
            return self._env.action_space.sample()
        else:
            return self._get_action_from_Q()

    def _get_action_from_Q(self):
        stacked_states = np.array(
            self._replay_memory.get_stacked_states(),
            dtype=np.float32,
        )
        Qs = self._get_Q_values(stacked_states)
        # TODO: Tiebreaking
        return np.argmax(Qs)

    def _get_Q_values(self, states):
        assert(states.ndim == 4)
        assert(states.dtype == np.float32)
        return self._tf_session.run(
            self._Q_network_layers[-1],
            feed_dict={self._Q_network_layers[0]: states},
        )

#    def _fill_minibatch_from_replay_memory(self, minibatch):
#        # TODO: Build minibatch from GPU-side
#        # In the replay memory, states/next_states have np.uint8 dtype.
#        experiences = random.sample(
#            self._replay_memory,
#            self._config['minibatch_size'],
#        )
#
#        for i, e in enumerate(experiences):
#            state, action, reward, next_state, terminal = e
#            minibatch['states'][i] = state
#            minibatch['actions'][i] = action
#            minibatch['rewards'][i] = reward
#            minibatch['next_states'][i] = next_state
#            minibatch['terminals'][i] = terminal 

    def _optimize_Q(self):
        gamma = self._config['discount_factor']
        minibatch = self._replay_memory.sample_minibatch()
        #states, actions, rewards, next_states, terminals = minibatch
        mask = (1 - minibatch['terminals'])
        Qs_next = self._get_Q_values(minibatch['next_states'])
        ys = minibatch['rewards'] + gamma * (
            np.amax(
                Qs_next * mask[:, np.newaxis],
                axis=1,
            )
        )

        _, loss, loss_summary_str, = self._tf_session.run(
            [self._train_op, self._loss_op, self._tf_summary['loss']],
            feed_dict={
               self._train_op_input['states']: minibatch['states'], 
               self._train_op_input['actions']: minibatch['actions'], 
               self._train_op_input['ys']: ys, 
            }
        )

        return (loss, loss_summary_str)

#    def _optimize_Q(self, gamma, tf_session):
#
#        m = self._config['minibatch_size']
#        minibatch = self._replay_memory.sample_minibatch(m)
#        states, actions, rewards, next_states, terminals = minibatch
#
#        Qs_next = np.amax(self._get_Q_values(next_states, tf_session), axis=1)
#
#        _, loss, loss_summary_str = tf_session.run(
#            [self._train_op, self._loss_op, self._tf_summary['loss']],
#            feed_dict={
#               self._train_op_input['states']: states, 
#               self._train_op_input['actions']: actions, 
#               self._train_op_input['rewards']: rewards, 
#               self._train_op_input['terminals']: terminals, 
#               self._train_op_input['Qs_next']: Qs_next, 
#            }
#        )

        return (loss, loss_summary_str)
    def _get_step_from_checkpoint(self, save_path):
        return int(save_path.split('-')[-1])
