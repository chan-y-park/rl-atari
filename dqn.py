import random
import numpy as np

import gym
import tensorflow as tf

from collections import deque

from PIL import Image

REPLAY_MEMORY_CAPACITY = 10 ** 6
NUM_OF_FRAMES_TO_STACK = 4
MINIBATCH_SIZE = 32
GAMMA = 0.95 # TODO: check original code
NUM_OF_STEPS_TO_MIN_EPSILON = 10 ** 6
MIN_EPSILON = 0.1

RMS_PROP_LEARNING_RATE = 0.0002 # TODO: check original code
RMS_PROP_DECAY = 0.99 # TODO: check original code
RMS_PROP_EPSILON = 1e-6 # TODO: check original code


class AtariDQNAgent:
    def __init__(self, game_name='Breakout-v0'):
        # TODO: Check random seed usages.
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
        self._replay_memory = deque(maxlen=REPLAY_MEMORY_CAPACITY)

        self._Q_graph_configuration = [
            ('input', {'size': 84, 'num_chs': 4}),
            ('conv1', {'W_size': 8, 'stride': 4, 'in': 4, 'out': 16}),
            ('conv2', {'W_size': 4, 'stride': 2, 'in': 16, 'out': 32}),
            ('fc1', {'num_relus': 256}),
            ('output', {'num_actions': self.get_num_of_actions()}),
        ]
        
        self._Q_graph = tf.Graph()
        self._Q_graph_layers = None
        self._train_op_input = {}
        with self._Q_graph.as_default():
            self._build_Q_graph()

    def get_num_of_actions(self):
        return self._env.action_space.n

    def preprocess_observation(
        self,
        observation,
        state=None,
        new_width=84,
        new_height=110,
        threshold=1,
    ):
        img = Image.fromarray(observation)
        # TODO: remove resizing/grayscaling/thresholding.
        img = img.convert('L')
        new_size = (new_width, new_height) 
        img = img.resize(new_size)
        
        obs = np.array(img)
        
        obs[obs < threshold] = 0
        obs[obs >= threshold] = 255
        obs = obs[-new_width:, :]

        if state is None:
            # Prepare and return the initial state.
            # TODO: check the original code.
            return np.stack([obs] * 4, axis=2)
        else:
            return np.append(
                obs.reshape(new_width, new_width, 1),
                state[:, :, 1:],
                axis=2,
            )

        return obs

    def train(self, num_of_frames, tf_session):
        step = 0
        while step < num_of_frames:
            initial_observation = self._env.reset()
            prev_state = self.preprocess_observation(
                initial_observation,
            )
            done = False
            while not done:
                action = self._get_action(prev_state, step, tf_session)
                observation, reward, done, _ = self.env.step(action)
                step += 1
                new_state = self.preprocess_observation(
                    observation,
                    last_state=prev_state,
                )
                self._replay_memory.append(
                    (prev_state, action, reward, new_state)
                )
                self._optimize_Q(MINIBATCH_SIZE, GAMMA, tf_session)
                prev_state = new_state


    def play_game(self):
        pass
    
    def _build_Q_graph(self):
        self._Q_graph_layers = []
        prev_layer = None
        for layer_name, layer_conf in self._Q_graph_configuration:
            with tf.variable_scope(layer_name):
                if 'input' in layer_name:
                    s = layer_conf['size']
                    c = layer_conf['num_chs']
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
                    num_actions = layer_conf['num_actions']
                    W, b = self._get_filter_and_bias(
                        W_shape=(in_dim, num_actions),
                        b_shape=(num_actions),
                    )
                    new_layer = tf.nn.bias_add(
                        tf.matmul(prev_layer, W), b
                    )


                self._Q_graph_layers.append(new_layer)
                prev_layer = new_layer

        with tf.variable_scope('train_op'):
            action = tf.placeholder(
                tf.float32,
                shape=(MINIBATCH_SIZE, self.get_num_of_actions()),
                name='action',
            )
            y = tf.placeholder(
                tf.float32,
                shape=(MINIBATCH_SIZE),
                name='y'
            )
            Q = self._Q_graph_layers[-1]

            self._train_op_input = {
                'state': self._Q_graph_layers[0],
                'action': action,
                'y': y,
            }

            Q_of_action = tf.reduce_sum(tf.multiply(Q, action), axis=1)
            # Use mean rather than sum to track the value of loss.
            self._loss = tf.reduce_mean(tf.square(y - Q_of_action))
            self._train_op = tf.train.RMSPropOptimizer(
                learning_rate=RMS_PROP_LEARNING_RATE,
#                decay=RMS_PROP_DECAY,
#                epsilon=RMS_PROP_EPSILON,
            ).minimize(self._loss)

    def _get_action(self, state, step, tf_session):
        n_steps = NUM_OF_STEPS_TO_MIN_EPSILON 
        if step < n_steps:
            epsilon = 1 - (1 - MIN_EPSILON) / n_steps * step 
        else:
            epsilon = MIN_EPSILON
        if random.random() < epsilon:
            return self.env.action_space.sample()
        else:
            return self._get_action_from_Q(state, tf_session)

    def _get_action_from_Q(self, state, tf_session):
        Qs = self._get_Q_values(self, state, tf_session)
        return np.argmax(Qs)

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
        return tf.random_normal_initializer(
            seed=self._random_seed,
        )

    def _get_minibatch_from_replay_memory(self, minibatch_size):
        # TODO: Build minibatch from GPU-side
        experiences = random.sample(self._replay_memory, minibatch_size)
        minibatch = {
            'states': [],
            'actions' : [],
            'rewards': [],
            'next_states': [],
        }
        for state, action, reward, next_state in experiences:
            minibatch['states'].append(state)
            minibatch['actions'].append(action)
            minibatch['rewards'].append(reward)
            minibatch['next_states'].append(next_state)

        return minibatch


    def _get_Q_values(self, state, tf_session):
        return tf_session.run(
            self._Q_graph_layers[-1],
            feed_dict={self._Q_graph_layers[0]: state},
        )

    def _optimize_Q(self, minibatch_size, gamma, tf_session):
        minibatch = self.get_minibatch_from_replay_memory(minibatch_size)
        next_states = minibatch['next_states']

        ys = []
        next_Qs = self._get_Q_values(next_states, tf_session)

        for i in range(minibatch_size):
            reward = minibatch['rewards'][i]
            next_state = next_states[i]
            if next_state is None:
                ys.append(reward)
            else:
                ys.append(reward, gamma * np.max(next_Qs[i]))

        tf_session.run(
            self._train_op,
            feed_dict={
               self._train_op_input['state']: minibatch['states'], 
               self._train_op_input['action']: minibatch['actions'], 
               self._train_op_input['y']: ys, 
            }
        )
            

