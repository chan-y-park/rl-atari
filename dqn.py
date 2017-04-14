import random
import numpy as np

import gym
import tensorflow as tf

from collections import deque

from PIL import image

REPLAY_MEMORY_CAPACITY = 10 ** 6

class AtariDQNAgent:
    def __init__(self, game_name='Breakout-v0'):
        # TODO: Check random seed usages.
        self._random_seed = None
        self._env = gym.make(game_name)
        # TODO: Remove the following hack
        self._env.__init__(
            game='breakout',
            obs_type='image',
            frameskip=1,
            repeat_action_probability=0,
        )
        self._replay_memory = deque(maxlen=REPLAY_MEMORY_CAPACITY)

        self._Q_graph_confiuration = [
            ('input', {'size': 84, 'num_chs': 4),
            ('conv1', {'W_size': 8, 'stride': 4, 'in': 4, 'out': 16}),
            ('conv2', {'W_size': 4, 'stride': 2, 'in': 16, 'out': 32}),
            ('fc1', {'num_relus': 256}),
            ('output', {'num_actions': self.get_num_of_actions}),
        ]
        
        self._Q_graph = tf.Graph()
        self._Q_graph_layers = None
        self._train_op_input = {}
        with self._Q_graph.as_default():
            self._build_Q_graph()

    def get_num_of_actions(self):
        return len(self._env.action_space)

    def preprocess_observation(
        self,
        observation,
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
        return obs

    def train(self, num_of_episodes, num_of_steps):
        for i in range(num_of_episodes):
            initial_observation = self.env.reset()
            phi_0 = self.preprocess_observation(initial_observation)
            done = False
            while not done:
                action = self._get_action(phi_0)
                observation, reward, done, _ = self.env.step(action)
                phi_1 = self.preprocess_observation(observation)
                self._replay_memory.append(
                    (phi_0, a_t, reward, phi_1)
                )
                phi_0 = phi_1
                self._optimize_Q()


    def play_game(self):

    def _build_Q_graph(self):
        prev_layer = None
        for layer_name, layer_conf in self._Q_graph_configuration:
            with tf.variable_scope(layer_name):
                if 'input' in layer_name:
                    s = layer_conf['size']
                    c = layer_conf['num_chs']
                    new_layer = tf.placeholder(
                        tf.float32,
                        shape=(-1, s, s, c),
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
                    conv_out_flattened = tf.reshape(prev_layer, (1, -1))
                    in_dim = a_tensor.shape.as_list()[-1]
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
                shape=(-1, self.get_num_of_actions()),
                name='action',
            )
            y = tf.placeholder(
                tf.float32,
                shape=(-1),
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
                decay=RMS_PROP_DECAY,
                epsilon=RMS_PROP_EPSILON,
            ).minimize(self._loss)

    def _get_action(self):
        self.env.action_space.sample()

    def _get_action_from_Q(self):

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

    def _get_minibatch_from_replay_memory(self, batch_size):
        # TODO: Build minibatch from GPU-side
        return random.sample(self._replay_memory, batch_size)

    def _get_Q_values(self, states, tf_session):
        return tf_session.run(
            self._Q_graph_layers[-1],
            feed_dict={self._Q_graph_layers[0]: states},
        )

    def _optimize_Q(self, batch_size, gamma, tf_session):
        minibatch = self.get_minibatch_from_replay_memory(batch_size)
        next_states = minibatch['next_states']

        ys = []
        next_Qs = self._get_Q_values(next_states, tf_session)

        for i in range(batch_size):
            state = states[i]
            reward = rewards[i]
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
            

