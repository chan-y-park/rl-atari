import random
import os

import numpy as np

import gym
import tensorflow as tf

from collections import deque

from PIL import Image

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
#    'replay_memory_size': 10 ** 6,
    'replay_memory_size': 10 ** 4,
    'agent_history_length': 4,
    'target_network_update_frequency': 10 ** 4,
    'discount_factor': 0.99,
    'action_repeat': 4,
    'update_frequenct': 4,
    'learning_rate': 0.00025,
    'gradient_momentum': 0.95,
    'squared_gradient_momentum': 0.95,
    'min_squared_gradient': 0.01,
    'initial_exploration': 1,
    'final_exploration': 0.1,
#    'final_exploration_frame': 10 ** 6,
    'final_exploration_frame': 10 ** 4,
#    'replay_start_size': 5 * (10 ** 4),
    'replay_start_size': 5 * (10 ** 2),
    'no-op_max': 30
}

class ReplayMemory:
    def __init__(self, config):
        s = config['Q_network_input_size']
        c = config['agent_history_length']

        self.size = config['replay_memory_size']
        self.index = 0
        self.full = False
        self.states = np.zeros((self.size, s, s, c), dtype=np.float32)
        self.actions = np.zeros((self.size), dtype=np.uint8)
        self.rewards = np.zeros((self.size), dtype=np.float32)
        self.next_states = np.zeros((self.size, s, s, c), dtype=np.float32)
        self.done = np.zeros((self.size), dtype=np.uint8)

    def store(self, state, action, reward, next_state, done):
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_states[self.index] = next_state
        self.done[self.index] = done
        self.index += 1
        if self.index == self.size:
            self.full = True
            self.index = 0
        #self.index %= self.size

    def sample_minibatch(self, minibatch_size):
        if self.full:
            current_size = self.size
        else:
            current_size = self.index + 1
        samples = np.random.choice(current_size, minibatch_size)
        return(
            self.states[samples, :, :, :],
            self.actions[samples],
            self.rewards[samples],
            self.next_states[samples, :, :, :],
            self.done[samples],
        )

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
#        self._replay_memory = deque(
#            maxlen=self._config['replay_memory_size']
#        )
        self._replay_memory = ReplayMemory(config=self._config)

        self.losses = deque(maxlen=1000)

        self._graph = tf.Graph()
        self._Q_network_layers = None
        self._train_op_input = {}
        self._loss_op = None
        self.tf_summary = {}
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
            action = tf.placeholder(
                dtype=tf.uint8,
                shape=(self._config['minibatch_size']),
                name='action',
            )
            y = tf.placeholder(
                tf.float32,
                shape=(self._config['minibatch_size']),
                name='y'
            )
            Q_input = self._Q_network_layers[0]
            self._train_op_input = {
                'state': Q_input,
                'action': action,
                'y': y,
            }

            one_hot_action = tf.one_hot(
                action,
                self.get_num_of_actions(),
                dtype=tf.float32,
            )
            Q_output = self._Q_network_layers[-1]
            Q_of_action = tf.reduce_sum(
                tf.multiply(Q_output, one_hot_action),
                axis=1,
            )
            self.Q_of_action = Q_of_action
            # Use mean rather than sum to track the value of loss.
            self._loss_op = tf.reduce_mean(tf.square(y - Q_of_action))
            self._train_op = tf.train.RMSPropOptimizer(
                learning_rate=self._config['learning_rate'],
                decay=0.9,
#                momentum = self._config['gradient_momentum'],
                epsilon=self._config['min_squared_gradient'],
#                learning_rate=0.0002,
#                decay=0.99,
                momentum=0.0,
#                epsilon=1e-6,
            ).minimize(self._loss_op)
            self.tf_summary['loss'] = tf.summary.scalar('loss', self._loss_op)

    def _get_filter_and_bias(self, W_shape, b_shape):
        W = tf.get_variable(
            'W',
            shape=W_shape,
            #initializer=self._get_variable_initializer()
            initializer=tf.constant_initializer(0.1)
        )
        b = tf.get_variable(
            'b',
            shape=b_shape,
            #initializer=self._get_variable_initializer()
            initializer=tf.constant_initializer(0.1)
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
        state=None,
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

    def train(self, max_num_of_steps, save_path=None):
#        m = self._config['minibatch_size']
#        s = self._config['Q_network_input_size']
#        c = self._config['agent_history_length']
#        minibatch = {
#            'states': np.zeros((m, s, s, c), dtype=np.float32),
#            'actions' : np.zeros(m, dtype=np.uint8),
#            'rewards': np.zeros(m, dtype=np.float32),
#            'next_states': np.zeros((m, s, s, c), dtype=np.float32),
#            'done': [False] * m,
#        }

        with self._graph.as_default():
            tf_session = tf.Session()

            tf_session.run(tf.global_variables_initializer())
            
            saver = tf.train.Saver()
            if save_path is not None:
                saver.restore(tf_session, save_path) 
                step = self._get_step_from_checkpoint(save_path)
            else:
                step = 0

            #rewards = np.zeros(max_num_of_steps - step, dtype=np.float32)
            rewards = []    # reward per episode

            summary_writer = tf.summary.FileWriter('log')
            # TODO: check if add_graph is necessary.
            summary_writer.add_graph(self._graph, step)

            done = True
            while step < max_num_of_steps:
                if done:
                    initial_observation = self._env.reset()
                    prev_state = self.preprocess_observation(
                        initial_observation
                    )
                    done = False
                    rewards.append(0)

                action = self._get_action(prev_state, step, tf_session)
                observation, reward, done, _ = self._env.step(action)
#                rewards[step] = reward
                rewards[-1] += reward
#                    self._env.render()
                step += 1
                new_state = self.preprocess_observation(
                    observation,
                    state=prev_state,
                )
                self._replay_memory.store(
                    prev_state, action, reward, new_state, int(done)
                )
                prev_state = new_state

                # TODO: Wait until when?
                if step > self._config['replay_start_size']:
                    loss, loss_summary_str = self._optimize_Q(
                        #minibatch,
                        self._config['discount_factor'],
                        tf_session,
                    )
                    self.losses.append(loss)
                    summary_writer.add_summary(
                        loss_summary_str,
                        step,
                    )


                    if step % 100 == 0:
                        print(
                            'step: {}, loss: {:g}, ave. reward: {:g}'
                            .format(step, loss, np.mean(rewards)),
                        )
                        #self.play_game(tf_session)
                        save_path = saver.save(
                            tf_session,
                            'checkpoints/{}'.format(self.game_name),
                            global_step=step,
                        )

            assert(step >= max_num_of_steps)
            self.play_game(tf_session)
#            save_path = saver.save(
#                tf_session,
#                'checkpoints/{}'.format(self.game_name),
#                global_step=step,
#            )
            summary_writer.close()
            return save_path


    def play_game(
        self,
        tf_session=None,
        save_path=None,
        render=False,
    ):
        play_images = []
        if tf_session is not None:
            initial_observation = self._env.reset()
            prev_state = self.preprocess_observation(initial_observation)
            done = False
            while not done:
                action = self._get_action_from_Q(prev_state, tf_session)
                observation, reward, done, _ = self._env.step(action)
                if render:
                    self._env.render()
                new_state = self.preprocess_observation(
                    observation,
                    state=prev_state,
                )
                prev_state = new_state
                play_images.append(Image.fromarray(observation))
        play_images[0].save(
            'play.gif',
            save_all=True,
            append_images=play_images[1:],
        )
    
    def _get_action(self, state, step, tf_session):
        n_steps = self._config['final_exploration_frame'] 
        min_epsilon = self._config['final_exploration']
        if step < n_steps:
            epsilon = 1 - (1 - min_epsilon) / n_steps * step 
        else:
            epsilon = min_epsilon
        if random.random() < epsilon:
            return self._env.action_space.sample()
        else:
            return self._get_action_from_Q(state, tf_session)

    def _get_action_from_Q(self, state, tf_session):
        Qs = self._get_Q_values(state[np.newaxis,:,:,:], tf_session)
        return np.argmax(Qs)

    def _get_Q_values(self, states, tf_session):
        assert(states.ndim == 4)
        assert(states.dtype == np.float32)
        return tf_session.run(
            self._Q_network_layers[-1],
            feed_dict={self._Q_network_layers[0]: states},
        )

    def _fill_minibatch_from_replay_memory(self, minibatch):
        # TODO: Build minibatch from GPU-side
        # In the replay memory, states/next_states have np.uint8 dtype.
        experiences = random.sample(
            self._replay_memory,
            self._config['minibatch_size'],
        )

        for i, e in enumerate(experiences):
            state, action, reward, next_state, done = e
            minibatch['states'][i] = state
            minibatch['actions'][i] = action
            minibatch['rewards'][i] = reward
            minibatch['next_states'][i] = next_state
            minibatch['done'][i] = done

#    def _optimize_Q(self, minibatch, gamma, tf_session):
    def _optimize_Q(self, gamma, tf_session):
        #self._fill_minibatch_from_replay_memory(minibatch)
        #next_states = minibatch['next_states']

        m = self._config['minibatch_size']
        minibatch = self._replay_memory.sample_minibatch(m)
        states, actions, rewards, next_states, done = minibatch

        #Qs = np.amax(self._get_Q_values(states, tf_session), axis=1)
        Qs = self._get_Q_values(states, tf_session)

        mask = (1 - done)
        next_Qs = (
            self._get_Q_values(next_states, tf_session) * mask[:, np.newaxis]
        )
        ys = rewards + (gamma * np.amax(next_Qs, axis=1))

        _, loss, loss_summary_str, Q_of_action = tf_session.run(
            [self._train_op, self._loss_op, self.tf_summary['loss'],
             self.Q_of_action],
            feed_dict={
               self._train_op_input['state']: states, 
               self._train_op_input['action']: actions, 
               self._train_op_input['y']: ys, 
            }
        )

        return (loss, loss_summary_str)

    def _get_step_from_checkpoint(self, save_path):
        return int(save_path.split('-')[-1])
