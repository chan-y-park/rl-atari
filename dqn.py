import os
import time
import subprocess
import json

import numpy as np

import gym
import tensorflow as tf

from PIL import Image

from configs import dqn_nature_configuration

ACTION_NO_OP = 0

NVIDIA_SMI_ARGS = [
    'nvidia-smi',
    '--query-compute-apps=pid,used_memory',
#    '--format=csv,noheader,nounits',
    '--format=csv,noheader',
]

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


class AtariDQNAgent:
    def __init__(
        self,
        game_name='Breakout-v0',
        log_dir='logs',
        checkpoint_dir='checkpoints',
        run_dir='runs',
        random_seed=None,
        config=dqn_nature_configuration,
        gpu_memory_fraction=None,
        gpu_memory_allow_growth=True,
    ):
        self.game_name = game_name

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)

        self._pid = os.getpid()
        self._config = config
        self._env = gym.make(game_name)

        self._random_seed = random_seed
        # Seeding Numpy.
        self._np_random = np.random.RandomState(self._random_seed)
        if self._random_seed is not None:
            # Seeding Gym Atari environment.
            self._env.env._seed(self._random_seed)
            # Seeding Tensorflow.
            tf.set_random_seed(self._random_seed)

        self._replay_memory = ReplayMemory(
            config=self._config,
            np_random=self._np_random,
        )

        self._target_Q_update_ops = None

        self._validation_states = None
        self._tf_summary = {}
        self._tf_session = None
       
        self._tf_graph = tf.Graph()
        with self._tf_graph.as_default():
            with tf.variable_scope('Q_network'):
                self._build_Q_network()
            tnuf = self._config['target_network_update_frequency']
            if tnuf  is not None:
                print(
                    'Use a target Q network with update frequency {}.'
                    .format(tnuf)
                )
                with tf.variable_scope('target_Q_network'):
                    self._build_Q_network()
                with tf.variable_scope('update'):
                    self._build_update_ops()

            with tf.variable_scope('train'):
                self._build_train_ops()

            with tf.variable_scope('validation'):
                self._build_validation_ops()

            with tf.variable_scope('stat'):
                tf.Variable(0.0, name='used_gpu_memory')

            with tf.variable_scope('summary'):
                self._build_summary_ops()
            
            tf_config = tf.ConfigProto()
            tf_config.gpu_options.allow_growth = gpu_memory_allow_growth 
            if gpu_memory_fraction is not None:
                tf_config.gpu_options.per_process_gpu_memory_fraction = (
                    gpu_memory_fraction
                )
            self._tf_session = tf.Session(config=tf_config)
            self._tf_session.run(tf.global_variables_initializer())

    def _build_Q_network(self):
        s = self._config['Q_network_input_size']
        c = self._config['agent_history_length']

        # The size of the minibatch can be either 32 when training
        # or 1 when evaluating Q for a given state,
        # therefore set to None.
        prev_layer = tf.placeholder(
            tf.float32,
            shape=(None, s, s, c),
            name='input',
        )
        for layer_name, layer_conf in self._config['Q_network']:
            with tf.variable_scope(layer_name):
                if 'conv' in layer_name:
                    W_s = layer_conf['filter_size']
                    s = layer_conf['stride']
                    _, _, _, i = prev_layer.shape.as_list()
                    o = layer_conf['num_filters']
                    W, b = self._get_filter_and_bias(
                        W_shape=(W_s, W_s, i, o),
                        b_shape=o,
                    )
                    a_tensor = tf.nn.conv2d(
                        prev_layer,
                        W,
                        strides=(1, s, s, 1),
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
                        tf.matmul(prev_layer, W), b,
                        name='Qs',
                    )

                prev_layer = new_layer

    def _build_train_ops(self):
        ys = tf.placeholder(
            dtype=tf.float32,
            shape=(self._config['minibatch_size']),
            name='ys',
        )
        actions = tf.placeholder(
            dtype=tf.uint8,
            shape=(self._config['minibatch_size']),
            name='actions',
        )

        Q_output = self._get_tf_t('Q_network/output/Qs:0')
        one_hot_actions = tf.one_hot(
            actions,
            self.get_num_of_actions(),
        )
        Qs_of_action = tf.reduce_sum(
            tf.multiply(Q_output, one_hot_actions),
            axis=1,
            name='Qs_of_action',
        )

        diffs = ys - Qs_of_action
        if self._config['clip_error']:
            print('Use error clipping...')
            deltas = tf.minimum(
                tf.square(diffs),
                tf.abs(diffs),
                name='clipped_deltas',
            )
        else:
            deltas = tf.square(diffs, name='deltas')

        loss = tf.reduce_mean(deltas, name='loss')

        optimizer = tf.train.RMSPropOptimizer(
            learning_rate=self._config['learning_rate'],
            decay=self._config['rms_prop_decay'],
            momentum=self._config['gradient_momentum'],
            epsilon=self._config['min_squared_gradient'],
            centered=True,
        )
        optimizer.minimize(
            loss=loss,
            name='minimize_loss',    
        )

    def _build_validation_ops(self):
        Q_output = self._get_tf_t('Q_network/output/Qs:0')
        average_Q = tf.reduce_mean(
            tf.reduce_max(Q_output, axis=1),
            name='average_Q',
        )

        rewards = tf.placeholder(
           tf.float32,
           name='rewards',
        )
        ave_reward = tf.reduce_mean(
            rewards,
            name='ave_reward',
        )

    def _build_update_ops(self):
        self._target_Q_update_ops = []
        for layer_name, layer_conf in self._config['Q_network']:
            for var_name in ['W', 'b']:
                layer_var_name = layer_name + '/' + var_name
                source_var = self._tf_graph.get_tensor_by_name(
                    'Q_network/' + layer_var_name + ':0'
                )
                target_var = self._tf_graph.get_tensor_by_name(
                    'target_Q_network/' + layer_var_name + ':0'
                )
                self._target_Q_update_ops.append(
                    tf.assign(target_var, source_var, name=layer_var_name)
                )

    def _build_summary_ops(self):
        with tf.variable_scope('stat'):
            tf.summary.scalar(
                name='used_gpu_memory',
#                tensor=tf.contrib.memory_stats.MaxBytesInUse(),
                tensor=self._get_tf_t('stat/used_gpu_memory:0')
            )
        with tf.variable_scope('train'):
            tf.summary.scalar(
                name='loss',
                tensor=self._get_tf_t('train/loss:0'),
            )
        with tf.variable_scope('validation'):
            tf.summary.scalar(
                name='Q',
                tensor=self._get_tf_t('validation/average_Q:0'),
            )
            tf.summary.scalar(
                name='reward',
                tensor=self._get_tf_t('validation/ave_reward:0'),
            )

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
        return tf.truncated_normal_initializer(
            mean=self._config['var_init_mean'],
            stddev=self._config['var_init_stddev'],
            seed=self._random_seed,
        )

    def _get_tf_op(self, name):
        return self._tf_graph.get_operation_by_name(name)

    def _get_tf_t(self, name):
        return self._tf_graph.get_tensor_by_name(name)

    def get_num_of_actions(self):
        return self._env.action_space.n

    def preprocess_observation(
        self,
        observation,
        new_width=84,
        new_height=110,
        threshold=1,
    ):
        img = Image.fromarray(observation)
        img = img.convert('L')
        new_size = (new_width, new_height)
        img = img.resize(new_size)
        obs = np.array(img, dtype=np.uint8)
        obs = obs[-new_width:, :]
        return obs

    def play(
        self,
        max_num_of_steps=(10**6),
        max_num_of_episodes=None,
        train=True,
        save_path=None,
        step=None,
        run_name=None,
        var_list=None,
    ):

        s = self._config['Q_network_input_size']
        c = self._config['agent_history_length']
        rss = self._config['replay_start_size']
        n_steps = self._config['final_exploration_frame']
        min_epsilon = self._config['final_exploration']
        tnuf = self._config['target_network_update_frequency']
        stats = {
            'average_loss': [],
            'average_reward': [],
            'average_Q': [],
        }

        with self._tf_graph.as_default():
            saver = tf.train.Saver(var_list=var_list)
            if save_path is not None:
                saver.restore(self._tf_session, save_path)
                step = self._get_step_from_checkpoint(save_path)

            if step is None:
                step = 0

            if train:
                if run_name is None:
                    run_name = ('{:02}{:02}_{:02}{:02}{:02}'
                                .format(*time.localtime()[1:6]))
                summary_writer = tf.summary.FileWriter(
                    logdir='logs/{}'.format(run_name),
                    graph=self._tf_graph
                )
                with open('runs/{}'.format(run_name), 'w') as fp:
                    json.dump(self._config, fp)
            else:
                play_images = []
                actions = []
                rewards = []

            done = True
            observation = None
            episode = 0
            rewards_per_episode = []
            losses = []
            phi = np.empty((1, s, s, c), dtype=np.float32)
            num_no_ops = None

            while (
                step < max_num_of_steps
                or (max_num_of_episodes is not None
                    and episode <= max_num_of_episodes)
            ):
                if done:
                    done = False
                    observation = self._env.reset()
                    num_no_ops = 0
                    phi.fill(0)
                    episode += 1
                    rewards_per_episode.append(0)
                    losses.append(0)

                state = self.preprocess_observation(observation)

                if train:
                    if step < rss:
                        epsilon = 1
                    elif (step < n_steps):
                        epsilon = 1 - (1 - min_epsilon) / n_steps * step
                    else:
                        epsilon = min_epsilon
                else:
                    epsilon = self._config['evaluation_exploration'] 

                phi[0,:,:,:3] = phi[0,:,:,1:]
                phi[0,:,:,3] = np.array(state, dtype=np.float32)

                action = self._get_action(epsilon, phi)
                if action == ACTION_NO_OP and num_no_ops is not None:
                    num_no_ops += 1
                    if num_no_ops == self._config['no_op_max']:
                        while action != ACTION_NO_OP:        
                            action = self._env.action_space.sample()
                else:
                    num_no_ops = None

                observation, reward, done, _ = self._env.step(action)
                step += 1
                rewards_per_episode[-1] += reward

                if train:
                    self._replay_memory.store(state, action, reward, int(done))

                    if (step % 1000 == 0):
                        try:
                            used_gpu_memory = self._get_used_gpu_memory()
                        except:
                            used_gpu_memory = 0.0
                        tf_assign_op = tf.assign(
                            self._get_tf_t('stat/used_gpu_memory:0'),
                            used_gpu_memory,
                        )
                        _, used_gpu_memory_summary = self._tf_session.run(
                            [tf_assign_op,
                             self._get_tf_t('summary/stat/used_gpu_memory:0')]
                        )
                        summary_writer.add_summary(
                            used_gpu_memory_summary,
                            step,
                        )

                    if (self._replay_memory.get_size() >= rss):

                        if (self._validation_states is None):
                            vs = self._config['validation_size']
                            self._validation_states = np.zeros(
                                (vs, s, s, c), dtype=np.float32,
                            )
                            samples = self._replay_memory.get_samples(vs)
                            self._replay_memory.fill_states(
                                samples,
                                self._validation_states,
                            )

                        loss, loss_summary = self._optimize_Q()
                        summary_writer.add_summary(loss_summary, step)
                        losses[-1] += loss

                        if (step % 1000 == 0):
                            ave_loss = np.mean(loss)

                            # Get ave. reward per episode
                            # and its summary string.
                            fetches = [
                                self._get_tf_t('validation/ave_reward:0'),
                                self._get_tf_t('summary/validation/reward:0'),
                            ]
                            feed_dict = {
                                self._get_tf_t('validation/rewards:0'):
                                    rewards_per_episode,
                            }
                            ave_reward, reward_summary = self._tf_session.run(
                                fetches=fetches,
                                feed_dict=feed_dict,
                            )
                            summary_writer.add_summary(reward_summary, step)

                            # Get ave. Q over validation states.
                            fetches = [
                                self._get_tf_t('validation/average_Q:0'),
                                self._get_tf_t('summary/validation/Q:0'),
                            ]
                            feed_dict = {
                                self._get_tf_t('Q_network/input:0'):
                                    self._validation_states,
                            }
                            ave_Q, Q_summary = self._tf_session.run(
                                fetches=fetches,
                                feed_dict=feed_dict,
                            )
                            summary_writer.add_summary(Q_summary, step)

                            print(
                                'step: {}, ave. loss: {:g}, '
                                'ave. reward: {:g}, ave. Q: {:g}'
                                .format(
                                    step,
                                    ave_loss,
                                    ave_reward,
                                    ave_Q,
                                )
                            )
                            stats['average_loss'].append(ave_loss)
                            stats['average_reward'].append(ave_reward)
                            stats['average_Q'].append(ave_Q)

                        if (tnuf is not None and step % tnuf == 0):
                            self._update_target_Q_network()

                        if (step % 10000 == 0 or step == max_num_of_steps):
                            save_path = saver.save(
                                self._tf_session,
                                'checkpoints/{}'.format(run_name),
                                global_step=step,
                            )
                            print('checkpoint saved at {}'.format(save_path))

                        if (step % 50000 == 0):
                            rewards_per_episode = rewards_per_episode[-1:]
                            losses = losses[-1:]
                else:
                    actions.append(action)
                    rewards.append(reward)
                    play_images.append(Image.fromarray(observation))

            if train:
                if (step >= max_num_of_steps):
                    print(
                        'Finished: reached the maximum number of steps {}.'
                        .format(max_num_of_steps)
                    )
                summary_writer.close()
                return stats
            else:
                play_images[0].save(
                    'play_{}.gif'.format(self.game_name),
                    save_all=True,
                    append_images=play_images[1:],
                    duration=30,
                )
                return (actions, rewards)

    def _get_action(self, epsilon, phi):
        if (self._np_random.rand() < epsilon):
            action = self._env.action_space.sample()
        else:
            action = self._get_action_from_Q(phi)
        return action

    def _get_action_from_Q(self, phi):
        Qs = self._get_Q_values(phi)
        # TODO: Tiebreaking
        return np.argmax(Qs)

    def _get_Q_values(self, states, from_target_Q=False):
        assert(states.ndim == 4)
        assert(states.dtype == np.float32)
        name = 'Q_network'
        if from_target_Q:
            name = 'target_' + name
        Q_input = self._get_tf_t(name + '/input:0')
        Q_output = self._get_tf_t(name + '/output/Qs:0')
        return self._tf_session.run(
            Q_output,
            feed_dict={Q_input: states},
        )

    def _optimize_Q(self):
        gamma = self._config['discount_factor']
        minibatch = self._replay_memory.sample_minibatch()
        mask = (1 - minibatch['terminals'])
        if self._config['target_network_update_frequency'] is None:
            from_target_Q = False
        else:
            from_target_Q = True
        Qs_next = self._get_Q_values(
            minibatch['next_states'],
            from_target_Q=from_target_Q,
        )
        ys = minibatch['rewards'] + gamma * (
            np.amax(
                Qs_next * mask[:, np.newaxis],
                axis=1,
            )
        )

        fetches = [
            self._get_tf_op('train/minimize_loss'),
            self._get_tf_t('train/loss:0'),
            self._get_tf_t('summary/train/loss:0'),
        ]
        feed_dict = {
            self._get_tf_t('Q_network/input:0'): minibatch['states'],
            self._get_tf_t('train/actions:0'): minibatch['actions'],
            self._get_tf_t('train/ys:0'): ys,
        }
        _, loss, loss_summary = self._tf_session.run(
            fetches=fetches,
            feed_dict=feed_dict,
        )

        return (loss, loss_summary)

    def _update_target_Q_network(self):
        self._tf_session.run(self._target_Q_update_ops)

    def _get_step_from_checkpoint(self, save_path):
        return int(save_path.split('-')[-1])

    def _get_used_gpu_memory(self):
        # TODO subprocess too slow, use nvmlDeviceGetComputeRunningProcesses.
        smi_outputs = subprocess.check_output(
            NVIDIA_SMI_ARGS,
            universal_newlines=True,
        ).splitlines()

        used_gpu_memory = 0.0

        for output in smi_outputs:
            pid_str, used_gpu_memory_str = output.split(', ')
            if(int(pid_str) == self._pid):
                val_str, unit_str = used_gpu_memory_str.split(' ')
                if unit_str == 'MiB':
                    used_gpu_memory = float(val_str)
                else:
                    raise RuntimeError(
                        'Unknown memory unit: {}'.format()
                    )

        return used_gpu_memory

    def _get_weight_dict(self):
        weight_dict = {}
        for layer_name, layer_conf in self._config['Q_network']:
            for var_name in ['W', 'b']:
                layer_var_name = layer_name + '/' + var_name
                key = 'Q_network/{}'.format(layer_var_name)
                weight_dict[key] = self._get_tf_t(key + ':0')
                key = 'target_' + key
                weight_dict[key] = self._get_tf_t(key + ':0')
        return weight_dict
