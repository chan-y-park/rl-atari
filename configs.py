from copy import deepcopy

dqn_configuration = {
    'Q_network': [
#        ('conv1', {'W_size': 8, 'stride': 4, 'in': 4, 'out': 16}),
#        ('conv2', {'W_size': 4, 'stride': 2, 'in': 16, 'out': 32}),
        ('conv1', {'filter_size': 8, 'stride': 4, 'num_filters': 16}),
        ('conv2', {'filter_size': 4, 'stride': 2, 'num_filters': 32}),
        ('fc1', {'num_relus': 256}),
        ('output', {}),
    ],
    'Q_network_input_size': 84,
    'var_init_mean': 0.0,
    'var_init_stddev': 0.01,
    'minibatch_size': 32,
    'replay_memory_size': 10 ** 6,
    'agent_history_length': 4,
    'discount_factor': 0.99,
    'action_repeat': 4,
    'update_frequency': 4,
    'learning_rate': 0.00025,
    'rms_prop_decay': 0.95,
    'gradient_momentum': 0.0,
    'min_squared_gradient': 0.01,
    'initial_exploration': 1,
    'final_exploration': 0.1,
    'final_exploration_frame': 10 ** 6,
    'replay_start_size': 5 * (10 ** 4),
    'no_op_max': 30,
    'validation_size': 500,
    'evaluation_exploration': 0.05,
    'target_network_update_frequency': None,
    'clip_error': False,
}

dqn_nips_configuration = deepcopy(dqn_configuration)

dqn_nature_configuration = deepcopy(dqn_configuration)
dqn_nature_configuration['target_network_update_frequency'] = 10 ** 4
dqn_nature_configuration['Q_network'] = [
    ('conv1', {'filter_size': 8, 'stride': 4, 'num_filters': 32}),
    ('conv2', {'filter_size': 4, 'stride': 2, 'num_filters': 64}),
    ('conv3', {'filter_size': 3, 'stride': 1, 'num_filters': 64}),
    ('fc1', {'num_relus': 512}),
    ('output', {}),
]
dqn_nature_configuration['clip_error'] = True
