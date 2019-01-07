from keras import optimizers, backend, models, layers
class Critic:
    
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.critic_model()

    def critic_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # input layers
        states = layers.Input( shape=(self.state_size,), name='states')
        actions = layers.Input( shape=(self.action_size,), name='actions')

        # state hidden layers
        states_net = layers.Dense(units=32, activation="relu")(states)
        states_net = layers.Dense(units=64, activation="relu")(states_net)

        # action hidden layer
        actions_net = layers.Dense(units=32, activation="relu")(actions)
        actions_net = layers.Dense(units=64, activation="relu")(actions_net)

        # Combine state and action pathways
        net = layers.Add()([states_net, actions_net])
        net = layers.Activation('relu')(net)

        # Add final output layer to prduce action values (Q values)
        starting_weights =layers.initializers.RandomUniform(minval=-0.05, maxval=0.05)
        Q_values = layers.Dense(units=1, kernel_initializer=starting_weights, name='q_values')(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam(lr=0.01)
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = backend.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = backend.function(
            inputs=[*self.model.input, backend.learning_phase()],
            outputs=action_gradients)