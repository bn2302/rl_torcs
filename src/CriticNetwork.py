import tensorflow as tf


class CriticNetwork(object):

    HIDDEN1_UNITS = 300
    HIDDEN2_UNITS = 600

    def __init__(self, sess, state_size, action_size, batch_size, tau,
                 learning_rate):
        self.sess = sess
        self.batch_size = batch_size
        self.tau = tau
        self.learning_rate = learning_rate
        self.action_size = action_size

        # Now create the model
        self.model, self.action, self.state = self.create_critic_network(
            state_size, action_size)
        self.target_model, self.target_action, self.target_state = \
            self.create_critic_network(state_size, action_size)
        self.action_grads = tf.gradients(
            self.model.output, self.action)  # GRADIENTS for policy update
        self.sess.run(tf.initialize_all_variables())

    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i, _ in enumerate(critic_weights):
            critic_target_weights[i] = self.tau * critic_weights[i] + \
                (1 - self.tau) * critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def create_critic_network(self, state_size, action_dim):

        state = tf.placeholder(shape=[None, state_size], dtype=tf.float32)
        action = tf.placeholder(shape=[None, action_dim], dtype=tf.float32)

        def model(state, action):
            state_layer1 = tf.layers.dense(inputs=state,
                                           units=CriticNetwork.HIDDEN1_UNITS,
                                           activation=tf.nn.relu)
            state_linear_weight = tf.Variable(tf.truncated_normal(
                [CriticNetwork.HIDDEN1_UNITS, CriticNetwork.HIDDEN2_UNITS]),
                dtype=tf.float32)
            state_linear_bias = tf.Variable(
                tf.zeros([CriticNetwork.HIDDEN2_UNITS]))
            state_layer2 = (tf.matmul(state_layer1, state_linear_weight) +
                            state_linear_bias)

            action_layer1 = tf.layers.dense(inputs=action,
                                            units=CriticNetwork.HIDDEN2_UNITS,
                                            activation=tf.nn.relu)

            sum_layer = state_layer2 + action_layer1

            hidden_layer1 = tf.layers.dense(inputs=sum_layer,
                                            units=CriticNetwork.HIDDEN2_UNITS,
                                            activation=tf.nn.relu)

            critic_linear_weight = tf.Variable(tf.truncated_normal(
                [CriticNetwork.HIDDEN2_UNITS, action_dim]),
                dtype=tf.float32)
            critic_linear_bias = tf.Variable(
                tf.zeros([action_dim]))
            critic_layer = (tf.matmul(hidden_layer1, critic_linear_weight) +
                            critic_linear_bias)

            return critic_layer

        return model, action, state

    def train(self, model, state, action, y_t):

        loss = tf.losses.mean_squared_error(y_t, model(state, action))
        
        trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        updateModel = trainer.minimize(loss)

        return model, action, state
