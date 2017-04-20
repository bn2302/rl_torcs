import tensorflow as tf


class Network(object):

    HIDDEN1_UNITS = 300
    HIDDEN2_UNITS = 600

    def __init__(self, sess, state_size, action_size, tau, learning_rate,
                 net_scope, target_net_scope):

        self.sess = sess
        self.tau = tau
        self.learning_rate = learning_rate
        self.state_size = state_size
        self.action_size = action_size
        self.net_scope = net_scope
        self.target_net_scope = target_net_scope
        self.is_training = False

        self.weights = None
        self.target_weights = None
        self.cp_trgt_wgt_frm_wgt = None

    def _create_target_train(self):
        self.cp_trgt_wgt_frm_wgt = tf.group(
            *[v1.assign(self.tau*v2 + (1-v1))
              for v1, v2 in zip(self.target_weights, self.weights)])

    def target_train(self):
        self.is_training = True
        self.sess.run(self.cp_trgt_wgt_frm_wgt)


class CriticNetwork(Network):

    def __init__(self, sess, state_size, action_size, tau, learning_rate):

        super(CriticNetwork, self).__init__(sess, state_size, action_size,
                                            tau, learning_rate,
                                            'critic_network',
                                            'target_critic_network')
        # Now create the model
        self.critic, self.weights, self.state, self.action = \
            self._create_network(self.net_scope)
        self.target_critic, self.target_weights, self.target_state, \
            self.target_action = self._create_network(self.target_net_scope)
        self._create_target_train()
        # GRADIENTS for policy update
        self.action_grads = tf.gradients(self.critic, self.action)
        self.optimize, self.loss, self.expected_critic = self._create_train()
        self.sess.run(tf.global_variables_initializer())

    def target_predict(self, states, actions):
        self.is_training = False
        return self.sess.run(self.target_critic,
                             feed_dict={self.target_state: states,
                                        self.target_action: actions})

    def gradients(self, states, actions):
        self.is_training = False
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states, self.action: actions})[0]

    def train(self, expected_critic, states, actions):
        self.is_training = True
        loss, _ = self.sess.run([self.loss, self.optimize], feed_dict={
            self.expected_critic: expected_critic, self.state: states,
            self.action: actions})

        return loss

    def _create_network(self, scope):
        with tf.variable_scope(scope):

            state = tf.placeholder(
                shape=[None, self.state_size], dtype=tf.float32, name='state')
            action = tf.placeholder(
                shape=[None, self.action_size],
                dtype=tf.float32, name='action')

            s_layer1 = tf.layers.batch_normalization(
                tf.layers.dense(
                    inputs=state, activation=tf.nn.relu,
                    units=CriticNetwork.HIDDEN1_UNITS),
                training=self.is_training, name='s_layer_1')

            s_layer2 = tf.layers.batch_normalization(
                tf.layers.dense(
                    inputs=s_layer1,
                    units=CriticNetwork.HIDDEN2_UNITS),
                training=self.is_training, name='s_layer_2')

            a_layer = tf.layers.batch_normalization(
                tf.layers.dense(
                    inputs=action,
                    units=CriticNetwork.HIDDEN2_UNITS),
                training=self.is_training, name='a_layer')

            c_layer = tf.layers.batch_normalization(
                tf.layers.dense(
                    inputs=(s_layer2 + a_layer),
                    activation=tf.nn.relu,
                    units=CriticNetwork.HIDDEN2_UNITS),
                training=self.is_training, name='c_layer')

            critic = tf.layers.batch_normalization(
                tf.layers.dense(inputs=c_layer,
                                units=self.action_size),
                training=self.is_training, name='critic')

            weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                        scope=scope)

        return critic, weights, state, action

    def _create_train(self):
        expected_critic = tf.placeholder(shape=[None, self.action_size],
                                         dtype=tf.float32,
                                         name='expected_critic')

        loss = tf.reduce_mean(tf.square(expected_critic-self.critic),
                              name="loss")

        trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        optimize = trainer.minimize(loss, name='optimize')

        return optimize, loss, expected_critic


class ActorNetwork(Network):

    def __init__(self, sess, state_size, action_size, tau, learning_rate):

        super(ActorNetwork, self).__init__(sess, state_size, action_size, tau,
                                           learning_rate, 'actor_network',
                                           'target_actor_network')
        # Now create the model
        self.action, self.weights, self.state = \
            self._create_network(self.net_scope)
        self.target_action, self.target_weights, self.target_state = \
            self._create_network(self.target_net_scope)
        self._create_target_train()
        self.optimize, self.action_gradient = self._create_train()
        self.sess.run(tf.global_variables_initializer())

    def predict(self, states):
        self.is_training = False
        return self.sess.run(self.action, feed_dict={self.state: states})

    def target_predict(self, states):
        self.is_training = False
        return self.sess.run(self.target_action,
                             feed_dict={self.target_state: states})

    def train(self, states, action_grads):
        self.training = True
        self.sess.run(self.optimize, feed_dict={
            self.state: states, self.action_gradient: action_grads})

    def _create_network(self, scope):
        with tf.variable_scope(scope):
            state = tf.placeholder(tf.float32, [None, self.state_size],
                                   name='state')

            hidden0 = tf.layers.batch_normalization(
                tf.layers.dense(
                    inputs=state, activation=tf.nn.relu,
                    units=ActorNetwork.HIDDEN1_UNITS),
                training=self.is_training, name='hidden_0')

            hidden1 = tf.layers.batch_normalization(
                tf.layers.dense(inputs=hidden0, activation=tf.nn.relu,
                                units=ActorNetwork.HIDDEN2_UNITS),
                training=self.is_training, name='hidden_1')

            steering = tf.layers.batch_normalization(
                tf.layers.dense(
                    inputs=hidden1, units=1, activation=tf.nn.tanh),
                training=self.is_training, name='steering')

            acceleration = tf.layers.batch_normalization(
                tf.layers.dense(
                    inputs=hidden1, units=1, activation=tf.nn.sigmoid),
                training=self.is_training, name='acceleration')

            brake = tf.layers.batch_normalization(
                tf.layers.dense(
                    inputs=hidden1, units=1, activation=tf.nn.sigmoid),
                training=self.is_training, name='brake')

            action = tf.concat(
                [steering, acceleration, brake], name='action', axis=1)

            weights = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

        return action, weights, state

    def _create_train(self):
        action_gradient = tf.placeholder(tf.float32, [None, self.action_size])
        params_grad = tf.gradients(self.action, self.weights,
                                   tf.negative(action_gradient))
        grads = zip(params_grad, self.weights)
        optimize = tf.train.AdamOptimizer(
            self.learning_rate).apply_gradients(grads)
        return optimize, action_gradient



