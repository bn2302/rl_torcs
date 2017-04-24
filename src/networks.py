import tensorflow as tf
import numpy as np


class Network(object):

    HIDDEN1_UNITS = 300
    HIDDEN2_UNITS = 600

    def __init__(self, state_size, action_size, trainer):
        self.state_size = state_size
        self.action_size = action_size
        self.trainer = trainer
        self.is_training = False


class ActorCriticBaseNetwork(Network):

    def __init__(self, state_size, action_size, trainer, tau):
        super(ActorCriticBaseNetwork, self).__init__(
            state_size, action_size, trainer)

        self.tau = tau
        self.weights = None
        self.target_weights = None
        self.cp_trgt_wgt_frm_wgt = None

    def _create_target_train(self):
        self.cp_trgt_wgt_frm_wgt = tf.group(
            *[v1.assign(self.tau*v2 + (1-v1))
              for v1, v2 in zip(self.target_weights, self.weights)])

    def target_train(self, sess):
        self.is_training = True
        sess.run(self.cp_trgt_wgt_frm_wgt)


class CriticNetwork(ActorCriticBaseNetwork):

    def __init__(self, state_size, action_size, trainer, tau):

        super(CriticNetwork, self).__init__(
            state_size, action_size, trainer, tau)

        self.net_scope = 'critic_network'
        self.target_net_scope = 'target_critic_network'
        # Now create the model
        self.critic, self.weights, self.state, self.action = \
            self._create_network(self.net_scope)
        self.target_critic, self.target_weights, self.target_state, \
            self.target_action = self._create_network(self.target_net_scope)
        self._create_target_train()
        # GRADIENTS for policy update
        self.action_grads = tf.gradients(self.critic, self.action)
        self.optimize, self.loss, self.expected_critic = self._create_train()

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

        optimize = self.trainer.minimize(loss, name='optimize')

        return optimize, loss, expected_critic

    def target_predict(self, sess, states, actions):
        self.is_training = False
        return sess.run(
            self.target_critic,
            feed_dict={self.target_state: states,
                       self.target_action: actions})

    def gradients(self, sess, states, actions):
        self.is_training = False
        return sess.run(
            self.action_grads,
            feed_dict={self.state: states, self.action: actions})[0]

    def train(self, sess, expected_critic, states, actions):
        self.is_training = True
        loss, _ = sess.run(
            [self.loss, self.optimize],
            feed_dict={
                self.expected_critic: expected_critic, self.state: states,
                self.action: actions})

        return loss


class ActorNetwork(ActorCriticBaseNetwork):

    def __init__(self, state_size, action_size, trainer, tau):

        super(ActorNetwork, self).__init__(
            state_size, action_size, trainer, tau)

        self.net_scope = 'actor_network'
        self.target_net_scope = 'target_actor_network'
        # Now create the model
        self.action, self.weights, self.state = \
            self._create_network(self.net_scope)
        self.target_action, self.target_weights, self.target_state = \
            self._create_network(self.target_net_scope)
        self._create_target_train()
        self.optimize, self.action_gradient = self._create_train()

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
                    inputs=hidden1, units=1, activation=tf.nn.tanh),
                training=self.is_training, name='acceleration')

            action = tf.concat(
                [steering, acceleration], name='action', axis=1)

            weights = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

        return action, weights, state

    def _create_train(self):
        action_gradient = tf.placeholder(tf.float32, [None, self.action_size])
        params_grad = tf.gradients(self.action, self.weights,
                                   tf.negative(action_gradient))
        grads = zip(params_grad, self.weights)
        optimize = self.trainer.apply_gradients(grads)
        return optimize, action_gradient

    def predict(self, sess, states):
        self.is_training = False
        return sess.run(self.action, feed_dict={self.state: states})

    def target_predict(self, sess, states):
        self.is_training = False
        return sess.run(
            self.target_action,
            feed_dict={self.target_state: states})

    def train(self, sess, states, action_grads):
        self.training = True
        sess.run(
            self.optimize,
            feed_dict={
                self.state: states, self.action_gradient: action_grads})


class AC_Network(Network):

    def __init__(self, state_size, action_size, trainer, scope):
        super(AC_Network, self).__init__(
            state_size, action_size, trainer)
        self.scope = scope
        self.is_training = False
        self._create_network()
        if self.scope != 'global':
            self._create_train()

    @staticmethod
    def update_target_graph(from_scope, to_scope):
        from_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
        to_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

        op_holder = []
        for from_var, to_var in zip(from_vars, to_vars):
            op_holder.append(to_var.assign(from_var))

        return op_holder

    def _create_network(self):
        with tf.variable_scope(self.scope):
            # Input and visual encoding layers
            self.inputs = tf.placeholder(
                shape=[None, self.state_size], dtype=tf.float32)

            s_layer1 = tf.layers.batch_normalization(
                tf.layers.dense(
                    inputs=self.inputs, activation=tf.nn.relu,
                    units=AC_Network.HIDDEN1_UNITS),
                training=self.is_training, name='s_layer_1')

            s_layer2 = tf.layers.batch_normalization(
                tf.layers.dense(
                    inputs=s_layer1, activation=tf.nn.relu,
                    units=AC_Network.HIDDEN2_UNITS),
                training=self.is_training, name='s_layer_2')

            # Output layers for policy and value estimations
            self.policy_mu = tf.layers.batch_normalization(
                tf.layers.dense(
                    inputs=s_layer2, units=2, activation=tf.nn.tanh),
                training=self.is_training, name='policy_mu')

            self.policy_sd = tf.clip_by_value(
                tf.layers.batch_normalization(
                    tf.layers.dense(
                        inputs=s_layer2, units=2, activation=tf.nn.softplus),
                    training=self.is_training),
                [1e-2]*self.action_size, [0.5]*self.action_size,
                name='policy_sd')

            self.value = tf.layers.batch_normalization(
                tf.layers.dense(inputs=s_layer2, units=1),
                training=self.is_training, name='value')

            self.normal_dist = tf.contrib.distributions.Normal(
                self.policy_mu, self.policy_sd, name='normal_dist')

            self.action = tf.clip_by_value(
                self.normal_dist.sample(1),
                [-1.0]*self.action_size, [1.0]*self.action_size,
                name='action')

    def _create_train(self):
        with tf.variable_scope(self.scope):
            self.actions = tf.placeholder(
                shape=[None, self.action_size], dtype=tf.float32,
                name='actions')
            self.target_v = tf.placeholder(
                shape=[None], dtype=tf.float32, name='target_v')
            self.advantages = tf.placeholder(
                shape=[None], dtype=tf.float32, name='advantages')

            log_prob = self.normal_dist.log_prob(self.actions)
            exp_v = tf.transpose(
                tf.multiply(tf.transpose(log_prob), self.advantages))
            entropy = self.normal_dist.entropy()
            exp_v = 0.01 * entropy + exp_v
            self.policy_loss = tf.reduce_sum(-exp_v)

            self.value_loss = 0.5 * tf.reduce_sum(
                tf.square(self.target_v - tf.reshape(self.value, [-1])))

            self.loss = 0.5*self.value_loss + self.policy_loss

            local_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

            self.gradients = tf.gradients(self.loss, local_vars)
            self.var_norms = tf.global_norm(local_vars)

            grads, self.grad_norms = tf.clip_by_global_norm(
                self.gradients, 40.0)

            global_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
            self.apply_grads = self.trainer.apply_gradients(
                zip(grads, global_vars))

    def predict(self, sess, state):
        action = sess.run(
            self.action,
            feed_dict={self.inputs: [state]})
        return action[0]
