import tensorflow as tf
import numpy as np


class DQN:
    def __init__(self, session, state_size, action_size, learning_rate, gamma):
        self.sess = session
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.temperature_c = np.expand_dims(0.5, axis=1) # NOT USED YET

        self.actions = tf.placeholder(tf.float32, [None, action_size], name="actions")
        self.target_Q = tf.placeholder(dtype=tf.float32, shape=[None], name="target")
        self.temperature = tf.placeholder(dtype=float, shape=[None], name="temperature")

        self.inputs = tf.placeholder(dtype=tf.float32, shape=[None, state_size], name="inputs")
        self.dense1 = tf.layers.dense(inputs=self.inputs, units=32, activation=tf.nn.relu)
        self.dense2 = tf.layers.dense(inputs=self.dense1, units=64, activation=tf.nn.relu)
        # TEMPERATURE IMPLEMENTATION - NOT USED YET
        # self.output_pre = tf.layers.dense(inputs=self.dense2, units=action_size, activation=None)
        # self.output = tf.nn.softmax(tf.multiply(self.output_pre, self.temperature))
        self.output = tf.layers.dense(inputs=self.dense2, units=action_size, activation=tf.nn.softmax)

        self.current_Q = tf.reduce_sum(tf.multiply(self.output, self.actions), axis=1)
        self.loss = tf.reduce_mean(tf.square(self.target_Q - self.current_Q))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        init = tf.global_variables_initializer()
        session.run(init)

    def select_action(self, state):
        """
        Used for selection an action based on a given state
        :param state: current state
        :return: chosen action
        """
        state = np.expand_dims(state, 0)
        pred = self.sess.run(self.output, feed_dict={self.inputs: state, self.temperature: self.temperature_c})[0]
        action = np.random.multinomial(1, pred)
        return action

    """ NOT USED FOR NOW
    def predict(self, state):
        state = np.expand_dims(state, 0)
        pred = self.sess.run(self.output, feed_dict={self.inputs: state, self.temperature: self.temperature_c})
        return pred"""

    def learn(self, state_batch, new_state_batch, reward_batch, action_batch, done_batch):
        """
        Used for training a model
        :param state_batch: (state_size-long vector) batch of startint states
        :param new_state_batch: (state_size-long vector) batch of ending states
        :param reward_batch: (float) batch of rewards for an action
        :param action_batch: (action_size-long one-hot vector) batch of played actions
        :param done_batch: (boolean - mapped as True=0 False=1) batch of booleans indicating whether the scene has ended
        :return:
        """
        done_batch = tuple(map(lambda x: 0 if x else 1, done_batch))
        current_Q = np.max(self.sess.run(self.output, feed_dict={self.inputs: new_state_batch, self.temperature: self.temperature_c}), axis=1)
        current_Q = reward_batch + self.gamma * current_Q
        # current_Q = reward_batch + self.gamma * current_Q * done_batch

        _, cost_value = self.sess.run([self.optimizer, self.loss], {self.inputs: state_batch,
                                                                    self.target_Q: current_Q,
                                                                    self.actions: action_batch,
                                                                    self.temperature: self.temperature_c})
        # print(cost_value)
