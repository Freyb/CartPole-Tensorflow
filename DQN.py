import tensorflow as tf
import numpy as np


class DQN:
    def __init__(self, session, state_size, action_size, learning_rate, gamma):
        self.sess = session
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.temperature_c = np.expand_dims(0.5, axis=1)

        self.actions = tf.placeholder(tf.float32, [None, action_size], name="actions")
        self.target_Q = tf.placeholder(dtype=tf.float32, shape=[None], name="target")
        self.temperature = tf.placeholder(dtype=float, shape=[None], name="temperature")

        self.inputs = tf.placeholder(dtype=tf.float32, shape=[None, state_size], name="inputs")
        self.dense1 = tf.layers.dense(inputs=self.inputs, units=32, activation=tf.nn.relu)
        self.dense2 = tf.layers.dense(inputs=self.dense1, units=64, activation=tf.nn.relu)
        # self.output_pre = tf.layers.dense(inputs=self.dense2, units=action_size, activation=None)
        # self.output = tf.nn.softmax(tf.multiply(self.output_pre, self.temperature))
        self.output = tf.layers.dense(inputs=self.dense2, units=action_size, activation=tf.nn.softmax)

        self.current_Q = tf.reduce_sum(tf.multiply(self.output, self.actions), axis=1)
        self.loss = tf.reduce_mean(tf.square(self.target_Q - self.current_Q))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        init = tf.global_variables_initializer()
        session.run(init)

    def select_action(self, state):
        state = np.expand_dims(state, 0)
        pred = self.sess.run(self.output, feed_dict={self.inputs: state, self.temperature: self.temperature_c})[0]
        # action = self.sess.run(tf.multinomial(pred, 1))
        action = np.random.multinomial(1, pred)
        # print(pred)
        # print(action)
        return action

    def predict(self, state):
        state = np.expand_dims(state, 0)
        pred = self.sess.run(self.output, feed_dict={self.inputs: state, self.temperature: self.temperature_c})
        return pred

    def learn(self, state_batch, new_state_batch, reward_batch, action_batch):
        # state = np.expand_dims(state, 0)
        # new_state = np.expand_dims(new_state, 0)
        current_Q = np.max(self.sess.run(self.output, feed_dict={self.inputs: new_state_batch, self.temperature: self.temperature_c}), axis=1)
        current_Q = reward_batch + self.gamma*current_Q
        # current_Q = np.expand_dims(current_Q, 0)

        # action_oh = np.zeros(self.action_size)
        # action_oh[action] = 1
        # action = np.expand_dims(action, 0)
        _, cost_value = self.sess.run([self.optimizer, self.loss], {self.inputs: state_batch,
                                                                    self.target_Q: current_Q,
                                                                    self.actions: action_batch,
                                                                    self.temperature: self.temperature_c})
        # print(cost_value)
