from CartPoleEnv import CartPoleEnv as cenv
from DQN import DQN
from Memory import Memory
import tensorflow as tf
import numpy as np

print("ELKEZDODOTT")
if __name__ == "__main__":
    env = cenv()
    with tf.Session() as sess:
        model = DQN(sess, 4, 2, 0.0003, 0.95)
        memory = Memory(100000)

        while True:
            state = env.reset()
            for t in range(1000):
                env.render()

                action = model.select_action(state)
                new_state, reward, done, info = env.step(np.argmax(action))
                # if done:
                #     reward = -1.0

                memory.add((state, new_state, reward, action, done))

                if len(memory.buffer) > 128:
                    state_batch, new_state_batch, reward_batch, action_batch, done_batch = memory.sample(128)
                    model.learn(state_batch, new_state_batch, reward_batch, action_batch, done_batch)
                state = new_state

                if len(memory.buffer) > 10000:
                    state_batch, new_state_batch, reward_batch, action_batch, done_batch = memory.sample(10000)
                    current_Q = np.max(sess.run(model.output, feed_dict={model.inputs: new_state_batch,
                                                                             model.temperature: model.temperature_c}), axis=1)
                    current_Q = reward_batch + model.gamma * current_Q
                    cost_value = sess.run([model.loss], {model.inputs: state_batch,
                                                         model.target_Q: current_Q,
                                                         model.actions: action_batch,
                                                         model.temperature: model.temperature_c})
                    print(cost_value)
                if done:
                    print("Episode finished after {} timesteps".format(t + 1))
                    break
