from CartPoleEnv import CartPoleEnv as cenv
from DQN import DQN
from Memory import Memory
import tensorflow as tf
import numpy as np


if __name__ == "__main__":
    state_size = 4
    action_size = 2
    learning_rate = 0.0003
    gamma = 0.95
    memory_size = 100000
    scene_iteration = 1000
    batch_size = 128
    test_batch_size = 10000

    env = cenv()
    with tf.Session() as sess:
        model = DQN(sess, state_size, action_size, learning_rate, gamma)
        memory = Memory(memory_size)

        while True:
            state = env.reset()
            for t in range(scene_iteration):
                env.render()

                # Play a new action
                action = model.select_action(state)
                new_state, reward, done, info = env.step(np.argmax(action))
                # if done:
                #     reward = -1.0

                memory.add((state, new_state, reward, action, done))

                # After batch_size experiences train the model
                if len(memory.buffer) > batch_size:
                    state_batch, new_state_batch, reward_batch, action_batch, done_batch = memory.sample(batch_size)
                    model.learn(state_batch, new_state_batch, reward_batch, action_batch, done_batch)
                state = new_state

                # Test phase
                if len(memory.buffer) > test_batch_size:
                    state_batch, new_state_batch, reward_batch, action_batch, done_batch = memory.sample(test_batch_size)
                    current_Q = np.max(sess.run(model.output, feed_dict={model.inputs: new_state_batch,
                                                                             model.temperature: model.temperature_c}), axis=1)
                    current_Q = reward_batch + model.gamma * current_Q
                    cost_value = sess.run([model.loss], {model.inputs: state_batch,
                                                         model.target_Q: current_Q,
                                                         model.actions: action_batch,
                                                         model.temperature: model.temperature_c})

                    # After test_batch_size experiences plot the cost value
                    # print(cost_value)

                # Scene end
                if done:
                    print("Episode finished after {} timesteps".format(t + 1))
                    break
