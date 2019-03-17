# Used for quick testing of an environment
import time

import gym

#env = gym.make('CartPole-v0').unwrapped
env = gym.make('Pendulum-v0').unwrapped

max_steps = 500

state = env.reset()
step = 0
done = False
while not done and step < max_steps:
    env.render()
    #state, reward, done, _ = env.step(0)
    state, reward, done, _ = env.step([2])
    print(reward, done)
    #time.sleep(0.1)

input('Press enter')
