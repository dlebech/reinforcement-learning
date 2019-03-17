import math
import random
import time
import statistics
import sys
from collections import defaultdict, deque

import gym
import numpy as np
import tensorflow as tf

from rl import constants
from rl.agent import base


def init_model(env):
    inp = tf.keras.layers.Input(shape=env.observation_space.shape)
    x = tf.keras.layers.Dense(16, activation="relu")(inp)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    x = tf.keras.layers.Dense(env.action_space.n, activation="linear")(x)
    model = tf.keras.models.Model(inputs=[inp], outputs=[x])
    model.compile(loss="mse", optimizer="adam")
    return model


def create_batch(arr):
    return np.array([arr])


class DQNAgent(base.Agent):
    def __init__(
        self, env_name, max_episodes=constants.DEFAULT_MAX_EPISODES, save_dir=None
    ):
        super().__init__(env_name, "dqn", save_dir=save_dir)
        self.model = init_model(self.env)
        self.max_episodes = max_episodes
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01
        self.memory = deque(maxlen=10000)

    def update_epsilon(self):
        if self.epsilon > self.min_epsilon and len(self.memory) > 500:
            self.epsilon *= self.epsilon_decay

    def memorize(self, state, action, reward, new_state, done, current_score):
        # Save to memory, except if we're done but reached the episode threshold
        # in which case we actually solved the game!
        if done and current_score < self.env.spec.reward_threshold:
            self.memory.append((state, action, reward, new_state, done))
        elif not done:
            self.memory.append((state, action, reward, new_state, done))

    def act(self, state, deterministic=False):
        return np.argmax(self.model.predict(create_batch(state))[0])

    def replay(self):
        # Replay from memory
        print("Replaying...")
        batch = random.sample(self.memory, min(64, len(self.memory)))
        for old_state, action, reward, new_state, done in batch:
            prediction = self.model.predict(create_batch(old_state))[0]

            # Maximize future reward... well... just predict for the new state.
            # Calculate the target from this and add the target reward to the
            # action.
            target = (
                reward
                if done
                else reward
                + self.gamma * np.max(self.model.predict(create_batch(new_state)))
            )

            prediction[action] = target
            x = create_batch(old_state)
            y = create_batch(prediction)
            self.model.fit(x, y, epochs=1, verbose=0)

    def train(self):
        explorations = deque(maxlen=100)
        exploitations = deque(maxlen=100)
        scores = deque(maxlen=100)

        best_score = -np.inf

        # Play some games
        for episode in range(self.max_episodes):
            state = self.env.reset()
            score = 0
            exploration = 0
            exploitation = 0

            while True:
                # Exploration: Use epsilon decay
                # Exploitation: Use the model output.
                if np.random.random() <= self.epsilon:
                    exploration += 1
                    action = self.env.action_space.sample()
                else:
                    exploitation += 1
                    # Find the model's proposed action.
                    action = self.act(state)

                # Perform the action, and if we're done, exit early.
                new_state, reward, done, _ = self.env.step(action)

                # Update score, memorize and update state.
                score += reward
                self.memorize(state, action, reward, new_state, done, score)
                state = new_state

                if done:
                    print(f"Game {episode + 1} is done, score: {score}")
                    break

            if score >= best_score:
                print('Saving new best model!')
                self.save_model(self.model)
                best_score = score

            scores.append(score)
            explorations.append(exploration)
            exploitations.append(exploitation)
            hundred_score = statistics.mean(scores)
            if hundred_score >= self.env.spec.reward_threshold:
                print("Solved!!")
                break
            print(f"Mean score last hundred episodes: {hundred_score}")
            print(f"Min/max score last hundred episodes: {min(scores)}/{max(scores)}")
            print(
                f"Mean exploration last hundred episodes: {statistics.mean(explorations)}"
            )
            print(
                f"Mean exploitations last hundred episodes: {statistics.mean(exploitations)}"
            )

            self.update_epsilon()
            self.replay()

        self.save_model(self.model)