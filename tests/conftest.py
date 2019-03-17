import logging
import os
import shutil
import tempfile

import gym
import pytest
import tensorflow as tf


tf.enable_eager_execution()


@pytest.fixture
def save_dir():
    d = tempfile.mkdtemp(prefix='rl_test_')
    yield d
    shutil.rmtree(d)


@pytest.fixture
def cartpole_env():
    return gym.make('CartPole-v0')


@pytest.fixture
def pendulum_env():
    return gym.make('Pendulum-v0')