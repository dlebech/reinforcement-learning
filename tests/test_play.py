from rl import play
from rl.agent.a3c import A3CAgent
from rl.agent.dqn import DQNAgent
from rl.agent.rnd import RandomAgent

discrete_env_name = "CartPole-v0"
continuous_env_name = "Pendulum-v0"


def test_play_discrete_random():
    agent = RandomAgent(discrete_env_name, 1)
    play.play(agent, discrete_env_name, max_steps=1, render=False)


def test_play_discrete_a3c():
    agent = A3CAgent(discrete_env_name)
    play.play(agent, discrete_env_name, max_steps=1, render=False)


def test_play_discrete_dqn():
    agent = DQNAgent(discrete_env_name)
    play.play(agent, discrete_env_name, max_steps=1, render=False)


def test_play_continuous_random():
    agent = RandomAgent(continuous_env_name, 1)
    play.play(agent, continuous_env_name, max_steps=1, render=False)


def test_play_continuous_a3c():
    agent = A3CAgent(continuous_env_name)
    play.play(agent, continuous_env_name, max_steps=1, render=False)
