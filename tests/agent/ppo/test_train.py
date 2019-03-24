from rl.agent.ppo import PPOAgent


def test_train():
    agent = PPOAgent("CartPole-v1")
    agent.max_episodes = 1
    agent.max_steps = 10
    agent.update_frequency = 5
    agent.train()
