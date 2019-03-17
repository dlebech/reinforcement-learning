from rl.agent.a3c import master


def test_train_discrete(save_dir):
    a3c_agent = master.A3CAgent(
        env_name="CartPole-v0",
        max_episodes=1,
        # This should ensure that the model gets updated during this test
        worker_update_frequency=5,
        save_dir=save_dir,
        thread_count=1,
    )

    # Just check that it doesn't fail and it executes
    a3c_agent.train()


def test_train_continuous(save_dir):
    a3c_agent = master.A3CAgent(
        env_name="Pendulum-v0",
        max_episodes=1,
        worker_update_frequency=5,
        save_dir=save_dir,
        thread_count=1,
    )

    # Just check that it doesn't fail and it executes
    a3c_agent.train()
