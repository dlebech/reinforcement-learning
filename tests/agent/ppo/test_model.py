import numpy as np

from rl.agent.ppo import model


def test_build_model(cartpole_env):
    m = model.build_model(cartpole_env)

    pol, val = m.predict(np.array([cartpole_env.reset()]))

    assert pol[0].shape == (2,)
    assert val[0].shape == (1,)
