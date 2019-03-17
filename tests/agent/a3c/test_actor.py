from rl.agent.a3c import model, actor


def test_rescale_mean(pendulum_env):
    a3c_model = model.A3CModel(pendulum_env)

    assert actor.rescale_mean(a3c_model, 0) == 0
    assert actor.rescale_mean(a3c_model, 0.5) == 1
    assert actor.rescale_mean(a3c_model, 1) == 2
    assert actor.rescale_mean(a3c_model, -1) == -2
    assert actor.rescale_mean(a3c_model, -0.5) == -1
