import numpy as np
import pytest
from stable_baselines3 import PPO

from tb3_nav.expand_model import expand_model
from tb3_nav.nav_env import NavEnv


@pytest.fixture(scope="module")
def small_model(tmp_path_factory):
    """A tiny untrained model without visit map, saved like a real one."""
    model = PPO("MlpPolicy", NavEnv(), device="cpu", seed=0, n_steps=64, batch_size=64,
                policy_kwargs=dict(net_arch=dict(pi=[32, 32], vf=[32, 32])))
    model.num_timesteps = 12345
    path = tmp_path_factory.mktemp("models") / "small.zip"
    model.save(path)
    return path


def test_expanded_model_acts_exactly_like_the_old_one(small_model):
    old = PPO.load(small_model, device="cpu")
    new = expand_model(small_model, visit_map=5)
    assert new.observation_space.shape == (68,)
    assert new.num_timesteps == 12345

    env = NavEnv(visit_map=5)
    obs, _ = env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(5):
        obs = obs.copy()
        obs[-25:] = rng.uniform(0, 1, 25)          # any visit map must not change anything yet
        a_old, _ = old.predict(obs[:43], deterministic=True)
        a_new, _ = new.predict(obs, deterministic=True)
        np.testing.assert_allclose(a_old, a_new, atol=1e-6)
        v_old = old.policy.predict_values(old.policy.obs_to_tensor(obs[:43])[0])
        v_new = new.policy.predict_values(new.policy.obs_to_tensor(obs)[0])
        np.testing.assert_allclose(v_old.detach().numpy(), v_new.detach().numpy(), atol=1e-5)
        obs, *_ = env.step(a_new)


def test_expanded_model_keeps_ppo_settings(small_model):
    new = expand_model(small_model, visit_map=3)
    assert new.n_steps == 64 and new.batch_size == 64
    assert new.policy_kwargs["net_arch"] == dict(pi=[32, 32], vf=[32, 32])


def test_model_with_visit_map_is_not_expanded_again(small_model, tmp_path):
    path = tmp_path / "expanded.zip"
    expand_model(small_model, visit_map=5).save(path)
    with pytest.raises(ValueError):
        expand_model(path, visit_map=5)
