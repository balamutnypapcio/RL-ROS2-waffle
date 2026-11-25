"""Give a trained model (without memory) the visit-map inputs, keeping everything it has learned.

The visit map adds visit_map x visit_map inputs at the END of the observation (43 -> 68 for 5x5).
The first layer of the policy network and of the value network gets one new weight column per
new input, all set to 0, every other weight is copied. So the expanded model drives EXACTLY like
the old one, until training changes the new weights (they get normal gradients, because the
visit-map inputs are not 0). The step counter is copied too; the optimizer (Adam) starts fresh.

Usage (tb3-train --init-model ... --visit-map 5 does this by itself):
    tb3-expand-model runs/world2_s0/best_model.zip runs/world2_s0/best_model_visit5.zip --visit-map 5
"""
import argparse
from pathlib import Path

import torch
from stable_baselines3 import PPO

from tb3_nav.nav_env import NavEnv, visit_map_size


def expand_model(old_path: Path, visit_map: int) -> PPO:
    """Return a new PPO model with the visit-map inputs and the weights of the model in old_path."""
    old = PPO.load(old_path, device="cpu")
    n_old = old.observation_space.shape[0]
    if visit_map_size(n_old) != 0:
        raise ValueError(f"{old_path} already has a visit map ({n_old} inputs)")
    env = NavEnv(visit_map=visit_map)
    n_new = env.observation_space.shape[0]
    if n_old != n_new - visit_map * visit_map:
        raise ValueError(f"{old_path} has {n_old} inputs, NavEnv without visit map has "
                         f"{n_new - visit_map * visit_map}: can't expand this (older) model")

    # Same PPO settings as the old model (they are stored in its file)
    new = PPO("MlpPolicy", env, device="cpu", policy_kwargs=old.policy_kwargs,
              learning_rate=old.learning_rate, n_steps=old.n_steps, batch_size=old.batch_size,
              n_epochs=old.n_epochs, gamma=old.gamma, gae_lambda=old.gae_lambda,
              clip_range=float(old.clip_range(1.0)), ent_coef=old.ent_coef, vf_coef=old.vf_coef,
              max_grad_norm=old.max_grad_norm)

    old_weights = old.policy.state_dict()
    new_weights = new.policy.state_dict()
    for key, w in new_weights.items():
        o = old_weights[key]
        if o.shape == w.shape:
            new_weights[key] = o.clone()
        elif w.dim() == 2 and o.shape == (w.shape[0], n_old) and w.shape[1] == n_new:
            # first layer: old columns copied, new input columns = 0
            expanded = torch.zeros_like(w)
            expanded[:, :n_old] = o
            new_weights[key] = expanded
        else:
            raise ValueError(f"unexpected weight shape {key}: old {tuple(o.shape)}, new {tuple(w.shape)}")
    new.policy.load_state_dict(new_weights)
    new.num_timesteps = old.num_timesteps
    env.close()
    return new


def main() -> None:
    parser = argparse.ArgumentParser(description="Add visit-map inputs (memory) to a trained model.")
    parser.add_argument("model", type=Path, help="model .zip without visit map")
    parser.add_argument("out", type=Path, help="where to save the expanded model .zip")
    parser.add_argument("--visit-map", type=int, default=5, help="visit-map window side, odd (default: 5)")
    args = parser.parse_args()
    model = expand_model(args.model, args.visit_map)
    model.save(args.out)
    print(f"Saved: {args.out} ({model.observation_space.shape[0]} inputs, {model.num_timesteps} steps)")


if __name__ == "__main__":
    main()
