import os

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import wandb
from jaxtyping import Array, Float


def european_payoff(
    x: Float[Array, "n_paths n_steps"],
    call: bool = True,
    strike: float = 1.0,
) -> Float[Array, " n_paths"]:

    if call:
        return jax.nn.relu(x[..., -1] - strike)
    else:
        return jax.nn.relu(strike - x[..., -1])


def pl(
    spot: Float[Array, "n_steps dim"],
    unit: Float[Array, "n_steps dim"],
    payoff: Float = None,
):
    """

    Use `jax.vmap` when working with with multiple paths
    """

    if spot.shape != unit.shape:
        raise ValueError(
            f"`spot` and `unit` should have the same shape. `spot` shape: {spot.shape}, `unit` shape: {unit.shape}"
        )

    output = unit[:-1] * jnp.diff(spot, axis=0)
    output = jnp.sum(output)

    if payoff is not None:
        output = output - payoff

    return output


def realized_variance(x: Float[Array, " n_steps"], dt: float):
    """
    The realized variance of the price

        dt*log(S[i+1]/S[i])^2

    """
    diff_log = jnp.diff(jnp.log(x), axis=-1)
    return jnp.mean(jnp.square(diff_log), axis=-1) / dt


def finite_diff(fn: callable, args: list, args_num=None, epsilon=1e-6):

    if args_num is None:
        args_num = list(range(len(args)))

    f0 = fn(*args)

    ret = []
    for arg_position in args_num:
        old_value = args[arg_position]
        new_value = args[arg_position] + epsilon
        args[arg_position] = new_value
        f1 = fn(*args)
        ret.append((f1 - f0) / epsilon)
        args[arg_position] = old_value

    return ret


def conditional_value_at_risk(pnl, fraction=0.5):

    sorted_pnl = jax.lax.sort(pnl, dimension=0)
    n_reduced = int(sorted_pnl.shape[0] * fraction)
    reduced_pnl = sorted_pnl[:n_reduced]
    return jnp.mean(reduced_pnl)


class CheckpointManager:
    def __init__(self, path="./checkpoint/best_model.eqx", wandb_log=True) -> None:
        self.vall_loss_min = np.Inf
        self.wandb_log = wandb_log
        self.path = path

    def __call__(self, val_loss, model):

        if val_loss <= self.vall_loss_min:
            self.vall_loss_min = val_loss
            eqx.tree_serialise_leaves(path_or_file=self.path, pytree=model)

            if self.wandb_log:
                wandb.save(self.path)

    def load_check_point(self, model):
        if not os.path.exists(self.path):
            raise ValueError(f"There is no check point in {self.path}")

        return eqx.tree_deserialise_leaves(self.path, model)


if __name__ == "__main__":

    def f(x, y):
        return x**2 + y**2

    print(finite_diff(f, [jnp.asarray(1), jnp.asarray(2.0)]))
