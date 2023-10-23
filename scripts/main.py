import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import matplotlib.pyplot as plt
import numpy as np
import optax
import wandb
from fire import Fire
from plot_utils import plot_histogram
from sigformer.hedger import (
    DeepHedger,
    RBergomiDeltaHedger,
    SigHedger,
    SigHedger_v2,
    SignatureOnlyHedger,
    TransformerHedger,
)
from sigformer.instruments.derivative import EuropeanOption, VarianceSwap
from sigformer.instruments.primary import RoughBergomiStock
from sigformer.loss import (  # noqa
    CustomLoss,
    EntropyRiskMeasure,
    EntropyRiskMeasureWithP0,
    QuadraticLoss,
)
from sigformer.utils import CheckpointManager, conditional_value_at_risk
from tqdm import tqdm


class ForwardVariance(VarianceSwap):
    def __init__(
        self, underlier, maturity: float = 45.0 / 356, strike: float = 0.04
    ) -> None:
        super().__init__(underlier, maturity, strike)

    def price(self, input):
        forward_variance = input["forward_variance"]
        return forward_variance


def main(
    wandb_mode="online",
    # wandb_mode = "offline",
    # rough Bergomi price model
    hurst=0.1,
    rho=-0.7,
    eta=1.9,
    xi=0.235**2,
    cost=0.0,
    dt=1.0 / 356,
    forward_offset=15.0 / 356,
    # derivative
    T=30,  # maturity
    S0=100.0,  # initial price
    K=100.0,  # strike
    # SigHedger
    model_dim=3,
    n_attn_heads=12,
    n_attn_blocks=5,
    signature_depth=3,
    inputs=["LogMoneyness", "Volatility"],
    loss_type="quadratic",
    model_name="SigFormer",  # either SigFormer, RNN, SignatureOnly, VanillaTransformer
    # DeepHedger
    recur_type="Recur",  # either "Recur", "SemiRecur", "NoRecur". Default is None indicates using `SigHedger` instead
    # traing
    n_train_paths=1_000,
    n_valid_paths=10_000,
    n_epochs=1000,
    valid_freq=20,
    lr=1e-4,
    n_test_paths=10_000,
    seed=123,
):
    T = T * dt
    # initialize wandb
    wandb.init(
        project="deep-hedge",
        config={
            "hurst": hurst,
            "rho": rho,
            "eta": eta,
            "xi": xi,
            "dt": dt,
            "forward_offset": forward_offset,
            "T": T,
            "S0": S0,
            "K": K,
            "model_name": model_name,
            "model_dim": model_dim,
            "n_attn_heads": n_attn_heads,
            "n_attn_blocks": n_attn_blocks,
            "signature_depth": signature_depth,
            "recur_type": recur_type,
            "seed": seed,
        },
        name=f"rbergomi_H_{hurst}_T_{int(T/dt)}_{model_name}",
        mode=wandb_mode,
    )

    key = jrandom.PRNGKey(seed)
    model_key, train_key, valid_key, test_key, p0_key = jrandom.split(key, 5)

    stock = RoughBergomiStock(
        hurst=hurst,
        rho=rho,
        xi=xi,
        cost=cost,
        dt=dt,
        forward_offset=forward_offset,
    )
    init_state = (S0, stock.xi, None)
    derivative = EuropeanOption(
        underlier=stock,
        call=True,
        strike=K,
        maturity=T,
    )

    forward_variance = ForwardVariance(stock, maturity=T)
    hedge = [derivative, forward_variance]

    # Perfect hedge
    delta_hedger = RBergomiDeltaHedger(derivative=derivative, n_paths=2000)
    delta_pl = delta_hedger.compute_pnl(
        rng_key=test_key,
        n_paths=n_test_paths,
        init_state=init_state,
    )
    # obtain p0 from expectation of payoff
    payoff_fn = lambda key: derivative.payoff(rng_key=key, init_state=init_state)
    payoff = jax.vmap(payoff_fn)(jrandom.split(p0_key, n_test_paths))
    p0 = jnp.mean(payoff)
    if loss_type == "entropy":
        loss_fn = CustomLoss(a=10, p0=p0)
    else:
        loss_fn = QuadraticLoss(p0=p0)
    print(f"p0: {p0.item():.3f}")

    if model_name == "SigFormer":
        # SigHedge
        model = SigHedger(
            derivative=derivative,
            inputs=inputs,
            hedge=hedge,
            criterion=loss_fn,
            model_dim=model_dim,
            n_attn_heads=n_attn_heads,
            n_attn_blocks=n_attn_blocks,
            signature_depth=signature_depth,
            rng_key=model_key,
        )
    elif model_name == "SigFormer_v2":
        model = SigHedger_v2(
            derivative=derivative,
            inputs=inputs,
            hedge=hedge,
            criterion=loss_fn,
            model_dim=model_dim,
            n_attn_heads=n_attn_heads,
            n_attn_blocks=n_attn_blocks,
            signature_depth=signature_depth,
            rng_key=model_key,
        )
    elif model_name == "RNN":
        # RNN, Semi-RNN, No Recurence
        model = DeepHedger(
            derivative=derivative,
            inputs=inputs,
            hedge=hedge,
            criterion=loss_fn,
            recur_type=recur_type,
            rng_key=model_key,
        )
    elif model_name == "VanillaTransformer":
        model = TransformerHedger(
            derivative=derivative,
            inputs=inputs,
            hedge=hedge,
            criterion=loss_fn,
            model_dim=model_dim,
            n_attn_heads=n_attn_heads,
            n_attn_blocks=n_attn_blocks,
            signature_depth=signature_depth,
            rng_key=model_key,
        )
    elif model_name == "SignatureOnly":
        model = SignatureOnlyHedger(
            derivative=derivative,
            inputs=inputs,
            hedge=hedge,
            criterion=loss_fn,
            model_dim=model_dim,
            n_attn_heads=n_attn_heads,
            n_attn_blocks=n_attn_blocks,
            signature_depth=signature_depth,
            rng_key=model_key,
        )
    else:
        raise ValueError(
            f"Invalid `model_name`={model_name}. "
            + " Please choose among: [SigFormer, RNN, Signatureonly, VanillaTransformer]"
        )

    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    chkpt_manager = CheckpointManager()
    chkpt_manager_05 = CheckpointManager(path="./checkpoint/best_cvar_05_model.eqx")
    chkpt_manager_099 = CheckpointManager(path="./checkpoint/best_cvar_099_model.eqx")

    @eqx.filter_value_and_grad
    def compute_loss(model: SigHedger, n_simulate_paths, rng_key):
        return model.compute_loss(
            init_state=init_state,
            n_paths=n_simulate_paths,
            rng_key=rng_key,
        )

    @eqx.filter_jit
    def make_step(model: SigHedger, opt_state, n_simulate_paths, rng_key):

        loss_value, grads = compute_loss(model, n_simulate_paths, rng_key)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    valid_keys = jrandom.split(valid_key, n_valid_paths)
    valid_data = jax.vmap(
        lambda key: derivative.simulate(rng_key=key, init_state=init_state)
    )(valid_keys)

    @eqx.filter_jit
    def validate(model: SigHedger, valid_data):
        portfolio, payoff = model.compute_pnl(
            rng_key=None,
            n_paths=n_valid_paths,
            simulated_data=valid_data,
            return_portfolio_and_payoff=True,
        )
        return model.criterion(portfolio, payoff)

    all_pl = []
    progress = tqdm(range(n_epochs), desc="Train")
    for i in progress:

        model, opt_state, loss_value = make_step(
            model,
            opt_state,
            n_simulate_paths=n_train_paths,
            rng_key=jrandom.fold_in(train_key, i),
        )

        wandb.log({"train/train_loss": loss_value.item()})

        if i % valid_freq == 0:
            valid_model = eqx.tree_inference(model, True)

            # plot historgram
            valid_loss_pl = valid_model.compute_pnl(
                rng_key=test_key, n_paths=n_test_paths, init_state=init_state
            )
            fig = plot_histogram(
                [valid_loss_pl, delta_pl], ["Hedger", "Delta Hedge"], bins=50
            )
            fig.savefig("dummy.png", dpi=200, bbox_inches="tight")
            plt.close(fig)

            # compute CVar-0.5 and CVar-0.90
            neutral_risk_adjusted_pnl = valid_loss_pl - jnp.mean(valid_loss_pl)
            cvar_05 = conditional_value_at_risk(neutral_risk_adjusted_pnl, 0.5)
            cvar_099 = conditional_value_at_risk(neutral_risk_adjusted_pnl, 0.99)

            # compute validate loss
            valid_loss = validate(valid_model, valid_data)
            wandb.log(
                {
                    "train/valid_loss": valid_loss.item(),
                    "train/cvar-0.5": cvar_05.item(),
                    "train/cvar-0.99": cvar_099.item(),
                    "test/histogram": wandb.Image("dummy.png"),
                }
            )
            chkpt_manager(valid_loss.item(), model)
            chkpt_manager_05(-cvar_05.item(), model)  # bigger cvar-0.5 is better
            chkpt_manager_099(-cvar_099.item(), model)  # bigger cvar-0.99 is better
            all_pl.append(np.array(valid_loss_pl))

    # best model according to valid loss
    model = chkpt_manager.load_check_point(model)
    model = eqx.tree_inference(model, True)

    valid_loss_pl = model.compute_pnl(
        rng_key=test_key, n_paths=n_test_paths, init_state=init_state
    )

    hist, bins = np.histogram(valid_loss_pl, bins=50, density=False)
    plt.figure(figsize=(10, 5))
    plt.bar(bins[:-1], hist, width=(bins[1] - bins[0]) * 0.8)

    fig = plot_histogram(
        [valid_loss_pl, delta_pl], ["Sig Hedger", "Delta Hedge"], bins=50
    )
    fig.savefig("dummy.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    wandb.log(
        {
            "final/valid": wandb.Image("dummy.png"),
        }
    )

    # best model according to cvar 0.5
    model = chkpt_manager_05.load_check_point(model)
    model = eqx.tree_inference(model, True)
    cvar_05_pl = model.compute_pnl(
        rng_key=test_key, n_paths=n_test_paths, init_state=init_state
    )
    hist, bins = np.histogram(cvar_05_pl, bins=50, density=False)
    plt.figure(figsize=(10, 5))
    plt.bar(bins[:-1], hist, width=(bins[1] - bins[0]) * 0.8)

    fig = plot_histogram([cvar_05_pl, delta_pl], ["Sig Hedger", "Delta Hedge"], bins=50)
    fig.savefig("dummy.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    wandb.log(
        {
            "final/cvar-0.5": wandb.Image("dummy.png"),
        }
    )

    # best model according to cvar 0.5
    model = eqx.tree_inference(model, True)
    model = chkpt_manager_099.load_check_point(model)
    cvar_099_pl = model.compute_pnl(
        rng_key=test_key, n_paths=n_test_paths, init_state=init_state
    )
    hist, bins = np.histogram(cvar_099_pl, bins=50, density=False)
    plt.figure(figsize=(10, 5))
    plt.bar(bins[:-1], hist, width=(bins[1] - bins[0]) * 0.8)

    fig = plot_histogram(
        [cvar_099_pl, delta_pl], ["Sig Hedger", "Delta Hedge"], bins=50
    )
    fig.savefig("dummy.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    wandb.log(
        {
            "final/cvar-0.99": wandb.Image("dummy.png"),
        }
    )

    all_sig_pl = np.stack(all_pl)
    delta_pl = np.array(delta_pl)
    valid_loss_pl = np.array(valid_loss_pl)
    cvar_05_pl = np.array(cvar_05_pl)
    cvar_099_pl = np.array(cvar_099_pl)
    pnl_file = "./checkpoint/pnl.npz"
    np.savez_compressed(
        pnl_file, delta_pl, valid_loss_pl, cvar_05_pl, cvar_099_pl, all_sig_pl
    )
    wandb.save(pnl_file)

    wandb.finish()


if __name__ == "__main__":
    Fire(main)
