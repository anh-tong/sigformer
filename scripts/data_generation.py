"""Generate training data for deep calibration model"""
import sys
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import pandas as pd
from jaxtyping import Array, Float
from sigformer.stochastic_process.rbergomi import generate_rough_bergomi
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from tqdm import tqdm


def bisect(fn, target, lower, upper, precision=1e-6, max_iter=100_000):

    lower = jnp.ones_like(target) * lower
    upper = jnp.ones_like(target) * upper

    if not jnp.all(lower < upper):
        raise ValueError("`lower` should be smaller than `upper`")

    if jnp.all(fn(lower) > fn(upper)):
        fn = lambda x: -fn(x)
        return bisect(fn, -target, lower, upper, precision, max_iter)

    def cond_fn(val):
        lower, upper, n_iter = val
        tol_cond = jnp.any(upper - lower > precision)
        return jnp.logical_and(tol_cond, n_iter <= max_iter)

    def body_fn(val):
        lower, upper, n_iter = val

        n_iter = n_iter + 1
        m = 0.5 * (lower + upper)
        output = fn(m)
        lower = jnp.where(output >= target, lower, m)
        upper = jnp.where(output < target, upper, m)

        return lower, upper, n_iter

    _, upper, _ = jax.lax.while_loop(
        cond_fun=cond_fn, body_fun=body_fn, init_val=(lower, upper, 0)
    )

    return upper


def test_bisect():
    def f(x):
        return 2 * x

    target = jnp.ones((2,))

    print(bisect(f, target, jnp.ones((2,)) * 0, jnp.ones((2,)) * 4))


def ncdf(input):
    return jax.scipy.stats.norm.cdf(input)


def d1(log_moneyness, time_to_maturity, volatility):

    """
    .. math::
        \\frac{s}{\\sigma \\sqrt(t)} +  \\frac{\\sigma \\sqrt{t}}{2}

    """

    variance = volatility * jnp.sqrt(time_to_maturity)
    output = log_moneyness / variance + variance / 2

    return output


def d2(log_moneyness, time_to_maturity, volatility):
    variance = volatility * jnp.sqrt(time_to_maturity)
    output = log_moneyness / variance - variance / 2

    return output


def blackscholes_european_price(
    log_moneyness,
    time_to_maturity,
    volatility,
    strike=1.0,
    call=True,
):

    spot = jnp.exp(log_moneyness) * strike
    d1_value = d1(log_moneyness, time_to_maturity, volatility)
    d2_value = d2(log_moneyness, time_to_maturity, volatility)
    price = spot * ncdf(d1_value) - strike * ncdf(d2_value)
    price = price + strike * (1 - jnp.exp(log_moneyness)) if not call else price

    return price


def test_bs_european_price():

    log_moneyness = jnp.ones((3,))
    time_to_maturity = jnp.ones((3,))
    volatility = jnp.ones((3,)) * 0.2

    def price_call(log_moneyness, time_to_maturity, volatility):
        return blackscholes_european_price(log_moneyness, time_to_maturity, volatility)

    print(jax.vmap(price_call)(log_moneyness, volatility, time_to_maturity))


def blackscholes_implied_volatility(
    log_moneyness,
    time_to_maturity,
    strike,
    price,
):
    def fn(volatility):
        return blackscholes_european_price(
            log_moneyness=log_moneyness,
            time_to_maturity=time_to_maturity,
            strike=strike,
            volatility=volatility,
        )

    implied_vol = bisect(
        fn,
        target=price,
        lower=1e-3,
        upper=100.0,
    )

    return implied_vol


def test_blacksholes_implied_volatility():

    log_moneyness = jnp.ones((3,))
    time_to_maturity = jnp.ones((3,))

    price = jnp.ones((3,)) * 0.1

    print(blackscholes_implied_volatility(log_moneyness, time_to_maturity, price))


def rbergomi_pricer(
    hurst: Float[Array, " dim"],
    rho: Float[Array, " dim"],
    xi: Float[Array, " dim"],
    eta: Float[Array, " dim"],
    S0: Float[Array, " dim"],
    K: Float[Array, " dim"],
    maturity: Float[Array, " dim"],
    dt=1.0 / 365,
    n_samples=10_000,
    *,
    key: jrandom.PRNGKey,
):
    dim = hurst.shape[0]

    # consider to sample
    n_steps = np.ceil(jnp.max(maturity) / dt).astype(np.int16)
    max_T_fn = partial(generate_rough_bergomi, n_steps=n_steps, dt=dt)
    vmapped_dim_fn = partial(
        jax.vmap(max_T_fn), hurst=hurst, rho=rho, xi=xi, eta=eta, s0=S0
    )

    def simulate_fn(key):
        key = jrandom.split(key, dim)
        return vmapped_dim_fn(rng_key=key)

    # actual simulation
    key = jrandom.split(key, n_samples)
    output = jax.jit(jax.vmap(simulate_fn))(key)[0]

    # need to select index out of
    index = jnp.ceil(maturity / dt).astype(jnp.int16)
    output = output[..., index]
    S_T = jax.vmap(jnp.diagonal)(output)

    # return European call option
    return jnp.mean(jax.nn.relu(S_T - K[None, :]), axis=0)


def test_rbergomi_pricer():

    d = 3
    one = jnp.ones((d,))
    s0 = 1.0 * one
    K = 1.0 * one
    rho = -0.7 * one
    hurst = 0.1 * one
    eta = 1.8 * one
    xi = 0.245**2 * one
    maturity = 0.5 * one

    output = rbergomi_pricer(
        hurst=hurst,
        rho=rho,
        xi=xi,
        eta=eta,
        S0=s0,
        K=K,
        maturity=maturity,
        key=jrandom.PRNGKey(0),
    )

    print(output.shape)


def test():
    import matplotlib.pyplot as plt

    ones = jnp.ones((2,))
    for rho in jnp.asarray([-0.7, 0, 0.7]):
        hurst = jnp.asarray(0.1) * ones
        xi = jnp.asarray(0.235**2) * ones
        eta = jnp.asarray(1.8) * ones
        maturity = jnp.asarray(20.0 / 250) * ones
        strike = jnp.asarray(1.0) * ones
        key = jax.random.PRNGKey(0)

        iv = []
        for s in jnp.linspace(-0.1, 0.1, 20):
            price = rbergomi_pricer(
                hurst=hurst,
                xi=xi,
                rho=rho * ones,
                eta=eta,
                S0=jnp.exp(s) * strike,
                K=strike,
                maturity=maturity,
                dt=1.0 / 250,
                key=key,
            )

            implied_vol = blackscholes_implied_volatility(
                log_moneyness=s * ones,
                time_to_maturity=maturity,
                strike=strike,
                price=price,
            )

            iv.append(implied_vol[0])

        plt.plot(jnp.linspace(-0.1, 0.1, 20), iv)

    plt.savefig("dummy.png")


def truncated_normal(key, loc, scale, lower, upper):

    new_lower = (lower - loc) / scale
    new_upper = (upper - loc) / scale

    x = jrandom.truncated_normal(
        key,
        upper=new_upper,
        lower=new_lower,
    )

    return x * scale + loc


def spx_parse(input_file="./data/SPX_20190409.csv"):

    spot_price = 2878.99
    today = pd.to_datetime("20190409", format="%Y%m%d", errors="coerce")
    nb_trading_days = 365

    raw_spx = pd.read_csv(input_file, skiprows=2)
    raw_spx = raw_spx[raw_spx["Calls"].apply(lambda s: s.startswith("SPXW"))]
    raw_spx["Expiration Date"] = pd.to_datetime(
        raw_spx["Expiration Date"], format="%m/%d/%Y"
    )
    raw_spx["Time to Maturity (years)"] = raw_spx["Expiration Date"].apply(
        lambda exp_date: (exp_date - today).days / nb_trading_days
    )
    raw_spx["Log Moneyness"] = np.log(spot_price / raw_spx["Strike"])

    call_data = raw_spx.loc[
        :,
        [
            "Bid",
            "Ask",
            "IV",
            "Open Int",
            "Time to Maturity (years)",
            "Log Moneyness",
            "Strike",
        ],
    ]
    put_data = raw_spx.loc[
        :,
        [
            "Bid.1",
            "Ask.1",
            "IV.1",
            "Open Int.1",
            "Time to Maturity (years)",
            "Log Moneyness",
            "Strike",
        ],
    ]
    put_data.columns = call_data.columns

    call_data["Mid"] = (call_data["Bid"] + call_data["Ask"]) / 2
    put_data["Mid"] = (put_data["Bid"] + put_data["Ask"]) / 2

    call_data.to_csv("./data/processed_spx_calls_all_liquids.csv")
    put_data.to_csv("./data/processed_spx_puts_all_liquids.csv")

    data = pd.read_csv("./data/processed_spx_calls_all_liquids.csv")

    total_interest = data["Open Int"].sum()
    interests = pd.DataFrame(
        index=np.arange(total_interest),
        columns=["Time to Maturity (years)", "Log Moneyness"],
        dtype=float,
    )

    counter = 0
    for idx in data.index:
        num_int = data.loc[idx, "Open Int"]
        end_counter = counter + num_int
        interests.iloc[counter:end_counter, :] = data.loc[
            idx, ["Time to Maturity (years)", "Log Moneyness"]
        ].values
        counter = end_counter

    K_T = data[["Time to Maturity (years)", "Log Moneyness"]].values
    K_T[:, 1] = np.exp(K_T[:, 1])
    params = {"bandwidth": np.logspace(-3, -1, 5)}
    grid = GridSearchCV(KernelDensity(), params, verbose=sys.maxsize, n_jobs=-1, cv=5)
    grid.fit(K_T)

    kde = grid.best_estimator_
    kde.fit(K_T)

    n_samples = int(1e6)

    generated_K_T = pd.DataFrame(
        index=np.arange(n_samples),
        columns=["Moneyness", "Time to Maturity (years)"],
        dtype=float,
    )

    counter = 0

    while counter < n_samples:
        still_need = n_samples - counter
        samples = kde.sample(still_need)
        is_valid = (
            (samples[:, 0] > 0.75)
            & (samples[:, 0] < 1.2)
            & (samples[:, 1] > 0)
            & (samples[:, 1] < 0.25)
        )
        valid_samples = samples[is_valid]
        n_valid = len(valid_samples)
        new_counter = counter + n_valid
        generated_K_T.iloc[counter:new_counter, :] = valid_samples
        counter = new_counter

    generated_K_T.to_csv("./data/strike_maturity.csv")


def main(n_samples=100_000, seed=123, batch_size=100, dt=1.0 / 365):

    key = jrandom.PRNGKey(seed)

    rho_key, eta_key, H_key, xi_key = jax.random.split(key, 4)

    eta_gen_fn = partial(truncated_normal, loc=2.5, scale=0.5, lower=1.0, upper=4.0)
    rho_gen_fn = partial(truncated_normal, loc=-0.95, scale=0.2, lower=-1.0, upper=-0.5)
    H_gen_fn = partial(truncated_normal, loc=0.07, scale=0.05, lower=0.01, upper=0.5)
    xi_sqrt_gen_fn = partial(truncated_normal, loc=0.3, scale=0.1, lower=0.05, upper=1)

    eta_priors = jax.vmap(eta_gen_fn)(jax.random.split(eta_key, n_samples))
    rho_priors = jax.vmap(rho_gen_fn)(jax.random.split(rho_key, n_samples))
    H_priors = jax.vmap(H_gen_fn)(jax.random.split(H_key, n_samples))
    xi_sqrt_priors = jax.vmap(xi_sqrt_gen_fn)(jax.random.split(xi_key, n_samples))
    xi_priors = xi_sqrt_priors**2

    df = pd.read_csv("./data/strike_maturity.csv")
    maturity_priors = jnp.asarray(df["Time to Maturity (years)"].to_numpy())[:n_samples]
    moneyness_priors = jnp.asarray(df["Moneyness"].to_numpy())[:n_samples]

    n_batchs = n_samples // batch_size

    prices = jnp.zeros((n_samples,))
    implied_vols = jnp.zeros((n_samples,))

    for ith_batch in tqdm(range(n_batchs), desc="Sampling"):

        selected = slice(ith_batch * batch_size, (ith_batch + 1) * batch_size)
        hurst = H_priors[selected]
        rho = rho_priors[selected]
        xi = xi_priors[selected]
        eta = eta_priors[selected]
        moneyness = moneyness_priors[selected]  # K / S
        maturity = maturity_priors[selected]
        S0 = jnp.ones_like(maturity)
        K = S0 * moneyness

        price = rbergomi_pricer(
            hurst=hurst,
            rho=rho,
            xi=xi,
            eta=eta,
            S0=S0,
            K=K,
            maturity=maturity,
            dt=dt,
            key=jrandom.fold_in(key, ith_batch),
            n_samples=40_000,
        )

        implied_vol = blackscholes_implied_volatility(
            log_moneyness=jnp.log(1.0 / moneyness),  # here moneyness = S / K
            time_to_maturity=maturity,
            strike=K,
            price=price,
        )

        implied_vol = jnp.where(
            jnp.logical_or(price <= 0, price + K < S0),
            jnp.nan,
            implied_vol,
        )
        prices = prices.at[selected].set(price)
        implied_vols = implied_vols.at[selected].set(implied_vol)

    data_nn = pd.DataFrame.from_dict(
        {
            "hurst": np.array(H_priors),
            "rho": np.array(rho_priors),
            "eta": np.array(eta_priors),
            "xi": np.array(xi_priors),
            "moneyness": np.array(moneyness_priors),
            "maturity": np.array(maturity_priors),
            "implied_vol": np.array(implied_vols),
        }
    )
    data_nn.to_csv(f"./data/rbergomi_data_{n_samples}.csv")


if __name__ == "__main__":
    main()
