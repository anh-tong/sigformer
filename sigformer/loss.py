from math import ceil

import jax
import jax.numpy as jnp


class BaseCriterion:
    def __call__(self, input, target):
        pass


class EntropyRiskMeasure(BaseCriterion):

    a: float

    def __init__(self, a=1.0) -> None:
        if a <= 0:
            raise ValueError("Risk aversion coefficient must be positive")
        self.a = a

    def __call__(self, input, target):
        diff = input - target

        result = jax.nn.logsumexp(-diff * self.a, axis=0) - jnp.log(
            input.shape[0] * 1.0
        )
        result = result / self.a

        return result


class ExpectedShortFall(BaseCriterion):

    p: float

    def __init__(self, p) -> None:
        self.p = p

    def __call__(self, input, target):
        diff = input - target
        values, _ = jax.lax.top_k(input, k=ceil(self.p * diff.shape[0]))
        return jnp.mean(values)


class CVar(BaseCriterion):

    fraction: float

    def __init__(self, fraction=0.5):
        self.fraction = fraction

    def __call__(self, input, target):
        gains = input - target
        sorted_gains = jax.lax.sort(gains, dimension=0)
        n_reduced = int(gains.shape[0] * self.fraction)
        reduced_gains = sorted_gains[:n_reduced]
        return jnp.mean(reduced_gains)


class MeanVariance(BaseCriterion):

    risk_aversion: float

    def __init__(self, risk_aversion):
        self.risk_aversion = risk_aversion

    def __call__(self, input, target):
        gains = input - target
        return jnp.mean(gains) - 0.5 * self.risk_aversion * jnp.var(gains)


class QuadraticLoss(BaseCriterion):

    p0: float

    def __init__(self, p0) -> None:
        super().__init__()
        self.p0 = p0

    def __call__(self, input, target):
        return jnp.mean(jnp.square(input - target + self.p0))


class CustomLoss(BaseCriterion):
    def __init__(self, a=1.0, p0=0.0) -> None:
        self.quadratic = QuadraticLoss(p0)
        self.entropy_risk = EntropyRiskMeasureWithP0(a, p0)

    def __call__(self, input, target):
        return self.quadratic(input, target) + self.entropy_risk(input, target)


class EntropyRiskMeasureWithP0(BaseCriterion):

    a: float
    p0: float

    def __init__(self, a=1.0, p0=0.0) -> None:
        if a <= 0:
            raise ValueError("Risk aversion coefficient must be positive")
        self.a = a
        self.p0 = p0

    def __call__(self, input, target):
        diff = input - target + self.p0
        result = jax.nn.logsumexp(-diff * self.a, axis=0) - jnp.log(
            input.shape[0] * 1.0
        )
        result = result / self.a

        return result


class EntropicLoss(BaseCriterion):

    a: float
    p0: float

    def __init__(self, a=1.0, p0=0.0) -> None:
        if a <= 0:
            raise ValueError("Risk aversion coefficient must be positive")
        self.a = a
        self.p0 = p0

    def __call__(self, input, target):
        diff = input - target + self.p0

        result = jnp.mean(jnp.exp(-self.a * diff))

        return result


class OCE(BaseCriterion):

    a: float
    p0: float

    def __init__(self, a=1.0, p0=0.0) -> None:
        if a <= 0:
            raise ValueError("Risk aversion coefficient must be positive")
        self.a = a
        self.p0 = p0
        self.ultility = EntropyRiskMeasure(a=a)

    def __call__(self, input, target):
        diff = input - target + self.p0
        result = self.p0 - jnp.mean(1 - jnp.exp(-diff))

        return result
