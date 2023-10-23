from abc import ABC, abstractmethod

from jax.random import PRNGKey


class BaseInstrument(ABC):

    cost: float

    @abstractmethod
    def simulate(self, rng_key: PRNGKey, time_horizon: float, **kwargs):
        """Simulate time series"""
