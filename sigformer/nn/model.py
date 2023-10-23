from collections import namedtuple
from typing import List

import equinox as eqx
import equinox.nn as nn
import jax
import jax.random as jrandom
from jax.random import PRNGKey
from jaxtyping import Array, Float

from .layer import (
    LeadLagSignature,
    Signature,
    TensorAdd,
    TensorDropout,
    TensorFlatten,
    TensorLayerNorm,
    TensorMLP,
    TensorSelfAttention,
)
from .utils import split_key


Config = namedtuple(
    "Config",
    [
        "in_dim",
        "out_dim",
        "dim",
        "num_heads",
        "d_ff",
        "dropout",
        "n_attn_blocks",
        "order",
    ],
)


class Block(eqx.Module):

    attn_block: TensorSelfAttention
    attn_norm: TensorLayerNorm

    mlp_block: TensorMLP
    mlp_norm: TensorLayerNorm

    dropout: TensorDropout

    add_fn: TensorAdd

    def __init__(self, config: Config, *, key: PRNGKey) -> None:
        attn_key, mlp_key = jrandom.split(key)

        dim = config.dim
        order = config.order
        dropout = config.dropout
        n_heads = config.num_heads
        self.attn_block = TensorSelfAttention(
            order=order, dim=dim, dropout=dropout, n_heads=n_heads, key=attn_key
        )
        self.attn_norm = TensorLayerNorm(dim, order)

        self.mlp_block = TensorMLP(
            dim=dim, order=order, d_ff=dim * dim * 4, key=mlp_key
        )
        self.mlp_norm = TensorLayerNorm(dim, order)
        self.dropout = TensorDropout(dropout)
        self.add_fn = TensorAdd()

    def __call__(
        self,
        x: List[Array],
        *,
        key: PRNGKey = None,
    ) -> List[Array]:

        # attention
        resid = x
        x = self.attn_block(x, key=key)
        key = split_key(key)
        x = self.add_fn(resid, self.dropout(x, key=key))
        x = self.attn_norm(x)

        # MLP
        resid = x
        x = jax.vmap(self.mlp_block)(x)
        key = split_key(key)
        x = self.add_fn(resid, self.dropout(x, key=key))
        x = self.mlp_norm(x)

        return x


class SigFormer(eqx.Module):

    project: nn.Linear
    signature: Signature
    blocks: List[Block]
    readout: nn.Linear
    flatten: TensorFlatten

    def __init__(self, config: Config, *, key: PRNGKey):
        block_key, proj_key, readout_key = jrandom.split(key, 3)
        in_dim = config.in_dim
        out_dim = config.out_dim
        dim = config.dim

        self.project = nn.Linear(in_dim, dim, key=proj_key)
        self.signature = Signature(depth=config.order)
        blocks = []
        for i in range(config.n_attn_blocks):
            block = Block(config, key=jrandom.fold_in(block_key, i))
            blocks.append(block)
        self.blocks = blocks
        self.flatten = TensorFlatten()
        readout_in_dim = sum(config.dim ** (i + 1) for i in range(config.order))
        self.readout = nn.Linear(readout_in_dim, out_dim, key=readout_key)

    def __call__(
        self, x: Float[Array, "seq_len in_dim"], *, key: PRNGKey
    ) -> Float[Array, "seq_len out_dim"]:

        x = jax.vmap(self.project)(x)

        # compute signature
        x = self.signature(x)

        for block in self.blocks:
            key = split_key(key)
            x = block(x, key=key)

        x = self.flatten(x)

        x = jax.vmap(self.readout)(x)

        return x


class SigFormer_v2(eqx.Module):

    project: nn.Linear
    signature: LeadLagSignature
    blocks: List[Block]
    readout: nn.Linear
    flatten: TensorFlatten

    def __init__(self, config: Config, *, key: PRNGKey):
        block_key, proj_key, readout_key = jrandom.split(key, 3)
        out_dim = config.out_dim
        dim = config.in_dim

        # let's not use projection
        self.project = nn.Identity()
        self.signature = LeadLagSignature(depth=config.order, patch_len=5)

        # double the dimension because of lead-lage transform
        config = config._replace(dim=dim * 2)
        blocks = []
        for i in range(config.n_attn_blocks):
            block = Block(config, key=jrandom.fold_in(block_key, i))
            blocks.append(block)
        self.blocks = blocks
        self.flatten = TensorFlatten()
        readout_in_dim = sum(config.dim ** (i + 1) for i in range(config.order))
        self.readout = nn.Linear(readout_in_dim, out_dim, key=readout_key)

    def __call__(
        self, x: Float[Array, "seq_len in_dim"], *, key: PRNGKey
    ) -> Float[Array, "seq_len out_dim"]:

        x = jax.vmap(self.project)(x)

        # compute signature
        x = self.signature(x)

        for block in self.blocks:
            key = split_key(key)
            x = block(x, key=key)

        x = self.flatten(x)

        x = jax.vmap(self.readout)(x)

        return x


class VallinaTransformer(eqx.Module):

    project: nn.Linear
    blocks: List[Block]
    readout: nn.Linear
    flatten: TensorFlatten

    def __init__(self, config: Config, *, key: PRNGKey):
        block_key, proj_key, readout_key = jrandom.split(key, 3)
        in_dim = config.in_dim
        out_dim = config.out_dim
        dim = config.dim
        config = config._replace(order=1)
        assert config.order == 1

        self.project = nn.Linear(in_dim, dim, key=proj_key)
        blocks = []
        for i in range(config.n_attn_blocks):
            block = Block(config, key=jrandom.fold_in(block_key, i))
            blocks.append(block)
        self.blocks = blocks
        self.flatten = TensorFlatten()
        readout_in_dim = sum(config.dim ** (i + 1) for i in range(config.order))
        self.readout = nn.Linear(readout_in_dim, out_dim, key=readout_key)

    def __call__(
        self, x: Float[Array, "seq_len in_dim"], *, key: PRNGKey
    ) -> Float[Array, "seq_len out_dim"]:

        x = jax.vmap(self.project)(x)

        # wrapping as list to match
        x = [x]

        for block in self.blocks:
            key = split_key(key)
            x = block(x, key=key)

        x = self.flatten(x)

        x = jax.vmap(self.readout)(x)

        return x
