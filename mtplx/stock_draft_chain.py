"""``MTPLX_STOCK_SAMPLED_DRAFT_CHAIN`` -- the sampled draft chain on the full draft row.

The problem
-----------
At ``temperature > 0`` the stock draft loop is serial: every depth runs one MTP
forward, blocks on one ``mx.eval`` of its top-k support, builds the proposal on
the host, draws the token with the request's NumPy generator, and only then
builds the next depth, because that forward consumes the drawn token.  The GPU
idles while the host samples and encodes each next depth.

``qwen4_draft_device_chain`` removes those syncs for Flash-Next, but it is
bound to the FR-Spec head's compact 65,536-row output.  The dense Qwen 3.5/3.6/
3.8 packs draft through the ordinary full-vocabulary head, so that lane
declines there (``no_frspec_head``) and the stock serial reader runs.

What this module does
---------------------
The same predict-and-confirm scheme, on the stock serial reader
(``fast_sampling.sparse_distribution_from_mlx_logits``) itself:

1. Before the round the host draws the ``D`` uniforms the serial lane would
   draw, in the same order from the same generator (``Generator.choice`` with
   ``p=`` draws one ``random()`` and does a right-sided ``searchsorted`` on the
   normalized cdf; ``qwen4_draft_device_chain.host_pick``).
2. On the device, per depth: the serial reader's own lazy candidate arrays
   (``fast_sampling.serial_support_device_arrays``: the same expressions on the
   same row), plus a float32 *prediction* of the host's pick that feeds the
   next depth's forward without a sync.
3. On the host, per depth: the serial reader's own host half
   (``fast_sampling.serial_support_host_rows``, including the cutoff-tie spill
   fallback) and the same inverse-cdf pick.  The host result is the authority.
4. A prediction the host does not confirm cuts the chain at that depth; the
   caller rolls the MTP cache back and finishes the round serially with the
   uniforms already drawn.

Exactness
---------
Drafted tokens, proposal distributions and the generator stream are those of
the stock serial reader for every seed: same device expressions on the same
logits, same host arithmetic, same uniforms in the same order.  The device
prediction is only ever confirmed, never trusted.

NO device work happens at import.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from .sampling import SamplerConfig, SparseDistribution

_ENV_VAR = "MTPLX_STOCK_SAMPLED_DRAFT_CHAIN"


def is_enabled(env: Any | None = None) -> bool:
    source = os.environ if env is None else env
    return str(source.get(_ENV_VAR, "")).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class StockChainPlan:
    """Request-bound draft sampler view for the full-row chain."""

    config: SamplerConfig

    def to_dict(self) -> dict[str, object]:
        return {
            "installed": True,
            "kind": "stock_full_row",
            "top_k": int(self.config.top_k),
            "temperature": float(self.config.temperature),
            "top_p": float(self.config.top_p),
        }


def claim(
    rt: Any, draft_sampler: SamplerConfig
) -> tuple[StockChainPlan | None, str | None]:
    """Bind the chain to a sampled top-k draft; ``(None, reason)`` if it cannot."""

    del rt
    if float(draft_sampler.temperature) <= 0.0:
        return None, "greedy_draft"
    if int(draft_sampler.top_k) <= 0:
        return None, "not_a_top_k_draft"
    if (
        float(draft_sampler.presence_penalty) != 0.0
        or float(draft_sampler.frequency_penalty) != 0.0
    ):
        return None, "draft_penalties"
    return StockChainPlan(config=draft_sampler), None


def release(plan: StockChainPlan | None) -> None:
    del plan


def _draft_row(draft_logits: Any) -> Any:
    """The row the stock reader samples: ``draft_logits[:, -1, :][0]``.

    ``sparse_distribution_from_mlx_logits`` then applies a no-op penalty pass
    (same array) and ``astype(float32)``; both are reproduced here.
    """

    import mlx.core as mx

    return draft_logits[:, -1, :][0].reshape(-1).astype(mx.float32)


def device_support(plan: StockChainPlan, draft_logits: Any):
    """Lazy serial-reader candidate arrays of one draft step."""

    from .fast_sampling import serial_support_device_arrays

    return serial_support_device_arrays(_draft_row(draft_logits), plan.config)


def support_arrays(support: Any) -> tuple[Any, ...]:
    """The arrays the serial reader evaluates (what the chain hands the GPU)."""

    arrays = [support.cand_idx, support.cand_vals]
    if support.cand_probs is not None:
        arrays.append(support.cand_probs)
    return tuple(arrays)


def device_predict(plan: StockChainPlan, support: Any, uniform: Any) -> Any:
    """Lazy float32 prediction of the host pick: a real token id (int32 scalar).

    Value-descending top-k over the candidate superset, the nucleus filter,
    kept entries ordered by id (dropped ones last), cdf, right-sided search.
    A prediction only: the host confirms it against its float64 pick.
    """

    import mlx.core as mx

    k = int(support.k)
    cand_idx = support.cand_idx.reshape(-1)
    cand_vals = support.cand_vals.reshape(-1)
    order = mx.argsort(-cand_vals)[:k]
    ids = mx.take(cand_idx, order)
    if support.cand_probs is not None:
        probs = mx.take(support.cand_probs.reshape(-1), order)
        cumulative_before = mx.cumsum(probs) - probs
        probs = mx.where(cumulative_before < float(plan.config.top_p), probs, 0.0)
    else:
        vals = mx.take(cand_vals, order)
        probs = mx.softmax(vals, axis=-1)
    dropped = (probs <= 0).astype(mx.int32)
    key = dropped * (1 << 30) + ids.astype(mx.int32)
    by_id = mx.argsort(key)
    ids_by_id = mx.take(ids, by_id)
    probs_by_id = mx.take(probs, by_id)
    cdf = mx.cumsum(probs_by_id)
    cdf = cdf / mx.max(cdf)
    pick = mx.minimum(mx.sum(cdf <= uniform), k - 1)
    return mx.take(ids_by_id, pick).astype(mx.int32)


def host_distribution(plan: StockChainPlan, support: Any) -> SparseDistribution | None:
    """The stock serial reader's host half on the chain's candidate arrays.

    ``None`` on a row with no finite positive mass: the caller then takes the
    stock reader's own failure path.
    """

    from .fast_sampling import _serial_row_distribution, serial_support_host_rows

    token_rows, prob_rows, vocab_size = serial_support_host_rows(support, plan.config)
    return _serial_row_distribution(token_rows[0], prob_rows[0], vocab_size)


def host_pick(distribution: SparseDistribution, uniform: float) -> int:
    """``Generator.choice(ids, p=probs)`` given the one uniform it would draw."""

    cdf = np.cumsum(distribution.probs)
    cdf /= cdf[-1]
    index = int(np.searchsorted(cdf, float(uniform), side="right"))
    index = min(index, int(distribution.token_ids.shape[0]) - 1)
    return int(distribution.token_ids[index])


def serial_read(
    plan: StockChainPlan,
    draft_logits: Any,
    uniform: float,
) -> tuple[int, SparseDistribution | None]:
    """One stock serial draft read with an already drawn uniform."""

    from .fast_sampling import sparse_distribution_from_mlx_logits

    distribution = sparse_distribution_from_mlx_logits(
        draft_logits[:, -1, :][0], plan.config
    )
    if distribution is None:
        return -1, None
    return host_pick(distribution, uniform), distribution


# Chain-lane interface shared with ``qwen4_draft_device_chain``.
eval_arrays = support_arrays


def predict(plan: StockChainPlan, support: Any, uniform: Any) -> Any:
    return device_predict(plan, support, uniform)


def host_read(plan: StockChainPlan, support: Any) -> SparseDistribution | None:
    return host_distribution(plan, support)
