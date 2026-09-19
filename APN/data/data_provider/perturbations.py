"""Shared history perturbations for cross-baseline IMTS tests."""

from __future__ import annotations

import hashlib
from typing import Any

import torch


HISTORY_PERTURBATIONS = (
    "original",
    "shuffle_timestamps",
    "swap_halves",
    "drop_early_history",
    "keep_last_obs",
    "keep_time_gaps",
)


def _stable_seed(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & 0x7FFFFFFF


def _sample_key(batch: dict[str, torch.Tensor], index: int, flag: str) -> str:
    sample_ids = batch.get("sample_ID")
    if isinstance(sample_ids, torch.Tensor) and sample_ids.numel() > index:
        return f"{flag}:{float(sample_ids[index].detach().cpu().item()):.6f}:{index}"
    return f"{flag}:{index}"


def _observed_rows(x_mask: torch.Tensor) -> torch.Tensor:
    if x_mask.ndim == 1:
        return torch.where(x_mask > 0)[0]
    return torch.where(x_mask.sum(dim=-1) > 0)[0]


def _apply_keep_fraction(
    x: torch.Tensor,
    x_mask: torch.Tensor,
    keep_fraction: float,
    seed: int,
    sample_key: str,
) -> None:
    if keep_fraction >= 1.0:
        return
    if keep_fraction <= 0.0:
        raise ValueError(f"history_keep_fraction must be in (0, 1], got {keep_fraction}")

    observed = torch.nonzero(x_mask > 0, as_tuple=False)
    n_observed = int(observed.shape[0])
    if n_observed == 0:
        return

    n_keep = max(1, int(round(n_observed * keep_fraction)))
    n_keep = min(n_keep, n_observed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_stable_seed(seed, sample_key, "keep_fraction", keep_fraction))
    keep_local = torch.randperm(n_observed, generator=generator)[:n_keep]
    keep_mask = torch.zeros(n_observed, dtype=torch.bool)
    keep_mask[keep_local] = True
    drop = observed[~keep_mask]
    if drop.numel() == 0:
        return

    x_mask[tuple(drop.T)] = 0
    x[tuple(drop.T)] = 0


def _apply_keep_last_obs(x: torch.Tensor, x_mask: torch.Tensor) -> None:
    if x_mask.ndim == 1:
        observed = torch.where(x_mask > 0)[0]
        if observed.numel() > 1:
            drop = observed[:-1]
            x_mask[drop] = 0
            x[drop] = 0
        return

    n_variables = x_mask.shape[-1]
    for variable in range(n_variables):
        observed = torch.where(x_mask[:, variable] > 0)[0]
        if observed.numel() > 1:
            drop = observed[:-1]
            x_mask[drop, variable] = 0
            x[drop, variable] = 0


def apply_history_controls(
    batch: dict[str, torch.Tensor],
    configs: Any,
    flag: str,
) -> dict[str, torch.Tensor]:
    """Apply shared history perturbation/thinning to a collated APN batch.

    The target tensors are intentionally untouched. For neural baselines, use
    these flags mainly with ``is_training=0`` to evaluate an original checkpoint
    under perturbed test histories.
    """

    perturbation = getattr(configs, "history_perturbation", "original")
    keep_fraction = float(getattr(configs, "history_keep_fraction", 1.0))
    seed = int(getattr(configs, "history_perturb_seed", 1729))

    if perturbation not in HISTORY_PERTURBATIONS:
        raise ValueError(f"Unsupported history_perturbation={perturbation!r}")
    if perturbation == "original" and keep_fraction >= 1.0:
        return batch
    if "x" not in batch or "x_mask" not in batch or "x_mark" not in batch:
        return batch

    out = dict(batch)
    x = out["x"].clone()
    x_mask = out["x_mask"].clone()
    x_mark = out["x_mark"].clone()

    if x.ndim < 2 or x_mask.ndim < 2:
        out["x"] = x
        out["x_mask"] = x_mask
        out["x_mark"] = x_mark
        return out

    batch_size = int(x.shape[0])
    for sample_index in range(batch_size):
        sample_key = _sample_key(out, sample_index, flag)
        rows = _observed_rows(x_mask[sample_index])
        n_rows = int(rows.numel())

        if perturbation == "shuffle_timestamps" and n_rows > 1:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(_stable_seed(seed, sample_key, "shuffle_timestamps"))
            order = torch.randperm(n_rows, generator=generator)
            x_mark[sample_index, rows] = x_mark[sample_index, rows[order]]
        elif perturbation == "swap_halves" and n_rows > 1:
            midpoint = n_rows // 2
            order = torch.cat([rows[midpoint:], rows[:midpoint]])
            x[sample_index, rows] = x[sample_index, order]
            x_mask[sample_index, rows] = x_mask[sample_index, order]
        elif perturbation == "drop_early_history" and n_rows > 1:
            drop_rows = rows[: n_rows // 2]
            x_mask[sample_index, drop_rows] = 0
            x[sample_index, drop_rows] = 0
        elif perturbation == "keep_last_obs":
            _apply_keep_last_obs(x[sample_index], x_mask[sample_index])
        elif perturbation == "keep_time_gaps":
            x[sample_index] = torch.where(x_mask[sample_index] > 0, torch.zeros_like(x[sample_index]), x[sample_index])

        _apply_keep_fraction(
            x=x[sample_index],
            x_mask=x_mask[sample_index],
            keep_fraction=keep_fraction,
            seed=seed,
            sample_key=sample_key,
        )

    out["x"] = x
    out["x_mask"] = x_mask
    out["x_mark"] = x_mark
    return out
