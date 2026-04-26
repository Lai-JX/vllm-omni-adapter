from __future__ import annotations

from typing import Any

import einops
import numpy as np
import torch

TRAJ_TOKEN = {
    "history": "<|traj_history|>",
    "future": "<|traj_future|>",
    "history_start": "<|traj_history_start|>",
    "future_start": "<|traj_future_start|>",
    "history_end": "<|traj_history_end|>",
    "future_end": "<|traj_future_end|>",
}

_SPECIAL_TOKEN_KEYS = [
    "prompt_start",
    "prompt_end",
    "image_start",
    "_padding_0",
    "image_end",
    "traj_history_start",
    "_padding_1",
    "traj_history_end",
    "cot_start",
    "cot_end",
    "_padding_2",
    "_padding_3",
    "traj_future_start",
    "_padding_4",
    "traj_future_end",
    "traj_history",
    "traj_future",
    "image_pad",
    "_padding_5",
    "_padding_6",
    "_padding_7",
    "_padding_8",
    "route_start",
    "route_pad",
    "route_end",
    "question_start",
    "question_end",
    "answer_start",
    "answer_end",
]
SPECIAL_TOKENS = {key: f"<|{key}|>" for key in _SPECIAL_TOKEN_KEYS}


def tokenize_history_trajectory(
    tokenizer: Any,
    traj_data: dict[str, Any],
    start_idx: int = 0,
) -> torch.Tensor:
    assert "ego_history_xyz" in traj_data
    assert traj_data["ego_history_xyz"].ndim == 4, "ego_history_xyz must be 4D of [B, n_traj, T, 3]"

    batch_size = traj_data["ego_history_xyz"].shape[0]
    hist_xyz = traj_data["ego_history_xyz"].flatten(start_dim=0, end_dim=1)
    hist_rot = traj_data["ego_history_rot"].flatten(start_dim=0, end_dim=1)

    hist_idx = (
        tokenizer.encode(
            hist_xyz=hist_xyz[:, :1],
            hist_rot=hist_rot[:, :1],
            fut_xyz=hist_xyz,
            fut_rot=hist_rot,
        )
        + start_idx
    )
    return einops.rearrange(hist_idx, "(b n_traj) n -> b (n_traj n)", b=batch_size)


class DeltaTrajectoryTokenizer:
    def __init__(
        self,
        ego_xyz_min: tuple[float, float, float] = (-4, -4, -10),
        ego_xyz_max: tuple[float, float, float] = (4, 4, 10),
        ego_yaw_min: float = -np.pi,
        ego_yaw_max: float = np.pi,
        num_bins: int = 1000,
        predict_yaw: bool = False,
        load_weights: bool = False,
    ):
        del load_weights
        self.ego_xyz_min = ego_xyz_min
        self.ego_xyz_max = ego_xyz_max
        self.num_bins = num_bins
        self._predict_yaw = predict_yaw
        self.ego_yaw_min = ego_yaw_min
        self.ego_yaw_max = ego_yaw_max

    @property
    def vocab_size(self) -> int:
        return self.num_bins

    def encode(
        self,
        hist_xyz: torch.Tensor,
        hist_rot: torch.Tensor,
        fut_xyz: torch.Tensor,
        fut_rot: torch.Tensor,
        hist_tstamp: torch.Tensor | None = None,
        fut_tstamp: torch.Tensor | None = None,
    ) -> torch.LongTensor:
        del hist_xyz, hist_rot, hist_tstamp, fut_tstamp
        xyz = torch.nn.functional.pad(fut_xyz, [0, 0, 1, 0, 0, 0])
        xyz = xyz[:, 1:] - xyz[:, :-1]
        ego_xyz_max = torch.tensor(self.ego_xyz_max, dtype=xyz.dtype, device=xyz.device)
        ego_xyz_min = torch.tensor(self.ego_xyz_min, dtype=xyz.dtype, device=xyz.device)
        xyz = (xyz - ego_xyz_min) / (ego_xyz_max - ego_xyz_min)
        xyz = (xyz * (self.num_bins - 1)).round().long()
        xyz = xyz.clamp(0, self.num_bins - 1)
        if not self._predict_yaw:
            return einops.rearrange(xyz, "b n m -> b (n m)")

        yaw = torch.atan2(fut_rot[..., 0, 1], fut_rot[..., 0, 0])
        yaw_padded = torch.nn.functional.pad(yaw, [1, 0, 0, 0])
        delta_yaw = yaw_padded[:, 1:] - yaw_padded[:, :-1]
        delta_yaw = torch.atan2(torch.sin(delta_yaw), torch.cos(delta_yaw))
        delta_yaw = (delta_yaw - self.ego_yaw_min) / (self.ego_yaw_max - self.ego_yaw_min)
        delta_yaw = (delta_yaw * (self.num_bins - 1)).round().long()
        delta_yaw = delta_yaw.clamp(0, self.num_bins - 1)
        xyzw = torch.cat([xyz, delta_yaw.unsqueeze(-1)], dim=-1)
        return einops.rearrange(xyzw, "b n m -> b (n m)")


class DiscreteTrajectoryTokenizer:
    def __init__(
        self,
        num_bins: int,
        load_weights: bool = False,
        **_: Any,
    ):
        del load_weights
        self.num_bins = int(num_bins)

    @property
    def vocab_size(self) -> int:
        return self.num_bins


def instantiate_local_hist_traj_tokenizer(cfg: dict[str, Any] | None) -> Any:
    if not cfg:
        raise ValueError("Empty Alpamayo trajectory tokenizer config")
    target = str(cfg.get("_target_", ""))
    kwargs = {k: v for k, v in cfg.items() if not str(k).startswith("_")}
    if target.endswith("DeltaTrajectoryTokenizer"):
        return DeltaTrajectoryTokenizer(**kwargs)
    if target.endswith("DiscreteTrajectoryTokenizer"):
        return DiscreteTrajectoryTokenizer(**kwargs)
    raise ValueError(f"Unsupported local Alpamayo tokenizer target: {target}")
