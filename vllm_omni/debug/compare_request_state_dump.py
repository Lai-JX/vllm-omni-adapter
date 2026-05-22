"""Compare two request-state dump files produced by request_state_dump.py."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _decode_tensor_like(value: Any) -> Any:
    if isinstance(value, Mapping):
        if value.get("__tensor__") is True:
            data = value.get("data")
            dtype_name = str(value.get("dtype", "torch.float32"))
            dtype = getattr(torch, dtype_name.removeprefix("torch."), None)
            if dtype is None:
                return torch.tensor(data)
            return torch.tensor(data, dtype=dtype)
        if value.get("__ndarray__") is True:
            data = value.get("data")
            dtype_name = str(value.get("dtype", "float32"))
            dtype = getattr(np, dtype_name, None)
            if dtype is None:
                return np.asarray(data)
            return np.asarray(data, dtype=dtype)
        return {str(k): _decode_tensor_like(v) for k, v in value.items()}
    if _is_sequence(value):
        return [_decode_tensor_like(v) for v in value]
    return value


def _load_dump(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    data = _decode_tensor_like(data)
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict payload in {path}, got {type(data)!r}")
    return data


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _normalize(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(value))
    if isinstance(value, Mapping):
        return {str(k): _normalize(v) for k, v in value.items()}
    if _is_sequence(value):
        return [_normalize(v) for v in value]
    return value


def _tensor_first_diff(lhs: torch.Tensor, rhs: torch.Tensor) -> str | None:
    if lhs.shape != rhs.shape:
        return f"shape mismatch: {tuple(lhs.shape)} vs {tuple(rhs.shape)}"
    if lhs.dtype != rhs.dtype:
        return f"dtype mismatch: {lhs.dtype} vs {rhs.dtype}"
    if lhs.numel() == 0:
        return None

    if lhs.dtype.is_floating_point:
        diff = lhs != rhs
        if not bool(diff.any().item()):
            return None
        flat_idx = int(diff.reshape(-1).nonzero(as_tuple=False)[0].item())
        coord = np.unravel_index(flat_idx, tuple(lhs.shape))
        return f"first diff at {coord}: {lhs[coord].item()} vs {rhs[coord].item()}"

    diff = lhs != rhs
    if not bool(diff.any().item()):
        return None
    flat_idx = int(diff.reshape(-1).nonzero(as_tuple=False)[0].item())
    coord = np.unravel_index(flat_idx, tuple(lhs.shape))
    return f"first diff at {coord}: {lhs[coord].item()} vs {rhs[coord].item()}"


def _tensor_numeric_summary(lhs: torch.Tensor, rhs: torch.Tensor) -> str | None:
    if lhs.shape != rhs.shape:
        return None
    if lhs.numel() == 0:
        return "numerically equal (empty tensor), max_abs_diff=0.0"

    if lhs.is_floating_point() or rhs.is_floating_point():
        lhs_cmp = lhs.to(torch.float64)
        rhs_cmp = rhs.to(torch.float64)
    else:
        lhs_cmp = lhs.to(torch.int64)
        rhs_cmp = rhs.to(torch.int64)

    abs_diff = (lhs_cmp - rhs_cmp).abs()
    max_abs_diff = float(abs_diff.max().item()) if abs_diff.numel() else 0.0
    if max_abs_diff == 0.0:
        return f"numerically equal, max_abs_diff={max_abs_diff}"

    flat_idx = int(abs_diff.reshape(-1).argmax().item())
    coord = np.unravel_index(flat_idx, tuple(lhs.shape))
    return (
        f"numerically different, max_abs_diff={max_abs_diff}, "
        f"max_diff_at={coord}: {lhs_cmp[coord].item()} vs {rhs_cmp[coord].item()}"
    )


def _compare(lhs: Any, rhs: Any, path: str, diffs: list[str]) -> None:
    lhs = _normalize(lhs)
    rhs = _normalize(rhs)

    if isinstance(lhs, torch.Tensor) or isinstance(rhs, torch.Tensor):
        if not isinstance(lhs, torch.Tensor) or not isinstance(rhs, torch.Tensor):
            diffs.append(f"{path}: type mismatch {type(lhs).__name__} vs {type(rhs).__name__}")
            return
        shape_matches = lhs.shape == rhs.shape
        dtype_matches = lhs.dtype == rhs.dtype
        shape_desc = f"lhs_shape={tuple(lhs.shape)} rhs_shape={tuple(rhs.shape)}"
        if not shape_matches:
            diffs.append(f"{path}: shape mismatch {tuple(lhs.shape)} vs {tuple(rhs.shape)}")
        if not dtype_matches:
            diffs.append(f"{path}: dtype mismatch {lhs.dtype} vs {rhs.dtype}, {shape_desc}")

        if shape_matches:
            numeric_summary = _tensor_numeric_summary(lhs, rhs)
            if numeric_summary is not None and (not dtype_matches or not torch.equal(lhs, rhs)):
                diffs.append(f"{path}: {numeric_summary}, {shape_desc}")

        # Only do exact elementwise comparison when tensor metadata is compatible.
        if shape_matches and dtype_matches and not torch.equal(lhs, rhs):
            detail = _tensor_first_diff(lhs, rhs)
            diffs.append(f"{path}: tensor mismatch, {detail}, {shape_desc}")
        return

    if isinstance(lhs, Mapping) or isinstance(rhs, Mapping):
        if not isinstance(lhs, Mapping) or not isinstance(rhs, Mapping):
            diffs.append(f"{path}: type mismatch {type(lhs).__name__} vs {type(rhs).__name__}")
            return
        lhs_keys = set(lhs.keys())
        rhs_keys = set(rhs.keys())
        only_lhs = sorted(lhs_keys - rhs_keys)
        only_rhs = sorted(rhs_keys - lhs_keys)
        if only_lhs:
            diffs.append(f"{path}: keys only in lhs: {only_lhs}")
        if only_rhs:
            diffs.append(f"{path}: keys only in rhs: {only_rhs}")
        for key in sorted(lhs_keys & rhs_keys):
            child_path = f"{path}.{key}" if path else str(key)
            _compare(lhs[key], rhs[key], child_path, diffs)
        return

    if _is_sequence(lhs) or _is_sequence(rhs):
        if not _is_sequence(lhs) or not _is_sequence(rhs):
            diffs.append(f"{path}: type mismatch {type(lhs).__name__} vs {type(rhs).__name__}")
            return
        if len(lhs) != len(rhs):
            diffs.append(f"{path}: length mismatch {len(lhs)} vs {len(rhs)}")
        for idx, (l_item, r_item) in enumerate(zip(lhs, rhs, strict=False)):
            child_path = f"{path}[{idx}]"
            _compare(l_item, r_item, child_path, diffs)
        return

    if lhs != rhs:
        diffs.append(f"{path}: value mismatch {lhs!r} vs {rhs!r}")


def _default_focus_keys() -> list[str]:
    return [
        "prompt_token_ids",
        "mrope_positions",
        "mrope_position_delta",
        "sampling_params",
        "mm_features",
        "additional_information",
        "extra",
    ]


def _extract_key_path(data: Mapping[str, Any], key_path: str) -> Any:
    value: Any = data
    for part in key_path.split("."):
        if not isinstance(value, Mapping):
            raise KeyError(f"{key_path}: {part} is not under a mapping")
        value = value[part]
    return value


def compare_dump_dicts(
    lhs: Mapping[str, Any],
    rhs: Mapping[str, Any],
    *,
    keys: Sequence[str] | None = None,
) -> list[str]:
    diffs: list[str] = []
    key_paths = list(keys) if keys is not None else _default_focus_keys()
    for key_path in key_paths:
        try:
            lhs_value = _extract_key_path(lhs, key_path)
        except KeyError as exc:
            diffs.append(f"{key_path}: missing in lhs ({exc})")
            continue
        try:
            rhs_value = _extract_key_path(rhs, key_path)
        except KeyError as exc:
            diffs.append(f"{key_path}: missing in rhs ({exc})")
            continue
        _compare(lhs_value, rhs_value, key_path, diffs)
    return diffs


def compare_dump_files(
    lhs_path: Path,
    rhs_path: Path,
    *,
    keys: Sequence[str] | None = None,
) -> list[str]:
    lhs = _load_dump(lhs_path)
    rhs = _load_dump(rhs_path)
    return compare_dump_dicts(lhs, rhs, keys=keys)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lhs", type=Path, help="Left dump file")
    parser.add_argument("rhs", type=Path, help="Right dump file")
    parser.add_argument(
        "--keys",
        nargs="*",
        default=_default_focus_keys(),
        help="Top-level or dotted key paths to compare",
    )
    parser.add_argument(
        "--fail-on-diff",
        action="store_true",
        help="Return exit code 1 when differences are found",
    )
    args = parser.parse_args()

    print(f"lhs: {args.lhs}")
    print(f"rhs: {args.rhs}")
    print(f"keys: {args.keys}")
    diffs = compare_dump_files(args.lhs, args.rhs, keys=args.keys)
    if not diffs:
        print("No differences found.")
        return 0

    print(f"Differences found: {len(diffs)}")
    for diff in diffs:
        print(f"- {diff}")
    return 1 if args.fail_on_diff else 0


if __name__ == "__main__":
    raise SystemExit(main())
