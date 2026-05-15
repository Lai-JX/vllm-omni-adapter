from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from transformers.cache_utils import DynamicCache
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.logger import init_logger

from vllm_omni.debug.alpamayo_stage1_rollout_dump import maybe_dump_alpamayo_stage1_rollout
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.alpamayo1_5.runtime import (
    FlowMatching,
    PerWaypointActionInProjV2,
    UnicycleAccelCurvatureActionSpace,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest

logger = init_logger(__name__)


class Alpamayo1_5TrajectoryPipeline(nn.Module):
    """Trajectory diffusion stage for Alpamayo1.5.

    Stage 0 runs the real Qwen3VL backbone. This pipeline consumes the VLM
    outputs and the structured trajectory context carried in
    ``runtime_additional_information`` and returns trajectory tensors via
    ``DiffusionOutput.custom_output``.
    """

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        del prefix

        self.od_config = od_config
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder=None,
                revision=getattr(od_config, "revision", None),
                prefix="",
                fall_back_to_pt=True,
            )
        ]
        hf_config = getattr(getattr(od_config, "model_config", None), "hf_config", None)
        self.config = self._resolve_config(hf_config, od_config.model)
        config_dict = self._cfg_dict(self.config)
        logger.info("Alpamayo1_5TrajectoryPipeline config: %s", config_dict)

        self.vlm_backend = str(config_dict.get("vlm_backend", "")).lower()
        self.vlm_name_or_path = str(config_dict.get("vlm_name_or_path", ""))
        self.default_num_return_sequences = int(config_dict.get("num_return_sequences", 1) or 1)
        self.default_num_traj_sets = 1
        action_space_cfg = self._cfg_dict(config_dict.get("action_space_cfg"))
        self.default_future_steps = int(
            action_space_cfg.get("n_waypoints")
            or config_dict.get("n_waypoints", 0)
            or config_dict.get("tokens_per_future_traj", 128)
            or 128
        )
        self.default_num_inference_steps = int(config_dict.get("num_inference_steps", 50) or 50)
        self.default_guidance_scale = float(config_dict.get("guidance_scale", 7.5) or 7.5)

        self.traj_token_start_idx = int(config_dict.get("traj_token_start_idx", 0) or 0)
        self.traj_vocab_size = int(config_dict.get("traj_vocab_size", 4000) or 4000)
        traj_token_ids = self._cfg_dict(config_dict.get("traj_token_ids"))
        self.history_placeholder_id = int(traj_token_ids.get("history", -1) or -1)
        self.future_start_id = int(traj_token_ids.get("future_start", -1) or -1)
        self.future_end_id = int(traj_token_ids.get("future_end", -1) or -1)

        self.action_space: nn.Module | None = None
        self.diffusion: nn.Module | None = None
        self.action_in_proj: nn.Module | None = None
        self.action_out_proj: nn.Module | None = None
        self.expert: nn.Module | None = None
        self._reference_modules_init_attempted = False
        self._reference_modules_ready = False

        # Initialize the trainable stage-1 modules during pipeline construction
        # so the framework can immediately stream real checkpoint weights into them.
        self._ensure_reference_modules_initialized()

    @staticmethod
    def _dict_to_namespace(data: Any) -> Any:
        if isinstance(data, dict):
            return SimpleNamespace(**{k: Alpamayo1_5TrajectoryPipeline._dict_to_namespace(v) for k, v in data.items()})
        if isinstance(data, list):
            return [Alpamayo1_5TrajectoryPipeline._dict_to_namespace(v) for v in data]
        return data

    @classmethod
    def _resolve_config(cls, hf_config: Any, model_path: str | Path | None) -> Any:
        if hf_config is not None and cls._cfg_get(hf_config, "action_in_proj_cfg") is not None:
            return hf_config

        config_path = Path(str(model_path or "")) / "config.json"
        if config_path.is_file():
            try:
                raw_config = json.loads(config_path.read_text())
                return cls._dict_to_namespace(raw_config)
            except Exception as exc:
                logger.warning("Failed to load Alpamayo config.json from %s: %s", config_path, exc)

        return hf_config if hf_config is not None else SimpleNamespace()

    @staticmethod
    def _strip_target(cfg: dict[str, Any] | None) -> dict[str, Any]:
        normalized_cfg = Alpamayo1_5TrajectoryPipeline._cfg_dict(cfg)
        return {k: v for k, v in normalized_cfg.items() if not str(k).startswith("_")}

    @staticmethod
    def _cfg_dict(cfg: Any) -> dict[str, Any]:
        if cfg is None:
            return {}
        if isinstance(cfg, SimpleNamespace):
            return {
                k: Alpamayo1_5TrajectoryPipeline._cfg_value(v)
                for k, v in vars(cfg).items()
            }
        if isinstance(cfg, dict):
            return {
                k: Alpamayo1_5TrajectoryPipeline._cfg_value(v)
                for k, v in cfg.items()
            }
        return {}

    @staticmethod
    def _cfg_value(value: Any) -> Any:
        if isinstance(value, (dict, SimpleNamespace)):
            return Alpamayo1_5TrajectoryPipeline._cfg_dict(value)
        if isinstance(value, list):
            return [Alpamayo1_5TrajectoryPipeline._cfg_value(v) for v in value]
        return value

    @staticmethod
    def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        if isinstance(cfg, SimpleNamespace):
            return getattr(cfg, key, default)
        return default

    @staticmethod
    def _to_tensor(x: Any, *, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor | None:
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=dtype)
        try:
            return torch.as_tensor(x, device=device, dtype=dtype)
        except Exception:
            return None

    @staticmethod
    def _to_padded_long_tensor(
        x: Any,
        *,
        device: torch.device,
        pad_value: int,
    ) -> torch.Tensor | None:
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            tensor = x.to(device=device, dtype=torch.long)
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            return tensor.contiguous()
        if isinstance(x, (list, tuple)):
            if not x:
                return torch.empty((1, 0), dtype=torch.long, device=device)
            if isinstance(x[0], (list, tuple, torch.Tensor)):
                rows = [torch.as_tensor(row, device=device, dtype=torch.long).view(-1) for row in x]
                max_len = max((row.numel() for row in rows), default=0)
                padded = torch.full((len(rows), max_len), pad_value, dtype=torch.long, device=device)
                for i, row in enumerate(rows):
                    padded[i, : row.numel()] = row
                return padded.contiguous()
        try:
            tensor = torch.as_tensor(x, device=device, dtype=torch.long)
        except Exception:
            return None
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor.contiguous()

    @staticmethod
    def _infer_rollout_batch_size(x: Any) -> int:
        if isinstance(x, torch.Tensor):
            return int(x.shape[0]) if x.ndim > 1 else 1
        if isinstance(x, (list, tuple)) and x:
            first = x[0]
            if isinstance(first, (list, tuple, torch.Tensor)):
                return len(x)
        return 1

    @classmethod
    def _contains_token_id(cls, data: Any, token_id: int) -> bool:
        if token_id < 0 or data is None:
            return False
        if isinstance(data, torch.Tensor):
            return bool((data == token_id).any().item())
        if isinstance(data, (list, tuple)):
            for item in data:
                if cls._contains_token_id(item, token_id):
                    return True
            return False
        try:
            return int(data) == token_id
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _describe_debug_value(value: Any) -> str:
        if value is None:
            return "None"
        if isinstance(value, torch.Tensor):
            return (
                f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype}, "
                f"device={value.device})"
            )
        if isinstance(value, (list, tuple)):
            return f"{type(value).__name__}(len={len(value)})"
        return type(value).__name__

    @staticmethod
    def _as_hist_xyz(x: torch.Tensor | None, device: torch.device) -> torch.Tensor:
        if x is None:
            return torch.zeros((1, 2, 3), dtype=torch.float32, device=device)
        t = x.to(device=device, dtype=torch.float32)
        if t.ndim == 5:
            t = t[:, 0]
        elif t.ndim == 4:
            t = t[:, 0] if t.shape[1] == 1 else t[:, :, :, 0]
        elif t.ndim == 2:
            t = t.unsqueeze(0)
        if t.ndim != 3 or t.shape[-1] != 3:
            return torch.zeros((1, 2, 3), dtype=torch.float32, device=device)
        return t

    @staticmethod
    def _as_hist_rot(r: torch.Tensor | None, xyz_like: torch.Tensor, device: torch.device) -> torch.Tensor:
        if r is None:
            t_steps = xyz_like.shape[-2]
            eye = torch.eye(3, dtype=torch.float32, device=device)
            return eye.view(1, 1, 3, 3).expand(xyz_like.shape[0], t_steps, 3, 3).contiguous()

        t = r.to(device=device, dtype=torch.float32)
        if t.ndim == 6:
            t = t[:, 0]
        elif t.ndim == 5:
            t = t[:, 0] if t.shape[1] == 1 else t
        elif t.ndim == 3 and t.shape[-1] == 3:
            b, ts, _ = t.shape
            eye = torch.eye(3, dtype=torch.float32, device=device)
            return eye.view(1, 1, 3, 3).expand(b, ts, 3, 3).contiguous()

        if t.ndim != 4 or t.shape[-2:] != (3, 3):
            t_steps = xyz_like.shape[-2]
            eye = torch.eye(3, dtype=torch.float32, device=device)
            return eye.view(1, 1, 3, 3).expand(xyz_like.shape[0], t_steps, 3, 3).contiguous()
        return t

    @staticmethod
    def _linear_rollout(history: torch.Tensor, steps: int) -> torch.Tensor:
        if history.shape[1] >= 2:
            delta = history[:, -1, :] - history[:, -2, :]
        else:
            delta = torch.zeros_like(history[:, -1, :])
        last = history[:, -1, :]
        seq = [last + delta * float(i + 1) for i in range(steps)]
        return torch.stack(seq, dim=1)

    @staticmethod
    def _stochastic_rollout(
        history: torch.Tensor,
        steps: int,
        num_samples: int,
        num_inference_steps: int,
        guidance_scale: float,
    ) -> torch.Tensor:
        base = Alpamayo1_5TrajectoryPipeline._linear_rollout(history, steps=steps).repeat(num_samples, 1, 1)
        x = base + 0.30 * torch.randn_like(base)
        denoise_steps = max(1, int(num_inference_steps))
        g = max(0.0, float(guidance_scale))
        for s in range(denoise_steps):
            alpha = min(0.35, (1.0 / denoise_steps) * (0.5 + 0.5 * g))
            sigma = max(0.0, 0.06 * (1.0 - (s + 1) / denoise_steps))
            x = x + alpha * (base - x)
            if sigma > 0:
                x = x + sigma * torch.randn_like(x)
        return x

    @staticmethod
    def _find_eos_offset(
        sequences: torch.Tensor,
        eos_token_id: int,
        device: torch.device,
        warn: bool = True,
    ) -> torch.Tensor:
        b_star = sequences.shape[0]
        mask = sequences == eos_token_id
        has_eos = mask.any(dim=1)
        if warn:
            for i in range(b_star):
                if not has_eos[i]:
                    logger.warning(
                        "No <traj_future_start> token found in transferred stage-0 sequence for sample %s, offset will be set to the end of the sequence. This may lead to suboptimal rollout performance. Sample sequence: %s. eos_token_id: %s",
                        i,
                        sequences[i],
                        eos_token_id
                    )
        eos_positions = mask.int().argmax(dim=1)
        last_positions = torch.full((b_star,), sequences.shape[1] - 1, device=device)
        return torch.where(has_eos, eos_positions, last_positions) + 1

    @staticmethod
    def _build_expert_pos_ids_and_attn_mask(
        offset: torch.Tensor,
        rope_deltas: torch.Tensor,
        kv_cache_seq_len: int,
        n_diffusion_tokens: int,
        b_star: int,
        device: torch.device,
        prefix_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        position_ids = torch.arange(n_diffusion_tokens, device=device)
        position_ids = position_ids.view(1, 1, -1).expand(3, b_star, -1).clone()
        position_ids += (rope_deltas + offset[:, None]).to(position_ids.device)

        attention_mask = torch.zeros(
            (b_star, 1, n_diffusion_tokens, kv_cache_seq_len + n_diffusion_tokens),
            dtype=torch.float32,
            device=device,
        )
        for i in range(b_star):
            attention_mask[i, :, :, offset[i] : -n_diffusion_tokens] = torch.finfo(attention_mask.dtype).min

        if prefix_mask is not None:
            input_mask = prefix_mask[:, None, None, :]
            attention_mask[:, :, :, : input_mask.shape[-1]] = torch.where(
                input_mask == 0,
                torch.finfo(attention_mask.dtype).min,
                attention_mask[:, :, :, : input_mask.shape[-1]],
            )

        return position_ids, attention_mask

    @staticmethod
    def _normalize_prompt(prompt: Any) -> dict[str, Any]:
        if prompt is None:
            return {}
        if isinstance(prompt, dict):
            return dict(prompt)
        if hasattr(prompt, "_asdict"):
            return dict(prompt._asdict())
        if hasattr(prompt, "__dict__"):
            return dict(vars(prompt))
        return {}

    def _ensure_reference_modules_initialized(self, device: torch.device | None = None) -> None:
        if self._reference_modules_ready or self._reference_modules_init_attempted:
            return
        self._reference_modules_init_attempted = True

        try:
            action_space_cfg = self._strip_target(self._cfg_get(self.config, "action_space_cfg") or {})
            self.action_space = UnicycleAccelCurvatureActionSpace(**action_space_cfg)
            action_dims = self.action_space.get_action_space_dims()

            diffusion_cfg = self._strip_target(self._cfg_get(self.config, "diffusion_cfg") or {})
            if diffusion_cfg.get("x_dims") is None:
                diffusion_cfg["x_dims"] = action_dims
            diffusion_cfg.setdefault("num_inference_steps", self.default_num_inference_steps)
            diffusion_cfg.setdefault("inference_guidance_weight", self.default_guidance_scale)
            self.diffusion = FlowMatching(**diffusion_cfg)

            expert_cfg = self._cfg_dict(self._cfg_get(self.config, "expert_cfg"))
            expert_hidden_size = int(expert_cfg.get("hidden_size", 2048))
            action_in_proj_cfg = self._strip_target(self._cfg_get(self.config, "action_in_proj_cfg") or {})
            self.action_in_proj = PerWaypointActionInProjV2(
                in_dims=list(action_dims),
                out_dim=expert_hidden_size,
                **action_in_proj_cfg,
            )
            self.action_out_proj = nn.Linear(expert_hidden_size, int(action_dims[-1]))

            if self.vlm_name_or_path:
                vlm_cfg = AutoConfig.from_pretrained(self.vlm_name_or_path, trust_remote_code=True)
                text_cfg = getattr(vlm_cfg, "text_config", vlm_cfg)
                for k, v in expert_cfg.items():
                    setattr(text_cfg, k, v)
                self.expert = AutoModel.from_config(text_cfg)
                if hasattr(self.expert, "embed_tokens"):
                    delattr(self.expert, "embed_tokens")

            target_dtype = getattr(self.od_config, "dtype", torch.float32)
            for m in [
                self.action_space,
                self.diffusion,
                self.action_in_proj,
                self.action_out_proj,
                self.expert,
            ]:
                if m is not None:
                    if device is not None:
                        m.to(device=device)
                    if hasattr(m, "to") and target_dtype is not None:
                        m.to(dtype=target_dtype)
                    m.eval()

            self._reference_modules_ready = True
            logger.info("Alpamayo1_5 diffusion pipeline initialized")
        except Exception as exc:
            self._reference_modules_ready = False
            logger.warning("Failed to initialize Alpamayo reference modules, falling back to baseline sampler: %s", exc)

    def _build_rollout_cache(
        self,
        transferred_kv: Any,
        total_samples: int,
        device: torch.device,
    ) -> DynamicCache | None:
        key_cache = getattr(transferred_kv, "key_cache", None)
        value_cache = getattr(transferred_kv, "value_cache", None)
        if not isinstance(key_cache, list) or not isinstance(value_cache, list) or len(key_cache) != len(value_cache):
            return None

        legacy_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_idx, (k, v) in enumerate(zip(key_cache, value_cache, strict=False)):
            if k is None or v is None:
                raise ValueError(f"Transferred KV cache missing layer {layer_idx}")
            if not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor):
                raise TypeError(f"Transferred KV cache layer {layer_idx} is not tensor-backed")

            if k.ndim == 3:
                k = k.permute(1, 0, 2).unsqueeze(0)
                v = v.permute(1, 0, 2).unsqueeze(0)
            elif k.ndim != 4 or v.ndim != 4:
                raise ValueError(
                    f"Unsupported transferred KV tensor shape for layer {layer_idx}: "
                    f"key={tuple(k.shape)}, value={tuple(v.shape)}"
                )

            if k.shape[0] == 1 and total_samples > 1:
                k = k.expand(total_samples, -1, -1, -1).contiguous()
                v = v.expand(total_samples, -1, -1, -1).contiguous()
            elif k.shape[0] != total_samples:
                raise ValueError(
                    f"Transferred KV batch size mismatch at layer {layer_idx}: "
                    f"expected {total_samples}, got {k.shape[0]}"
                )

            legacy_cache.append((k.to(device=device).contiguous(), v.to(device=device).contiguous()))

        return DynamicCache.from_legacy_cache(tuple(legacy_cache))

    @staticmethod
    def _select_transferred_kv_sample(transferred_kv: Any, sample_idx: int | None) -> Any:
        if transferred_kv is None or sample_idx is None:
            return transferred_kv

        key_cache = getattr(transferred_kv, "key_cache", None)
        value_cache = getattr(transferred_kv, "value_cache", None)
        if not isinstance(key_cache, list) or not isinstance(value_cache, list):
            return transferred_kv

        sliced_key_cache = []
        sliced_value_cache = []
        changed = False
        for k, v in zip(key_cache, value_cache, strict=False):
            if isinstance(k, torch.Tensor) and k.ndim >= 4 and k.shape[0] > sample_idx:
                k = k[sample_idx : sample_idx + 1].contiguous()
                changed = True
            if isinstance(v, torch.Tensor) and v.ndim >= 4 and v.shape[0] > sample_idx:
                v = v[sample_idx : sample_idx + 1].contiguous()
                changed = True
            sliced_key_cache.append(k)
            sliced_value_cache.append(v)

        if not changed:
            return transferred_kv

        return SimpleNamespace(
            key_cache=sliced_key_cache,
            value_cache=sliced_value_cache,
        )

    def _prepare_rollout_context(
        self,
        info: dict[str, Any],
        sampling_params: Any,
        total_samples: int,
        device: torch.device,
    ) -> tuple[
        torch.Tensor,
        DynamicCache,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        transferred_kv = getattr(sampling_params, "past_key_values", None)
        if transferred_kv is None:
            raise RuntimeError(
                "Missing stage-0 KV cache: sampling_params.past_key_values is None."
            )
        transferred_kv = self._select_transferred_kv_sample(
            transferred_kv,
            int(info["stage0_sample_index"]) if info.get("stage0_sample_index") is not None else None,
        )

        sequences = info.get("stage0_sequences")
        if sequences is None:
            prompt_ids_raw = info.get("stage0_prompt_token_ids")
            if isinstance(prompt_ids_raw, torch.Tensor):
                prompt_ids = prompt_ids_raw.detach().cpu().reshape(-1).tolist()
            elif prompt_ids_raw is None:
                prompt_ids = []
            else:
                prompt_ids = list(prompt_ids_raw)

            output_ids = info.get("stage0_output_token_ids")
            if isinstance(output_ids, torch.Tensor):
                output_ids = output_ids.detach().cpu().tolist()
            elif output_ids is None:
                output_ids = []
            else:
                output_ids = list(output_ids)

            if isinstance(output_ids, list) and output_ids and isinstance(output_ids[0], (list, tuple, torch.Tensor)):
                sequences = [prompt_ids + list(ids) for ids in output_ids]
            elif prompt_ids or output_ids:
                sequences = prompt_ids + list(output_ids)

        assert self._contains_token_id(sequences, self.future_start_id), (
            "Stage-0 rollout sequence must contain future_start_id. "
            f"future_start_id={self.future_start_id}, "
            f"stage0_sequences={self._describe_debug_value(info.get('stage0_sequences'))}, "
            f"stage0_prompt_token_ids={self._describe_debug_value(info.get('stage0_prompt_token_ids'))}, "
            f"stage0_output_token_ids={self._describe_debug_value(info.get('stage0_output_token_ids'))}."
        )
        sequence_tensor = self._to_padded_long_tensor(
            sequences,
            device=device,
            pad_value=self.future_end_id if self.future_end_id >= 0 else 0,
        )
        if sequence_tensor is None:
            raise RuntimeError(
                "Failed to build stage-0 sequence tensor for rollout context. "
                f"stage0_sequences={self._describe_debug_value(info.get('stage0_sequences'))}, "
                f"stage0_prompt_token_ids={self._describe_debug_value(info.get('stage0_prompt_token_ids'))}, "
                f"stage0_output_token_ids={self._describe_debug_value(info.get('stage0_output_token_ids'))}."
            )
        if sequence_tensor.shape[0] == 1 and total_samples > 1:
            sequence_tensor = sequence_tensor.expand(total_samples, -1).contiguous()

        prompt_cache = self._build_rollout_cache(
            transferred_kv=transferred_kv,
            total_samples=total_samples,
            device=device,
        )
        if prompt_cache is None:
            raise RuntimeError(
                "Failed to rebuild transferred stage-0 KV cache. "
                f"past_key_values={type(transferred_kv).__name__}, "
                f"key_cache={self._describe_debug_value(getattr(transferred_kv, 'key_cache', None))}, "
                f"value_cache={self._describe_debug_value(getattr(transferred_kv, 'value_cache', None))}, "
                f"total_samples={total_samples}."
            )

        rope_deltas = self._to_tensor(info.get("stage0_rope_deltas"), device=device, dtype=torch.long)
        if rope_deltas is None:
            rope_deltas = torch.zeros((1, 1), dtype=torch.long, device=device)
        if rope_deltas.ndim == 0:
            rope_deltas = rope_deltas.view(1, 1)
        elif rope_deltas.ndim == 1:
            rope_deltas = rope_deltas.unsqueeze(-1)
        elif rope_deltas.ndim > 2:
            rope_deltas = rope_deltas.view(rope_deltas.shape[0], -1)[:, :1]
        if rope_deltas.shape[0] == 1 and total_samples > 1:
            rope_deltas = rope_deltas.expand(total_samples, -1).contiguous()

        tokenized_data = info.get("tokenized_data") or {}
        prefix_mask = info.get("stage0_attention_mask")
        if prefix_mask is None and isinstance(tokenized_data, dict):
            prefix_mask = tokenized_data.get("attention_mask")
        prefix_mask_tensor = self._to_tensor(prefix_mask, device=device, dtype=torch.long)
        if prefix_mask_tensor is not None:
            if prefix_mask_tensor.ndim == 1:
                prefix_mask_tensor = prefix_mask_tensor.unsqueeze(0)
            if prefix_mask_tensor.shape[0] == 1 and total_samples > 1:
                prefix_mask_tensor = prefix_mask_tensor.expand(total_samples, -1).contiguous()

        initial_noise_x0 = self._to_tensor(
            info.get("initial_noise_x0"),
            device=device,
            dtype=torch.float32,
        )
        if initial_noise_x0 is not None:
            if initial_noise_x0.ndim == 2:
                initial_noise_x0 = initial_noise_x0.unsqueeze(0)
            if initial_noise_x0.shape[0] == 1 and total_samples > 1:
                initial_noise_x0 = initial_noise_x0.expand(total_samples, -1, -1).contiguous()

        return (
            sequence_tensor,
            prompt_cache,
            rope_deltas,
            prefix_mask_tensor,
            initial_noise_x0,
        )

    @staticmethod
    def _select_sample_slice(value: Any, sample_idx: int) -> Any:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.ndim == 0 or value.shape[0] <= sample_idx:
                return value
            return value[sample_idx : sample_idx + 1].contiguous()
        if isinstance(value, list):
            if value and isinstance(value[0], (list, tuple, torch.Tensor)) and sample_idx < len(value):
                return [value[sample_idx]]
            return value
        return value

    def _sample_diffusion_euler(
        self,
        *,
        batch_size: int,
        step_fn: Any,
        device: torch.device,
        inference_step: int,
        generator: torch.Generator | None,
        initial_noise_x0: torch.Tensor | None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        assert self.diffusion is not None
        if initial_noise_x0 is not None:
            x = initial_noise_x0.to(device=device, dtype=torch.float32).contiguous()
        else:
            x = torch.randn(
                batch_size,
                *self.diffusion.x_dims,
                device=device,
                generator=generator,
            ) * temperature
        time_steps = torch.linspace(0.0, 1.0, inference_step + 1, device=device)
        n_dim = len(self.diffusion.x_dims)

        for i in range(inference_step):
            dt = time_steps[i + 1] - time_steps[i]
            dt = dt.view(1, *[1] * n_dim).expand(batch_size, *[1] * n_dim)
            t_start = time_steps[i].view(1, *[1] * n_dim).expand(batch_size, *[1] * n_dim)
            v = step_fn(x=x, t=t_start)
            x = x + dt * v

        return x

    def _resolve_future_steps(self, info: dict[str, Any]) -> int:
        action_space_cfg = info.get("action_space_cfg")
        if isinstance(action_space_cfg, dict) and action_space_cfg.get("n_waypoints") is not None:
            return int(action_space_cfg["n_waypoints"])
        if info.get("future_steps") is not None:
            return int(info["future_steps"])
        if info.get("n_waypoints") is not None:
            return int(info["n_waypoints"])
        return self.default_future_steps

    @staticmethod
    def _is_dummy_warmup(req: OmniDiffusionRequest) -> bool:
        if req.sampling_params.num_inference_steps != 1:
            return False
        if req.request_ids != ["dummy_req_id"]:
            return False
        if not req.prompts:
            return False
        first_prompt = req.prompts[0]
        prompt_text = first_prompt if isinstance(first_prompt, str) else first_prompt.get("prompt")
        return prompt_text == "dummy run"

    def _sample_with_rollout_context(
        self,
        req_id: str,
        info: dict[str, Any],
        sampling_params: Any,
        hist_xyz: torch.Tensor,
        hist_rot: torch.Tensor,
        total_samples: int,
        num_traj_sets: int,
        num_samples: int,
        future_steps: int,
        num_inference_steps: int,
        guidance_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._reference_modules_ready:
            raise RuntimeError(
                "Alpamayo1_5 diffusion pipeline is not initialized with reference stage-1 modules."
            )
        assert self.action_space is not None
        assert self.diffusion is not None
        assert self.action_in_proj is not None
        assert self.action_out_proj is not None
        assert self.expert is not None

        hist_xyz_rep = hist_xyz.repeat_interleave(total_samples, dim=0)
        hist_rot_rep = hist_rot.repeat_interleave(total_samples, dim=0)

        prepared = self._prepare_rollout_context(
            info=info,
            sampling_params=sampling_params,
            total_samples=total_samples,
            device=hist_xyz.device,
        )
        (
            sequence_tensor,
            prompt_cache,
            rope_deltas,
            prefix_mask,
            initial_noise_x0,
        ) = prepared
        prefill_seq_len = prompt_cache.get_seq_length()
        n_diffusion_tokens = self.action_space.get_action_space_dims()[0]
        expected_x_shape = tuple(int(dim) for dim in self.diffusion.x_dims)
        if initial_noise_x0 is not None:
            actual_x_shape = tuple(int(dim) for dim in initial_noise_x0.shape)
            if actual_x_shape != (total_samples, *expected_x_shape):
                raise RuntimeError(
                    "Invalid Alpamayo initial_noise_x0 shape for stage-1 rollout. "
                    f"expected={(total_samples, *expected_x_shape)}, "
                    f"got={actual_x_shape}, "
                    f"req_id={req_id}."
                )
        offset = self._find_eos_offset(
            sequences=sequence_tensor,
            eos_token_id=self.future_start_id,
            device=hist_xyz.device,
        )

        position_ids, attention_mask = self._build_expert_pos_ids_and_attn_mask(
            offset=offset,
            rope_deltas=rope_deltas,
            kv_cache_seq_len=prefill_seq_len,
            n_diffusion_tokens=n_diffusion_tokens,
            b_star=sequence_tensor.shape[0],
            device=hist_xyz.device,
            prefix_mask=prefix_mask,
        )
        maybe_dump_alpamayo_stage1_rollout(
            req_id=req_id,
            phase="stage1_rollout_context",
            payload={
                "sequence_tensor": sequence_tensor,
                "rope_deltas": rope_deltas,
                "prefix_mask": prefix_mask,
                "prefill_seq_len": int(prefill_seq_len),
                "offset": offset,
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "hist_xyz": hist_xyz,
                "hist_rot": hist_rot,
                "hist_xyz_rep": hist_xyz_rep,
                "hist_rot_rep": hist_rot_rep,
                "future_steps": int(future_steps),
                "num_inference_steps": int(num_inference_steps),
                "guidance_scale": float(guidance_scale),
                "future_start_id": int(self.future_start_id),
                "initial_noise_x0": initial_noise_x0,
            },
        )

        forward_kwargs: dict[str, Any] = {}
        if bool(self._cfg_get(self.config, "expert_non_causal_attention", True)):
            forward_kwargs["is_causal"] = False

        expert_param = next(self.expert.parameters(), None)
        expert_dtype = expert_param.dtype if expert_param is not None else hist_xyz.dtype
        first_step_dumped = False

        def step_fn(*, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            nonlocal first_step_dumped
            raw_x = x
            raw_t = t
            x = x.to(device=hist_xyz.device)
            t = t.to(device=hist_xyz.device)

            autocast_ctx = nullcontext()
            if hist_xyz.device.type == "cuda" and expert_dtype in (torch.float16, torch.bfloat16):
                autocast_ctx = torch.autocast(device_type="cuda", dtype=expert_dtype)

            with autocast_ctx:
                future_token_embeds = self.action_in_proj(x, t)
                if future_token_embeds.dim() == 2:
                    future_token_embeds = future_token_embeds.view(total_samples, n_diffusion_tokens, -1)
                expert_out = self.expert(
                    inputs_embeds=future_token_embeds,
                    position_ids=position_ids,
                    past_key_values=prompt_cache,
                    attention_mask=attention_mask,
                    use_cache=True,
                    **forward_kwargs,
                )
            prompt_cache.crop(prefill_seq_len)
            last_hidden = expert_out.last_hidden_state[:, -n_diffusion_tokens:]
            with autocast_ctx:
                pred = self.action_out_proj(last_hidden).view(
                    -1,
                    *self.action_space.get_action_space_dims(),
                )
            if not first_step_dumped:
                maybe_dump_alpamayo_stage1_rollout(
                    req_id=req_id,
                    phase="stage1_rollout_step0",
                    payload={
                        "x": raw_x,
                        "t": raw_t,
                        "x_cast": x,
                        "t_cast": t,
                        "future_token_embeds": future_token_embeds,
                        "last_hidden": last_hidden,
                        "pred": pred,
                    },
                )
                first_step_dumped = True
            return pred

        seed = getattr(sampling_params, "seed", None)
        diffusion_generator = None
        if seed is not None:
            diffusion_generator = torch.Generator(device=hist_xyz.device)
            diffusion_generator.manual_seed(int(seed))

        if diffusion_generator is not None:
            sampled_action = self._sample_diffusion_euler(
                batch_size=total_samples,
                step_fn=step_fn,
                device=hist_xyz.device,
                inference_step=num_inference_steps,
                generator=diffusion_generator,
                initial_noise_x0=initial_noise_x0,
            )
        else:
            if initial_noise_x0 is not None:
                sampled_action = self._sample_diffusion_euler(
                    batch_size=total_samples,
                    step_fn=step_fn,
                    device=hist_xyz.device,
                    inference_step=num_inference_steps,
                    generator=None,
                    initial_noise_x0=initial_noise_x0,
                )
            else:
                sampled_action = self.diffusion.sample(
                    batch_size=total_samples,
                    step_fn=step_fn,
                    device=hist_xyz.device,
                    return_all_steps=False,
                    inference_step=num_inference_steps,
                    inference_guidance_weight=guidance_scale,
                    use_classifier_free_guidance=False,
                )

        pred_xyz, pred_rot = self.action_space.action_to_traj(
            sampled_action,
            hist_xyz_rep,
            hist_rot_rep,
        )
        pred_xyz = pred_xyz.view(num_traj_sets, num_samples, future_steps, 3).detach()
        pred_rot = pred_rot.view(num_traj_sets, num_samples, future_steps, 3, 3).detach()
        return pred_xyz, pred_rot

    def forward(
        self,
        req: OmniDiffusionRequest,
    ) -> DiffusionOutput:
        if self._is_dummy_warmup(req):
            future_steps = self.default_future_steps
            dummy_xyz = torch.zeros((1, 1, future_steps, 3), dtype=torch.float32)
            dummy_rot = torch.eye(3, dtype=torch.float32).view(1, 1, 1, 3, 3).expand(
                1, 1, future_steps, 3, 3
            )
            dummy_cot = torch.empty((1, 1, 0), dtype=torch.long)
            return DiffusionOutput(
                output=[],
                trajectory_latents=dummy_xyz,
                custom_output={
                    "pred_xyz": dummy_xyz,
                    "pred_rot": dummy_rot,
                    "cot_token_ids": dummy_cot,
                    "stage0_context_used": False,
                    "dummy_warmup": True,
                },
            )

        infos = [
            self._normalize_prompt(prompt).get("additional_information") or {}
            for prompt in (req.prompts or [{}])
        ]
        if not infos:
            infos = [{}]
        device = next(self.parameters(), torch.zeros((), dtype=torch.float32)).device
        self._ensure_reference_modules_initialized(device)

        pred_xyz_list: list[torch.Tensor] = []
        pred_rot_list: list[torch.Tensor] = []
        cot_ids_list: list[torch.Tensor] = []

        for info in infos:
            rollout_source = info.get("stage0_sequences")
            if rollout_source is None:
                rollout_source = info.get("stage0_output_token_ids")
            rollout_batch_size = self._infer_rollout_batch_size(rollout_source)
            num_samples = int(
                getattr(req.sampling_params, "num_outputs_per_prompt", None)
                or info.get("num_return_sequences", self.default_num_return_sequences)
                or 1
            )
            num_traj_sets = int(info.get("num_traj_sets", self.default_num_traj_sets) or self.default_num_traj_sets)
            total_samples = max(1, num_samples * num_traj_sets)
            if rollout_batch_size > 1 and rollout_batch_size != total_samples:
                if num_traj_sets > 1 and rollout_batch_size % num_traj_sets == 0:
                    num_samples = rollout_batch_size // num_traj_sets
                else:
                    num_traj_sets = 1
                    num_samples = rollout_batch_size
                total_samples = rollout_batch_size
            future_steps = self._resolve_future_steps(info)
            num_inference_steps = int(
                req.sampling_params.num_inference_steps
                or info.get("num_inference_steps", self.default_num_inference_steps)
                or self.default_num_inference_steps
            )
            logger.info(
                "Processing sample with rollout batch size %s, num_samples %s, num_traj_sets %s, future_steps %s, num_inference_steps %s",
                rollout_batch_size,
                num_samples,
                num_traj_sets,
                future_steps,
                num_inference_steps,
            )
            guidance_scale = float(
                req.sampling_params.guidance_scale
                or info.get("guidance_scale", self.default_guidance_scale)
                or self.default_guidance_scale
            )

            hist_xyz = self._to_tensor(info.get("ego_history_xyz"), device=device)
            hist_rot = self._to_tensor(info.get("ego_history_rot"), device=device)
            hist_xyz_ref = self._as_hist_xyz(hist_xyz, device)
            hist_rot_ref = self._as_hist_rot(hist_rot, hist_xyz_ref, device)
            req_id = str(req.request_ids[0]) if getattr(req, "request_ids", None) else "unknown"

            sample_info = dict(info)
            stage0_sample_index = sample_info.get("stage0_sample_index")
            if stage0_sample_index is not None:
                sample_idx = int(stage0_sample_index)
                for key in (
                    "stage0_sequences",
                    "stage0_output_token_ids",
                    "stage0_rope_deltas",
                    "stage0_attention_mask",
                    "initial_noise_x0",
                ):
                    sample_info[key] = self._select_sample_slice(sample_info.get(key), sample_idx)

            pred_xyz, pred_rot = self._sample_with_rollout_context(
                req_id=req_id,
                info=sample_info,
                sampling_params=req.sampling_params,
                hist_xyz=hist_xyz_ref,
                hist_rot=hist_rot_ref,
                total_samples=total_samples,
                num_traj_sets=num_traj_sets,
                num_samples=num_samples,
                future_steps=future_steps,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )

            output_ids = self._to_padded_long_tensor(
                sample_info.get("stage0_output_token_ids"),
                device=device,
                pad_value=self.future_end_id if self.future_end_id >= 0 else 0,
            )
            if output_ids is None:
                output_ids = self._to_padded_long_tensor(
                    sample_info.get("stage0_sequences"),
                    device=device,
                    pad_value=self.future_end_id if self.future_end_id >= 0 else 0,
                )
            if output_ids is None:
                output_ids = torch.empty((total_samples, 0), dtype=torch.long, device=device)
            if output_ids.shape[0] == 1 and total_samples > 1:
                output_ids = output_ids.expand(total_samples, -1).contiguous()
            cot_ids = output_ids[:, -256:] if output_ids.numel() else output_ids
            cot_ids = cot_ids.view(num_traj_sets, num_samples, -1).detach()

            pred_xyz_list.append(pred_xyz.cpu())
            pred_rot_list.append(pred_rot.cpu())
            cot_ids_list.append(cot_ids.cpu())

        custom_output: dict[str, Any] = {
            "pred_xyz": pred_xyz_list[0] if len(pred_xyz_list) == 1 else pred_xyz_list,
            "pred_rot": pred_rot_list[0] if len(pred_rot_list) == 1 else pred_rot_list,
            "cot_token_ids": cot_ids_list[0] if len(cot_ids_list) == 1 else cot_ids_list,
            "stage0_context_used": getattr(req.sampling_params, "past_key_values", None) is not None,
        }
        trajectory_latents = pred_xyz_list[0] if len(pred_xyz_list) == 1 else None
        return DiffusionOutput(
            output=[],
            trajectory_latents=trajectory_latents,
            custom_output=custom_output,
        )

    def load_weights(self, weights) -> set[str]:
        if not self._reference_modules_ready:
            raise RuntimeError(
                "Alpamayo1_5 stage-1 modules are not initialized; cannot load real checkpoint weights."
            )

        loader = AutoWeightsLoader(self, skip_prefixes=["vlm"])
        return loader.load_weights(weights)
