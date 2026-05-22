import asyncio
import gc
import os
import sys
import tempfile
from pathlib import Path

# OMNI_ROOT = Path(__file__).resolve().parents[5]
# WORKSPACE_ROOT = OMNI_ROOT.parent
# STAGE_CONFIG_PATH = OMNI_ROOT / "vllm_omni" / "model_executor" / "stage_configs" / "alpamayo1_5.yaml"
# print(sys.path)
# for path in (OMNI_ROOT, WORKSPACE_ROOT):
#     path_str = str(path)
#     if path_str not in sys.path:
#         sys.path.insert(0, path_str)
# print(sys.path, OMNI_ROOT, STAGE_CONFIG_PATH)
# exit(0)  # to check the sys.path before importing the rest of the code

import torch
import yaml
from transformers import AutoConfig
from vllm.outputs import RequestOutput

import common as ct
from tests.diffusion.models.alpamoya.custom_test.offline.alpamoya_compare_original import (
    REQUEST_ID,
    actual_request_state_dump_path,
    actual_stage1_transition_dump_path,
    build_fixed_initial_noise_x0,
    build_stage_params,
    capture_stage1_transition_dump,
    compare_tensors,
    extract_cot_body,
    load_shared_data,
    print_dump_compare_report,
    run_original,
    write_reference_request_dumps,
    write_reference_stage1_transition_dump,
)
from tests.diffusion.models.alpamoya.custom_test.offline.alpamoya_fixed_x0_compare import compute_min_ade
from vllm.inputs import TokensPrompt
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.inputs.data import OmniTokensPrompt

from my_example2.src.inputs.alpamayo_batch import (
    build_rollout_preprocess_fn,
    build_rollout_processor,
    build_traj_fuser,
    sample_to_vllm_prompt,
)


def _extract_prompt_logprobs(
    output: RequestOutput,
    num_prompt_logprobs: int | None,
) -> tuple[list[list[float | None]], list[list[int | None]]] | tuple[None, None]:
    if num_prompt_logprobs is None:
        return None, None

    prompt_logprobs_ls: list[list[float | None]] = []
    prompt_ids_ls: list[list[int | None]] = []
    for logprobs_dict in output.prompt_logprobs[1:]:
        if num_prompt_logprobs == 0:
            token_id_str = list(logprobs_dict.keys())[0]
            logprob = logprobs_dict[token_id_str].logprob
            prompt_logprobs_ls.append([logprob])
            prompt_ids_ls.append([int(token_id_str)])
        else:
            prompt_ids = [None] * num_prompt_logprobs
            prompt_logprobs = [None] * num_prompt_logprobs
            for token_id_str, token_logprob in logprobs_dict.items():
                rank = token_logprob.rank
                if rank > num_prompt_logprobs:
                    continue
                prompt_ids[rank - 1] = int(token_id_str)
                prompt_logprobs[rank - 1] = token_logprob.logprob
            prompt_logprobs_ls.append(prompt_logprobs)
            prompt_ids_ls.append(prompt_ids)

    prompt_logprobs_ls.append([0.0] * max(num_prompt_logprobs, 1))
    prompt_ids_ls.append([0] * max(num_prompt_logprobs, 1))
    return prompt_logprobs_ls, prompt_ids_ls


def _build_preencoded_prompt(
    *,
    initial_noise_x0: torch.Tensor | None = None,
) -> tuple[OmniTokensPrompt, int]:
    data, _, _ = load_shared_data()
    model_config = AutoConfig.from_pretrained(ct.MODEL_PATH, trust_remote_code=True)
    processor = build_rollout_processor(model_config)
    preprocess_fn = build_rollout_preprocess_fn(model_config)
    traj_fuser = build_traj_fuser(model_config)

    sample = dict(data)
    for key in list(sample.keys()):
        if key.startswith("ego_") and isinstance(sample[key], torch.Tensor):
            sample[key] = sample[key].squeeze(0)
    sample["tokenized_data"] = preprocess_fn(data=sample)

    prompt_dict = sample_to_vllm_prompt(
        sample,
        processor.tokenizer,
        model_config,
        traj_fuser,
    )
    additional_information = ct.build_additional_information(
        data,
        initial_noise_x0=initial_noise_x0,
    )
    prompt_dict["additional_information"] = additional_information

    prompt_token_ids = list(prompt_dict["prompt_token_ids"])
    postprocess_prompt_ids = list(prompt_dict.get("postprocess_prompt_token_ids") or prompt_token_ids)
    prompt_token_count = len(postprocess_prompt_ids)

    tokenized_data = dict(additional_information.get("tokenized_data") or {})
    tokenized_data["input_ids"] = torch.tensor(postprocess_prompt_ids, dtype=torch.long).unsqueeze(0)
    source_attention_mask = sample.get("tokenized_data", {}).get("attention_mask")
    if isinstance(source_attention_mask, torch.Tensor):
        tokenized_data["attention_mask"] = source_attention_mask.detach().cpu().to(torch.long).contiguous()
    else:
        tokenized_data["attention_mask"] = torch.ones((1, prompt_token_count), dtype=torch.long)
    additional_information["tokenized_data"] = tokenized_data

    multi_modal_data = prompt_dict.get("multi_modal_data")
    mm_processor_kwargs = prompt_dict.get("mm_processor_kwargs")
    prompt = TokensPrompt(
        prompt_token_ids=prompt_token_ids,
        multi_modal_data=multi_modal_data,
        mm_processor_kwargs=mm_processor_kwargs,
    )
    omni_prompt: OmniTokensPrompt = OmniTokensPrompt(
        **prompt,
        additional_information=additional_information,
    )
    omni_prompt["postprocess_prompt_token_ids"] = postprocess_prompt_ids
    return omni_prompt, prompt_token_count


def _build_two_stage_yaml_without_hooks() -> str:
    yaml_path = ct.build_two_stage_yaml(str(ct.YAML_PATH))
    yaml_config = yaml.safe_load(Path(yaml_path).read_text())
    stage0 = dict(yaml_config["stage_args"][0])
    stage0.pop("prompt_rewrite_func", None)
    stage0.pop("request_postprocess_func", None)
    yaml_config["stage_args"][0] = stage0
    Path(yaml_path).write_text(yaml.safe_dump(yaml_config, sort_keys=False))
    return yaml_path


async def run_omni_no_rewrite(
    dump_dir: Path | None = None,
    initial_noise_x0: torch.Tensor | None = None,
):
    prompt, prompt_token_count = _build_preencoded_prompt(initial_noise_x0=initial_noise_x0)
    yaml_path = _build_two_stage_yaml_without_hooks()

    tokenizer = ct.build_alpamayo_stage0_tokenizer(ct.MODEL_PATH)
    stage0_params, stage1_params = build_stage_params(tokenizer)
    stage0_params.logprobs = 0
    stage0_params.prompt_logprobs = 0
    yaml_config = yaml.safe_load(Path(yaml_path).read_text())
    stage0_engine_args = yaml_config["stage_args"][0].setdefault("engine_args", {})
    stage1_engine_args = yaml_config["stage_args"][1].setdefault("engine_args", {})
    requested_max_tokens = int(getattr(stage0_params, "max_tokens", 256) or 256)
    debug_max_model_len = max(
        3328,
        ((prompt_token_count + requested_max_tokens + 127) // 128) * 128,
    )
    debug_max_batched_tokens = max(
        prompt_token_count,
        min(debug_max_model_len, 3584),
    )

    stage0_engine_args["gpu_memory_utilization"] = 0.58
    stage0_engine_args["skip_mm_profiling"] = True
    stage0_engine_args["max_model_len"] = min(
        int(stage0_engine_args.get("max_model_len", debug_max_model_len)),
        debug_max_model_len,
    )
    stage0_engine_args["max_num_batched_tokens"] = min(
        int(stage0_engine_args.get("max_num_batched_tokens", debug_max_batched_tokens)),
        debug_max_batched_tokens,
    )
    stage1_engine_args["gpu_memory_utilization"] = 0.10
    Path(yaml_path).write_text(yaml.safe_dump(yaml_config, sort_keys=False))

    previous_env = {
        "VLLM_OMNI_REQUEST_DUMP_DIR": os.environ.get("VLLM_OMNI_REQUEST_DUMP_DIR"),
        "VLLM_OMNI_REQUEST_DUMP_REQ_IDS": os.environ.get("VLLM_OMNI_REQUEST_DUMP_REQ_IDS"),
        "VLLM_OMNI_REQUEST_DUMP_PHASES": os.environ.get("VLLM_OMNI_REQUEST_DUMP_PHASES"),
        "VLLM_OMNI_ALPAMAYO_STAGE1_DUMP_DIR": os.environ.get("VLLM_OMNI_ALPAMAYO_STAGE1_DUMP_DIR"),
        "VLLM_OMNI_ALPAMAYO_STAGE1_DUMP_REQ_IDS": os.environ.get("VLLM_OMNI_ALPAMAYO_STAGE1_DUMP_REQ_IDS"),
        "VLLM_OMNI_ALPAMAYO_STAGE1_DUMP_PHASES": os.environ.get("VLLM_OMNI_ALPAMAYO_STAGE1_DUMP_PHASES"),
    }
    if dump_dir is not None:
        os.environ["VLLM_OMNI_REQUEST_DUMP_DIR"] = str(dump_dir)
        os.environ["VLLM_OMNI_REQUEST_DUMP_REQ_IDS"] = REQUEST_ID
        os.environ["VLLM_OMNI_REQUEST_DUMP_PHASES"] = "request_state_initialized,request_state_batched"
        os.environ["VLLM_OMNI_ALPAMAYO_STAGE1_DUMP_DIR"] = str(dump_dir)
        os.environ["VLLM_OMNI_ALPAMAYO_STAGE1_DUMP_REQ_IDS"] = REQUEST_ID
        os.environ["VLLM_OMNI_ALPAMAYO_STAGE1_DUMP_PHASES"] = "stage1_transition,stage1_rollout_context,stage1_rollout_step"

    omni = AsyncOmni(model=ct.MODEL_PATH, stage_configs_path=yaml_path)
    try:
        final_output = None
        stage0_params.n = 2
        stage1_params.num_outputs_per_prompt = 1
        print(stage0_params)
        async for out in omni.generate(
            prompt=prompt,
            request_id=REQUEST_ID,
            sampling_params_list=[stage0_params, stage1_params],
        ):
            final_output = out
        if final_output is None:
            raise RuntimeError("no final output")

        custom = final_output.custom_output or {}
        inner_output = getattr(final_output, "request_output", None)
        inner_upstream_output = getattr(inner_output, "_upstream_request_output", None)
        print("debug final_output_type:", type(final_output).__name__)
        print("debug final_output.request_output_type:", type(inner_output).__name__ if inner_output is not None else None)
        print("debug inner_upstream_request_output_is_none:", inner_upstream_output is None)
        print(
            "debug inner_upstream_completion_outputs_len:",
            len(list(getattr(inner_upstream_output, "outputs", []) or [])),
        )
        print(
            "debug inner_upstream_completion_token_lens:",
            [len(list(getattr(output, "token_ids", []) or [])) for output in list(getattr(inner_upstream_output, "outputs", []) or [])],
        )
        upstream_output = getattr(final_output, "_upstream_request_output", None)
        upstream_completion_outputs = list(getattr(upstream_output, "outputs", []) or [])
        print("debug upstream_request_output_is_none:", upstream_output is None)
        print("debug upstream_completion_outputs_len:", len(upstream_completion_outputs))
        print(
            "debug upstream_completion_token_lens:",
            [len(list(getattr(output, "token_ids", []) or [])) for output in upstream_completion_outputs],
        )
        completion_outputs = list(final_output.outputs or [])
        print("debug final_output.outputs_len:", len(completion_outputs))
        token_ids = [list(getattr(output, "token_ids", []) or []) for output in completion_outputs]
        log_probs = []
        for sample_token_ids, completion_output in zip(token_ids, completion_outputs):
            sample_logprobs = getattr(completion_output, "logprobs", None)
            if sample_logprobs is None:
                log_probs.append(None)
                continue
            try:
                log_probs.append(
                    [token_logprobs[sample_token_ids[i]].logprob for i, token_logprobs in enumerate(sample_logprobs)]
                )
            except Exception:
                logger.exception("Failed to extract omni completion log_probs for one sample; dropping them.")
                log_probs.append(None)
        prompt_logprobs, prompt_ids = _extract_prompt_logprobs(final_output, stage0_params.prompt_logprobs)
        finish_reason = [getattr(output, "finish_reason", None) for output in completion_outputs]
        routed_experts = [getattr(output, "routed_experts", None) for output in completion_outputs]

        cot_token_ids = custom["cot_token_ids"].detach().cpu()
        result = {
            "token_ids": token_ids,
            "log_probs": log_probs,
            "prompt_ids": prompt_ids,
            "prompt_logprobs": prompt_logprobs,
            "finish_reason": finish_reason,
            "routed_experts": routed_experts,
            "extra_fields": {
                "pred_xyz": custom["pred_xyz"].detach().cpu(),
                "pred_rot": custom["pred_rot"].detach().cpu(),
                "cot_token_ids": cot_token_ids,
                "stage0_context_used": custom.get("stage0_context_used"),
            },
        }
        result["pred_xyz"] = result["extra_fields"]["pred_xyz"]
        result["pred_rot"] = result["extra_fields"]["pred_rot"]
        result["cot_token_ids"] = result["extra_fields"]["cot_token_ids"]
        result["stage0_context_used"] = result["extra_fields"]["stage0_context_used"]
        cot_token_id_rows = cot_token_ids.view(-1, cot_token_ids.shape[-1])
        result["cot_text_raw_all"] = [
            tokenizer.decode(sample_ids.tolist(), skip_special_tokens=False) for sample_ids in cot_token_id_rows
        ]
        result["cot_text_all"] = [extract_cot_body(text) for text in result["cot_text_raw_all"]]
        result["cot_text_raw"] = result["cot_text_raw_all"][0]
        result["cot_text"] = result["cot_text_all"][0]
        return result
    finally:
        omni.shutdown()
        Path(yaml_path).unlink(missing_ok=True)
        for env_name, env_value in previous_env.items():
            if env_value is None:
                os.environ.pop(env_name, None)
            else:
                os.environ[env_name] = env_value


async def main() -> None:
    dump_dir = Path(tempfile.mkdtemp(prefix="alpamayo_no_rewrite_compare_"))
    reference_dumps = write_reference_request_dumps(dump_dir)
    initial_noise_x0 = build_fixed_initial_noise_x0()
    original = run_original(dump_dir=dump_dir, forced_initial_noise_x0=initial_noise_x0)
    reference_stage1_dump = write_reference_stage1_transition_dump(
        dump_dir,
        original["stage0_debug"],
    )

    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.ipc_collect()
    await asyncio.sleep(2)

    with capture_stage1_transition_dump(dump_dir):
        omni = await run_omni_no_rewrite(dump_dir=dump_dir, initial_noise_x0=initial_noise_x0)

    shared_data, _, _ = load_shared_data()
    print("original pred_xyz.shape:", tuple(original["pred_xyz"].shape))
    print("omni pred_xyz.shape:", tuple(omni["pred_xyz"].shape))
    print("original pred_rot.shape:", tuple(original["pred_rot"].shape))
    print("omni pred_rot.shape:", tuple(omni["pred_rot"].shape))
    print("omni num_samples:", len(omni["token_ids"]))
    print("omni log_probs_num_samples:", None if omni["log_probs"] is None else len(omni["log_probs"]))
    print("omni prompt_logprobs_len:", None if omni["prompt_logprobs"] is None else len(omni["prompt_logprobs"]))
    print("omni finish_reason:", omni["finish_reason"])
    print("omni routed_experts:", omni["routed_experts"])
    for sample_idx, sample_token_ids in enumerate(omni["token_ids"]):
        print(f"omni sample[{sample_idx}] token_ids[:20]:", sample_token_ids[:20])
    for sample_idx, sample_log_probs in enumerate(omni["log_probs"]):
        if sample_log_probs is None:
            print(f"omni sample[{sample_idx}] log_probs: None")
        else:
            print(f"omni sample[{sample_idx}] log_probs[:5]:", sample_log_probs[:5])
    print("omni cot_token_ids[:20]:", omni["cot_token_ids"].reshape(-1).tolist()[:20])
    print(
        "omni token_vs_cot_len:",
        {
            "num_samples": len(omni["token_ids"]),
            "token_ids_per_sample": [len(sample_token_ids) for sample_token_ids in omni["token_ids"]],
            "cot_token_ids": int(omni["cot_token_ids"].numel()),
        },
    )
    if omni["prompt_ids"] is not None:
        print("omni prompt_ids[:5]:", omni["prompt_ids"][:5])
    if omni["prompt_logprobs"] is not None:
        print("omni prompt_logprobs[:5]:", omni["prompt_logprobs"][:5])
    print("original cot:", original["cot_text"])
    for sample_idx, cot_text in enumerate(omni["cot_text_all"]):
        print(f"omni cot[{sample_idx}]:", cot_text)
    print("omni stage0_context_used:", omni["stage0_context_used"])

    compare_tensors("pred_xyz", original["pred_xyz"], omni["pred_xyz"])
    compare_tensors("pred_rot", original["pred_rot"], omni["pred_rot"])
    print("COMPARE cot_text_equal:", original["cot_text"] == omni["cot_text"])
    original_ade = compute_min_ade(original["pred_xyz"], shared_data["ego_future_xyz"])
    omni_ade = compute_min_ade(omni["pred_xyz"], shared_data["ego_future_xyz"])
    print(f"original minADE: {original_ade:.6f} meters")
    print(f"omni minADE: {omni_ade:.6f} meters")
    print(f"minADE delta: {abs(original_ade - omni_ade):.6f} meters")

    actual_initialized = actual_request_state_dump_path(dump_dir, "request_state_initialized")
    actual_batched = actual_request_state_dump_path(dump_dir, "request_state_batched")
    if actual_initialized.is_file():
        print_dump_compare_report(
            title="REQUEST STATE INITIALIZED DUMP COMPARE",
            lhs_path=reference_dumps["initialized"],
            rhs_path=actual_initialized,
            keys=[
                "prompt_token_ids",
                "mrope_positions",
                "mrope_position_delta",
                "sampling_params",
                "additional_information",
            ],
        )
    if actual_batched.is_file():
        print_dump_compare_report(
            title="REQUEST STATE BATCHED DUMP COMPARE",
            lhs_path=reference_dumps["batched"],
            rhs_path=actual_batched,
            keys=[
                "prompt_token_ids",
                "mrope_positions",
                "mrope_position_delta",
                "additional_information",
                "extra.input_batch_prompt_token_ids",
                "extra.input_batch_num_tokens_no_spec",
            ],
        )

    actual_stage1_dump = actual_stage1_transition_dump_path(dump_dir)
    if actual_stage1_dump.is_file():
        print_dump_compare_report(
            title="NO REWRITE STAGE1 TRANSITION DUMP COMPARE",
            lhs_path=reference_stage1_dump,
            rhs_path=actual_stage1_dump,
            keys=[
                "stage0_prompt_token_ids",
                "stage0_output_token_ids",
                "stage0_sequences",
                "stage0_rope_deltas",
                "stage0_prefill_seq_len",
                "initial_noise_x0",
                "stage0_prompt_length",
                "stage0_output_length",
                "stage0_num_return_sequences",
                "stage0_attention_mask",
                "stage0_request_outputs_len",
                "stage0_completion_counts",
                "trajectory_inputs_len",
                "trajectory_input_sample_indices",
            ],
        )

    print("dump_dir:", dump_dir)


if __name__ == "__main__":
    asyncio.run(main())
