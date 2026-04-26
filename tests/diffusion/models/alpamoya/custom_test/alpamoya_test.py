import asyncio
from pathlib import Path

from vllm_omni.entrypoints.async_omni import AsyncOmni
from common import (
    MODEL_PATH,
    build_prompt_from_messages,
    build_stage0_only_sampling_params,
    build_stage0_only_yaml,
    build_alpamayo_stage0_tokenizer,
    load_shared_data,
)


async def main():
    stage0_yaml_path = build_stage0_only_yaml()
    data, frames, prompt_messages = load_shared_data()
    tokenizer = build_alpamayo_stage0_tokenizer(MODEL_PATH)
    prompt, _ = build_prompt_from_messages(
        prompt_messages,
        data=data,
        frames=frames,
        tokenizer=tokenizer,
    )
    stage0_params = build_stage0_only_sampling_params()
    omni = AsyncOmni(
        model=MODEL_PATH,
        stage_configs_path=stage0_yaml_path,
    )

    try:
        final_output = None
        async for out in omni.generate(
            prompt=prompt,
            request_id="alpamayo-e2e-test",
            sampling_params_list=[stage0_params],
        ):
            final_output = out

        if final_output is None:
            raise RuntimeError("no final output")

        request_output = final_output.request_output
        if request_output is None:
            raise RuntimeError("stage0 produced no request_output")
        if not request_output.outputs:
            raise RuntimeError("stage0 produced no completion outputs")

        output = request_output.outputs[0]
        output_token_ids = list(output.token_ids or [])
        cot_text = tokenizer.decode(output_token_ids, skip_special_tokens=False)

        print("stage0_final_output_type:", final_output.final_output_type)
        print("stage0_prompt_len:", len(request_output.prompt_token_ids or []))
        print("stage0_output_len:", len(output_token_ids))
        print("stage0_finish_reason:", getattr(output, "finish_reason", None))
        print("stage0_prompt_contains_fused_history:", 155684 not in (request_output.prompt_token_ids or []))
        print("\nStage0 text:\n", cot_text)
    finally:
        omni.shutdown()
        Path(stage0_yaml_path).unlink(missing_ok=True)


if __name__ == "__main__":
    asyncio.run(main())
