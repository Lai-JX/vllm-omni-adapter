from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import vllm_omni  # noqa: F401 - import for side effects
from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_next_step_after_special_token_uses_confirmed_computed_tokens() -> None:
    sched = OmniARScheduler.__new__(OmniARScheduler)
    sched.kv_transfer_criteria = {"type": "next_step_after_special_token", "token_id": 155681}
    sched._request_omits_kv_transfer_to_next_stage = lambda request: False
    sched._get_confirmed_num_computed_tokens = lambda request: request.num_computed_tokens - request.num_output_placeholders
    sched._mark_request_for_kv_transfer = Mock()
    sched.next_step_transfer_armed_requests = {"req-1"}
    sched.transfer_triggered_requests = set()
    sched.waiting_for_transfer_free = set()
    sched.requests_needing_kv_transfer = {}
    sched.pending_stop_after_extraction = set()

    request = SimpleNamespace(
        request_id="req-1",
        num_computed_tokens=10,
        num_output_placeholders=3,
        num_prompt_tokens=4,
    )

    should_stop = OmniARScheduler._process_kv_transfer_trigger(sched, request, [])

    assert should_stop is False
    sched._mark_request_for_kv_transfer.assert_called_once_with("req-1", 7)
    assert "req-1" not in sched.next_step_transfer_armed_requests
    assert "req-1" in sched.transfer_triggered_requests
