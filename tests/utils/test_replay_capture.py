from __future__ import annotations

import json
from pathlib import Path

from trae_agent.tools.base import ToolResult
from trae_agent.utils.llm_clients.llm_basics import LLMMessage, LLMResponse, LLMUsage
from trae_agent.utils.trajectory_recorder import TrajectoryRecorder


def test_trajectory_recorder_preserves_replay_request_and_tool_execution(
    tmp_path: Path,
) -> None:
    path = tmp_path / "trajectory.json"
    recorder = TrajectoryRecorder(str(path))
    recorder.start_recording("task", "vllm", "model", 3)
    recorder.record_llm_interaction(
        [LLMMessage(role="user", content="hello")],
        LLMResponse(
            content="done",
            usage=LLMUsage(input_tokens=7, output_tokens=3),
            model="model",
            finish_reason="length",
            replay_endpoint_api="responses",
            replay_request_body={
                "input": [{"role": "user", "content": "hello"}],
                "max_output_tokens": 10,
            },
            replay_started_at_ns=100,
            replay_ended_at_ns=200,
            replay_attempt_count=1,
        ),
        provider="vllm",
        model="model",
    )
    recorder.record_agent_step(
        1,
        "tool",
        tool_results=[
            ToolResult(
                call_id="call-1",
                name="bash",
                success=True,
                result="ok",
                exit_code=0,
                executed_command="printf ok",
                executor="docker",
                started_at_ns=210,
                ended_at_ns=220,
            )
        ],
    )

    trajectory = json.loads(path.read_text())
    wire = trajectory["llm_interactions"][0]["wire_request"]
    assert wire["endpoint_api"] == "responses"
    assert wire["body"]["input"][0]["content"] == "hello"
    assert wire["attempt_count"] == 1
    result = trajectory["agent_steps"][0]["tool_results"][0]
    assert result["executed_command"] == "printf ok"
    assert result["exit_code"] == 0
    assert result["started_at_ns"] == 210
