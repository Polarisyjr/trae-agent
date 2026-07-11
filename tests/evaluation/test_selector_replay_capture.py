from types import SimpleNamespace

from evaluation.patch_selection.trae_selector import selector_agent


class _Session:
    def execute(self, _command: str) -> str:
        return ""


class _Sandbox:
    def get_session(self) -> _Session:
        return _Session()


class _Recorder:
    def __init__(self, path: str):
        self.path = path


class _LLMClient:
    def __init__(self, _config):
        self.recorder = None

    def set_trajectory_recorder(self, recorder) -> None:
        self.recorder = recorder


def test_selector_attaches_recorder_to_llm_client(monkeypatch, tmp_path):
    monkeypatch.setattr(selector_agent, "LLMClient", _LLMClient)
    monkeypatch.setattr(selector_agent, "TrajectoryRecorder", _Recorder)
    monkeypatch.setattr(
        selector_agent,
        "tools_registry",
        {
            "bash": lambda **_kwargs: object(),
            "str_replace_based_edit_tool": lambda **_kwargs: object(),
        },
    )
    config = SimpleNamespace(
        model_provider=SimpleNamespace(provider="openai"),
        model="test-model",
    )

    agent = selector_agent.SelectorAgent(
        llm_config=config,
        sandbox=_Sandbox(),
        project_path="/testbed",
        issue_description="issue",
        trajectory_file_name=str(tmp_path / "selector.json"),
        candidate_list=[selector_agent.CandidatePatch(1, "patch", "patch", False, False)],
    )

    assert agent.llm_client.recorder is agent.trajectory_recorder
