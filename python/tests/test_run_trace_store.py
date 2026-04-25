import sys
import tempfile
import types

pydantic_settings = types.ModuleType("pydantic_settings")


class _BaseSettings:
    def __init__(self, **kwargs):
        for name, value in self.__class__.__dict__.items():
            if name.startswith("_") or callable(value) or isinstance(value, property):
                continue
            setattr(self, name, kwargs.get(name, value))


pydantic_settings.BaseSettings = _BaseSettings
sys.modules.setdefault("pydantic_settings", pydantic_settings)

from services.run_trace_store import RunTraceStore


def test_save_load_and_list_runs():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = RunTraceStore(base_dir=tmpdir)

        run1 = store.save_run(
            workflow="qa",
            input_payload={"question": "q1"},
            trace=[{"event": "qa_answer"}],
            output={"question": "q1", "confidence": 0.8},
        )
        run2 = store.save_run(
            workflow="deepsearch",
            input_payload={"question": "q2"},
            trace=[{"event": "plan"}, {"event": "synthesize"}],
            output={"question": "q2", "confidence": 0.9, "iterations": 2},
        )

        loaded = store.load_run(run1)
        assert loaded is not None
        assert loaded["workflow"] == "qa"

        rows = store.list_runs(limit=10)
        assert len(rows) == 2
        ids = {row["run_id"] for row in rows}
        assert run1 in ids and run2 in ids

        deep_rows = store.list_runs(limit=10, workflow="deepsearch")
        assert len(deep_rows) == 1
        assert deep_rows[0]["run_id"] == run2
