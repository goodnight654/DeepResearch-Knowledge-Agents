import json

from services.run_trace_store import RunTraceStore


def test_save_load_and_list_runs(tmp_path):
    store = RunTraceStore(base_dir=str(tmp_path))
    run1 = store.save_run("qa", {"question": "q1"}, [{"event": "qa_answer"}], {"confidence": 0.8})
    run2 = store.save_run(
        "deepresearch", {"question": "q2"}, [{"event": "synthesize"}], {"confidence": 0.9, "iterations": 2}
    )
    assert store.load_run(run1)["workflow"] == "qa"
    assert {row["run_id"] for row in store.list_runs(limit=10)} == {run1, run2}
    assert store.list_runs(workflow="deepresearch")[0]["run_id"] == run2
    assert not list(tmp_path.glob("*.tmp"))


def test_load_rejects_unsafe_ids_and_list_skips_invalid_json(tmp_path):
    store = RunTraceStore(base_dir=str(tmp_path))
    (tmp_path / "invalid.json").write_text("not json", encoding="utf-8")
    assert store.load_run("../../secret") is None
    assert store.list_runs() == []


def test_trace_files_are_valid_json(tmp_path):
    store = RunTraceStore(base_dir=str(tmp_path))
    run_id = store.save_run("qa", {"question": "中文"}, [], {"answer": "答案"})
    assert json.loads((tmp_path / f"{run_id}.json").read_text(encoding="utf-8"))["output"]["answer"] == "答案"
