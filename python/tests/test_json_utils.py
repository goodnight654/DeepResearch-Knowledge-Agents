from utils.json_utils import coerce_float, parse_json_object


def test_parse_json_object_accepts_fence_and_surrounding_text():
    assert parse_json_object('result:\n```json\n{"ok": true}\n```\ndone') == {"ok": True}


def test_parse_json_object_rejects_non_object_and_coerce_float_rejects_nan():
    assert parse_json_object("[1, 2]") is None
    assert coerce_float("nan", 0.4) == 0.4
