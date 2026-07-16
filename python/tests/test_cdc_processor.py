import json

import pytest

from services.cdc_processor import CDCProcessor


@pytest.mark.parametrize(
    ("code", "operation"),
    [("c", "INSERT"), ("r", "INSERT"), ("u", "UPDATE"), ("d", "DELETE")],
)
def test_debezium_operation_codes_are_normalized(code, operation):
    message = json.dumps(
        {"payload": {"op": code, "source": {"table": "documents"}, "before": None, "after": {"id": 1}}}
    ).encode()
    event = CDCProcessor.from_kafka_message(message)
    assert event.operation == operation
    assert event.resource_path == "documents"


def test_cdc_message_must_be_an_object():
    with pytest.raises(ValueError, match="JSON object"):
        CDCProcessor.from_kafka_message(b"[]")
