import io
import json
import structlog
from backend.logging_config import configure_logging


def test_logger_emits_structured_json():
    buf = io.StringIO()
    configure_logging(stream=buf, json_output=True)
    log = structlog.get_logger("test")
    log.info("hello", session_id="abc", count=3)
    line = buf.getvalue().strip()
    payload = json.loads(line)
    assert payload["event"] == "hello"
    assert payload["session_id"] == "abc"
    assert payload["count"] == 3
    assert payload["level"] == "info"
