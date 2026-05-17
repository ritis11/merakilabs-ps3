from datetime import datetime, timezone
from pathlib import Path
from backend.session.models import Message, Session
from backend.session.store import SessionManager


def _msg(role, content, ts_offset=0):
    return Message(
        role=role,
        content=content,
        timestamp=datetime.now(timezone.utc),
    )


def test_create_and_get_session(tmp_path):
    sm = SessionManager(dump_path=str(tmp_path / "sessions.json"), history_cap=10)
    s = sm.create_session()
    assert s.session_id
    fetched = sm.get_session(s.session_id)
    assert fetched is not None
    assert fetched.session_id == s.session_id


def test_append_and_trim_history_fifo(tmp_path):
    sm = SessionManager(dump_path=str(tmp_path / "s.json"), history_cap=4)
    s = sm.create_session()
    for i in range(6):
        sm.append_message(s.session_id, _msg("user", f"u{i}"))
        sm.append_message(s.session_id, _msg("assistant", f"a{i}"))
    fetched = sm.get_session(s.session_id)
    # cap=4 means keep last 4 messages, drop the older ones
    assert len(fetched.messages) == 4
    assert fetched.messages[0].content == "u4"
    assert fetched.messages[-1].content == "a5"


def test_dump_and_load_round_trip(tmp_path):
    path = tmp_path / "sessions.json"
    sm = SessionManager(dump_path=str(path), history_cap=10)
    s = sm.create_session()
    sm.append_message(s.session_id, _msg("user", "hello"))
    sm.add_uploaded_doc(s.session_id, "zomato_fy24.pdf")
    sm.set_pa_messages_json(s.session_id, '[{"kind":"request","parts":[]}]')
    sm.dump()
    assert path.exists()

    sm2 = SessionManager(dump_path=str(path), history_cap=10)
    sm2.load()
    fetched = sm2.get_session(s.session_id)
    assert fetched is not None
    assert fetched.messages[0].content == "hello"
    assert fetched.uploaded_docs == ["zomato_fy24.pdf"]
    # Multi-turn agent state survives the dump/load round-trip.
    assert fetched.pa_messages_json == '[{"kind":"request","parts":[]}]'


def test_set_pa_messages_json_unknown_session_raises(tmp_path):
    sm = SessionManager(dump_path=str(tmp_path / "s.json"), history_cap=10)
    import pytest
    with pytest.raises(KeyError):
        sm.set_pa_messages_json("no-such-session", "[]")


def test_delete_session(tmp_path):
    sm = SessionManager(dump_path=str(tmp_path / "s.json"), history_cap=10)
    s = sm.create_session()
    sm.delete_session(s.session_id)
    assert sm.get_session(s.session_id) is None


def test_load_missing_file_is_noop(tmp_path):
    sm = SessionManager(dump_path=str(tmp_path / "missing.json"), history_cap=10)
    sm.load()  # should not raise
    assert sm.get_session("anything") is None
