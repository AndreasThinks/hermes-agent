"""Real temporary storage contract; provider bytes are fixtures, not live TTS."""
import time
from pathlib import Path

import pytest
from hermes_state import SessionDB


def test_exact_reply_late_audio_retention_and_retry(tmp_path):
    from hermes_state_audio import retain_audio, message_audio, audio_file, prune_audio
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s", source="telegram")
    first = db.append_message("s", "assistant", "first", timestamp=1)
    db.append_message("s", "user", "next", timestamp=2)
    second = db.append_message("s", "assistant", "second", timestamp=3)
    a, b = tmp_path / "a.ogg", tmp_path / "b.ogg"
    a.write_bytes(b"fixture-one")
    b.write_bytes(b"fixture-two")
    ids = retain_audio(db, "s", first, [a, b], now=10)
    assert retain_audio(db, "s", first, [a, b], now=11) == ids
    db.close()
    db = SessionDB(tmp_path / "state.db")  # completed attachments survive a fresh process handle
    refs = message_audio(db, "s", [first, second], now=12)
    assert [x["id"] for x in refs[first]] == ids
    assert second not in refs
    assert audio_file(db, ids[0], now=12).read_bytes() == a.read_bytes()
    assert db.list_sessions_rich()[0]["last_active"] >= 10
    with pytest.raises(ValueError):
        audio_file(db, "../a.ogg", now=12)
    unrelated = tmp_path / "audio-kept.ogg"
    unrelated.write_bytes(b"keep")
    prune_audio(db, now=10 + 8 * 86400)
    assert message_audio(db, "s", [first], now=10 + 8 * 86400) == {}
    assert unrelated.exists() and a.exists()
    db.close()


def test_reject_wrong_session_and_inbound(tmp_path):
    from hermes_state_audio import retain_audio
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s", source="telegram")
    user = db.append_message("s", "user", "capture")
    f = tmp_path / "capture.ogg"
    f.write_bytes(b"inbound")
    with pytest.raises(ValueError):
        retain_audio(db, "s", user, [f])
    with pytest.raises(ValueError):
        retain_audio(db, "other", user, [f])
    db.close()


def test_budget_evicts_whole_replies_and_late_audio_advances_future_clock(tmp_path, monkeypatch):
    import hermes_state_audio as audio
    monkeypatch.setattr(audio, "MAX_TOTAL_BYTES", 12)
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s", source="telegram")
    one = db.append_message("s", "assistant", "one", timestamp=100)
    two = db.append_message("s", "assistant", "two", timestamp=200)
    clip = tmp_path / "fixture.ogg"
    clip.write_bytes(b"12345")
    audio.retain_audio(db, "s", one, [clip, clip], now=10)
    audio.retain_audio(db, "s", two, [clip], now=11)
    refs = audio.message_audio(db, "s", [one, two], now=12)
    assert list(refs) == [two]
    assert refs[two][0]["created_at"] > 200
    assert db.list_sessions_rich()[0]["last_active"] >= refs[two][0]["created_at"]
    db.close()


def test_concurrent_turns_keep_their_own_receipts(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from gateway.reply_audio import _retain
    from hermes_state_audio import message_audio, audio_file
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s", source="telegram")
    rows = [db.append_message("s", "assistant", f"reply-{i}") for i in range(2)]
    paths = [tmp_path / f"part-{i}.ogg" for i in range(2)]
    for i, path in enumerate(paths):
        path.write_bytes(f"fixture-{i}".encode())
    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(_retain, {"db_path": str(db.db_path), "session_id": "s", "message_id": row}, [path])
                   for row, path in zip(rows, paths)]
        for future in futures:
            future.result()
    refs = message_audio(db, "s", rows)
    for i, row in enumerate(rows):
        assert audio_file(db, refs[row][0]["id"]).read_bytes() == f"fixture-{i}".encode()
    db.close()


def test_expired_publication_can_be_retried_without_stale_receipt(tmp_path):
    from hermes_state_audio import retain_audio, message_audio, RETENTION_SECONDS
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s", source="telegram")
    row = db.append_message("s", "assistant", "answer")
    clip = tmp_path / "fixture.ogg"
    clip.write_bytes(b"original synthesis")
    retain_audio(db, "s", row, [clip], now=10)
    now = 11 + RETENTION_SECONDS
    assert message_audio(db, "s", [row], now=now) == {}
    ids = retain_audio(db, "s", row, [clip], now=now)
    assert message_audio(db, "s", [row], now=now)[row][0]["id"] == ids[0]
    db.close()
