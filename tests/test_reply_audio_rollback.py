"""Filesystem collection must never get ahead of committed attachment eviction."""
import pytest

from hermes_state import SessionDB
import hermes_state_audio as audio


@pytest.mark.parametrize("operation", ["maintenance", "publication"])
def test_eviction_rollback_preserves_committed_audio(tmp_path, monkeypatch, operation):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s", source="telegram")
    old = db.append_message("s", "assistant", "old")
    new = db.append_message("s", "assistant", "new")
    clip = tmp_path / "fixture.ogg"
    clip.write_bytes(b"fixture")
    ids = audio.retain_audio(db, "s", old, [clip], now=10)
    original_write = db._execute_write

    def fail_before_commit(fn, *args, **kwargs):
        def failed(conn):
            fn(conn)
            raise RuntimeError("injected pre-commit failure")
        return original_write(failed, *args, **kwargs)

    monkeypatch.setattr(db, "_execute_write", fail_before_commit)
    monkeypatch.setattr(audio, "MAX_TOTAL_BYTES", len(clip.read_bytes()) if operation == "publication" else 0)
    with pytest.raises(RuntimeError, match="pre-commit"):
        if operation == "publication":
            audio.retain_audio(db, "s", new, [clip], now=11)
        else:
            audio.prune_audio(db, now=11)
    assert audio.audio_file(db, ids[0], now=12).read_bytes() == b"fixture"
    assert audio.message_audio(db, "s", [old], now=12)[old][0]["id"] == ids[0]
    assert audio.message_audio(db, "s", [new], now=12) == {}
    db.close()


def test_committed_eviction_orphans_are_recovered_later(tmp_path, monkeypatch):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s", source="telegram")
    old = db.append_message("s", "assistant", "old")
    new = db.append_message("s", "assistant", "new")
    clip = tmp_path / "fixture.ogg"
    clip.write_bytes(b"fixture")
    ids = audio.retain_audio(db, "s", old, [clip], now=10)
    old_path = audio.audio_file(db, ids[0], now=11)
    original_write = db._execute_write
    calls = 0

    def stop_after_commit(fn, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected stop before collection")
        return original_write(fn, *args, **kwargs)

    monkeypatch.setattr(db, "_execute_write", stop_after_commit)
    monkeypatch.setattr(audio, "MAX_TOTAL_BYTES", len(clip.read_bytes()))
    with pytest.raises(RuntimeError, match="before collection"):
        audio.retain_audio(db, "s", new, [clip], now=12)
    assert old_path.exists()  # safe orphan, not a committed row referencing missing bytes
    assert audio.message_audio(db, "s", [old], now=13) == {}
    assert audio.message_audio(db, "s", [new], now=13)[new]
    monkeypatch.setattr(db, "_execute_write", original_write)
    audio.prune_audio(db, now=14)
    assert not old_path.exists()
    assert audio.message_audio(db, "s", [new], now=14)[new]
    db.close()
