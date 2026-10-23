"""Real DB/files through both voice delivery paths; synthesis/transport are fixtures."""
import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from gateway.config import Platform
from hermes_state import SessionDB
from hermes_state_audio import message_audio, audio_file
from tests.gateway.test_base_auto_tts_output_format import _DummyAdapter, _make_voice_event, _hold_typing
from tests.gateway.test_auto_voice_reply_format import _make_runner
from gateway.session import build_session_key


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["runner", "base"])
@pytest.mark.parametrize("succeeds", [True, False])
async def test_exact_split_files_survive_cleanup_and_intervening_turn(tmp_path, path, succeeds):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s", source="telegram")
    row = db.append_message("s", "assistant", "answer")
    event = _make_voice_event(Platform.TELEGRAM)
    event._reply_audio_owner = {"db_path": str(db.db_path), "session_id": "s", "message_id": row}
    adapter = _DummyAdapter(Platform.TELEGRAM)
    adapter._keep_typing = _hold_typing()
    adapter._should_auto_tts_for_chat = lambda _: True
    adapter.play_tts = AsyncMock()
    adapter.send_voice = AsyncMock()
    adapter.set_message_handler(lambda _: asyncio.sleep(0, result="answer"))
    files = [tmp_path / "one.ogg", tmp_path / "two.ogg"]

    def synth(**_):
        # A later turn lands while the original reply is being synthesized.
        db.append_message("s", "user", "later")
        db.append_message("s", "assistant", "other answer")
        for i, f in enumerate(files):
            f.write_bytes(f"fixture-{i}".encode())
        return json.dumps({"success": succeeds, "file_paths": [str(f) for f in files]})

    with patch("tools.tts_tool.text_to_speech_tool", side_effect=synth), patch(
            "tools.tts_tool.check_tts_requirements", return_value=True):
        if path == "runner":
            runner = _make_runner()
            runner.adapters[Platform.TELEGRAM] = adapter
            await runner._send_voice_reply(event, "answer")
        else:
            await adapter._process_message_background(event, build_session_key(event.source))
    refs = message_audio(db, "s", [m["id"] for m in db.get_messages("s")])
    if succeeds:
        assert list(refs) == [row]
        assert [audio_file(db, ref["id"]).read_bytes() for ref in refs[row]] == [b"fixture-0", b"fixture-1"]
        assert not any(f.exists() for f in files)
    else:
        assert refs == {}
        adapter.play_tts.assert_not_awaited()
        adapter.send_voice.assert_not_awaited()
    db.close()

@pytest.mark.asyncio
async def test_failed_base_delivery_retains_all_parts_and_still_sends_text(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s", source="telegram")
    row = db.append_message("s", "assistant", "answer")
    event = _make_voice_event(Platform.TELEGRAM)
    event._reply_audio_owner = {"db_path": str(db.db_path), "session_id": "s", "message_id": row}
    adapter = _DummyAdapter(Platform.TELEGRAM)
    adapter._keep_typing = _hold_typing()
    adapter._should_auto_tts_for_chat = lambda _: True
    adapter.play_tts = AsyncMock(side_effect=OSError("fixture delivery failure"))
    adapter.set_message_handler(lambda _: asyncio.sleep(0, result="answer"))
    files = [tmp_path / "one.ogg", tmp_path / "two.ogg"]
    for f in files:
        f.write_bytes(b"fixture")
    with patch.object(adapter, "_synthesize_auto_tts", return_value=([str(f) for f in files], None)):
        await adapter._process_message_background(event, build_session_key(event.source))
    assert len(message_audio(db, "s", [row])[row]) == 2
    assert not any(f.exists() for f in files)
    assert any(item["content"] == "answer" for item in adapter.sent)
    db.close()
