"""Turn receipt and outer-loop streaming publication contracts."""
import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gateway.reply_audio import bind_reply_audio, finalize_streaming_audio
from gateway.streaming_tts_consumer import StreamingTTSConsumer
from hermes_state import SessionDB
from hermes_state_audio import message_audio, audio_file
from tests.gateway.test_streaming_tts_consumer import FakeStreamer, FakeVoiceAdapter


@pytest.mark.parametrize("state", ["completed", "interrupted", "failed", "missing_receipt", "error"])
@pytest.mark.asyncio
async def test_outer_finalization_publishes_only_successful_turn_receipt(tmp_path, monkeypatch, state):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s", source="discord")
    row = db.append_message("s", "assistant", "answer")
    result = {"session_id": "s", "final_message_id": row, "completed": True}
    if state == "missing_receipt":
        result.pop("final_message_id")
    elif state != "completed":
        result[state] = True
    bind_reply_audio(result, SimpleNamespace(_session_db=db))
    monkeypatch.setattr("tools.tts_streaming.resolve_streaming_provider", lambda _: FakeStreamer())
    adapter = FakeVoiceAdapter()
    adapter._mark_streaming_tts_completed_turn = Mock()
    consumer = StreamingTTSConsumer(adapter, "chat", {}, asyncio.get_running_loop())
    consumer.start()
    consumer.on_delta("Hello.")
    ctx = SimpleNamespace(streaming_tts_consumer_holder=[consumer], result_holder=[result],
                          session_key="key", run_generation=7)
    await finalize_streaming_audio(ctx, adapter)
    refs = message_audio(db, "s", [row])
    if state == "completed":
        assert audio_file(db, refs[row][0]["id"]).read_bytes().startswith(b"RIFF")
    else:
        assert refs == {}
    assert consumer._recording.closed
    adapter._mark_streaming_tts_completed_turn.assert_called_once_with("key", 7)
    db.close()
