"""Carry a persisted turn receipt across gateway delivery and profile teardown."""
import asyncio
import contextlib
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def bind_reply_audio(result, agent):
    from hermes_state import SessionDB
    db = getattr(agent, "_session_db", None)
    row_id = result.get("final_message_id")
    if (isinstance(db, SessionDB) and type(row_id) is int
            and not result.get("interrupted") and not result.get("failed")
            and not result.get("error") and result.get("completed") is not False):
        result["_reply_audio_owner"] = {
            "db_path": str(db.db_path), "session_id": result["session_id"], "message_id": row_id}


def _retain(owner, paths):
    from hermes_state import SessionDB
    from hermes_state_audio import retain_audio
    db = SessionDB(Path(owner["db_path"]))
    try:
        return retain_audio(db, owner["session_id"], owner["message_id"], paths)
    finally:
        db.close()


async def retain_owned_audio(owner, paths):
    if not isinstance(owner, dict) or not paths:
        return
    try:
        await asyncio.to_thread(_retain, owner, paths)
    except Exception:  # health: allow BLE001 -- retention must preserve delivery; tracebacks can expose media paths
        # Retention failure must not change Telegram text/voice delivery. No paths/content in logs.
        logger.warning("Automatic reply audio could not be retained")


async def retain_event_audio(event, paths):
    await retain_owned_audio(getattr(event, "_reply_audio_owner", None), paths)


async def finalize_streaming_audio(turn_ctx, adapter) -> None:
    """Finalize the streaming-TTS consumer on the outer event-loop thread (covers early returns
    from run_sync). On drain timeout abort to free the task — audible streams keep whole-file
    suppression, silent streams stay eligible for the whole-file fallback."""
    _stts = turn_ctx.streaming_tts_consumer_holder[0]
    if _stts is None:
        return
    _stts.finish()
    try:
        await _stts.wait_complete(timeout=10.0)
    except Exception as _stts_done_err:  # health: allow BLE001 -- keep the existing bounded-abort fallback for adapter failures
        logger.debug("streaming TTS wait_complete error: %s", _stts_done_err)
    if not _stts.done:
        _stts.abort("streaming TTS finalisation timeout")
        await _stts.wait_complete(timeout=2.0)
    from gateway.streaming_tts_consumer import StreamingTTSConsumer
    if isinstance(_stts, StreamingTTSConsumer):
        result = turn_ctx.result_holder[0] or {}
        await _stts.retain_audio(result.get("_reply_audio_owner"))
    if _stts.suppress_whole_file and adapter is not None:
        _mark_turn = getattr(adapter, "_mark_streaming_tts_completed_turn", None)
        if callable(_mark_turn):
            _mark_turn(turn_ctx.session_key, turn_ctx.run_generation)


async def deliver_auto_tts(adapter, event, text, paths, requested_path, metadata, record_delivery):
    """Retain all synthesis parts before transport cleanup, preserving text fallback."""
    await retain_event_audio(event, paths)
    caption_delivered = False
    try:
        for index, path in enumerate(paths):
            try:
                caption_delivered |= await adapter._play_tts_file(
                    event, text, path, index == 0, metadata, record_delivery)
            except Exception:  # health: allow BLE001 -- transport failure must preserve text; exception text may contain reply content
                logger.warning("[%s] Automatic voice delivery failed; preserving text reply", adapter.name)
    finally:
        for path in paths or ([requested_path] if requested_path is not None else []):
            with contextlib.suppress(OSError):
                os.remove(path)
    return caption_delivered
