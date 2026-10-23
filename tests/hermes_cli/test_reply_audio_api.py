"""Authenticated HTTP with real temporary SQLite + actual file bytes, no provider calls."""
import time

from fastapi.testclient import TestClient
from hermes_state import SessionDB


def test_late_attachment_api_auth_and_safe_retrieval(tmp_path, monkeypatch):
    from hermes_cli import web_server as web
    from hermes_cli import web_server_sessions as sessions
    from hermes_state_audio import retain_audio
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s", source="telegram")
    row = db.append_message("s", "assistant", "answer")
    monkeypatch.setattr(sessions, "_open_session_db_for_profile", lambda *a, **kw: SessionDB(db.db_path, read_only=kw.get("read_only", False)))
    client = TestClient(web.app)
    assert client.get("/api/reply-audio/missing.ogg").status_code == 401
    client.headers[web._SESSION_HEADER_NAME] = web._SESSION_TOKEN
    before = client.get("/api/sessions/s/messages").json()
    assert before["messages"][0]["audio_attachments"] == []
    clip = tmp_path / "fixture.ogg"
    clip.write_bytes(b"fixture-audio")
    ident = retain_audio(db, "s", row, [clip])[0]
    after = client.get("/api/sessions/s/messages").json()
    assert after["messages"][0]["id"] == row
    assert after["messages"][0]["audio_attachments"][0]["id"] == ident
    client.headers.pop(web._SESSION_HEADER_NAME)
    assert client.get(f"/api/reply-audio/{ident}").status_code == 401
    client.headers[web._SESSION_HEADER_NAME] = web._SESSION_TOKEN
    response = client.get(f"/api/reply-audio/{ident}")
    assert response.status_code == 200 and response.content == b"fixture-audio"
    assert response.headers["content-type"] == "audio/ogg"
    db.append_message("s", "user", "next")
    db.append_message("s", "assistant", "later")
    page = client.get("/api/sessions/s/messages?limit=1&order=latest&include_audio=true").json()
    assert page["messages"][0]["id"] != row
    assert page["audio_messages"][0]["id"] == row
    assert page["audio_messages"][0]["audio_attachments"][0]["id"] == ident
    assert client.get("/api/reply-audio/%2e%2e%2fsecret").status_code == 404
    # Substituting a symlink cannot turn this route into arbitrary file access.
    target = tmp_path / "reply-audio" / ident
    target.unlink()
    target.symlink_to(clip)
    assert client.get(f"/api/reply-audio/{ident}").status_code == 404
    db.close()
