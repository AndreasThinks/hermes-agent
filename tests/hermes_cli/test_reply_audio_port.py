"""0.22.2 dashboard lineage, schema healing and profile audio isolation."""
import sqlite3

import pytest
from fastapi.testclient import TestClient

from hermes_state import SessionDB
from hermes_state_audio import retain_audio


@pytest.fixture
def client(monkeypatch, tmp_path):
    from hermes_cli import web_server as web
    from hermes_cli import web_server_sessions as sessions
    monkeypatch.setattr(sessions, "_session_db_path_for_profile",
                        lambda profile: tmp_path / (profile or "default") / "state.db")
    client = TestClient(web.app)
    client.headers["Authorization"] = f"Bearer {web._SESSION_TOKEN}"
    return client


def test_late_audio_in_compression_ancestor_keeps_exact_owner(client, tmp_path):
    db = SessionDB(tmp_path / "default" / "state.db")
    db.create_session("root", source="telegram")
    row = db.append_message("root", "assistant", "answer")
    db.end_session("root", end_reason="compression")
    db.create_session("tip", source="telegram", parent_session_id="root")
    newer = db.append_message("tip", "assistant", "next answer")
    before_activity = db.list_sessions_rich(order_by_last_active=True)[0]["last_active"]
    before = client.get("/api/sessions/tip/messages?limit=1&order=latest&include_audio=true")
    assert before.status_code == 200
    assert before.json()["messages"][0]["id"] == newer
    clip = tmp_path / "fixture.wav"
    clip.write_bytes(b"original bytes")
    ident = retain_audio(db, "root", row, [clip])[0]
    assert db.list_sessions_rich(order_by_last_active=True)[0]["last_active"] > before_activity
    page = client.get("/api/sessions/tip/messages?limit=1&order=latest&include_audio=true").json()
    assert page["messages"][0]["audio_attachments"] == []
    assert [(m["id"], m["session_id"]) for m in page["audio_messages"]] == [(row, "root")]
    assert page["audio_messages"][0]["audio_attachments"][0]["id"] == ident
    full = client.get("/api/sessions/root/messages").json()
    assert full["session_id"] == "tip"
    assert next(m for m in full["messages"] if m["id"] == row)["audio_attachments"][0]["id"] == ident
    db.close()


def test_dashboard_heals_existing_store_missing_audio_table(client, tmp_path):
    path = tmp_path / "default" / "state.db"
    db = SessionDB(path)
    db.create_session("s", source="telegram")
    db.append_message("s", "assistant", "old answer")
    db.close()
    with sqlite3.connect(path) as conn:
        conn.execute("DROP TABLE reply_audio")
    response = client.get("/api/sessions/s/messages")
    assert response.status_code == 200
    assert response.json()["messages"][0]["audio_attachments"] == []
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM reply_audio").fetchone()[0] == 0


def test_audio_download_stays_in_selected_profile(client, tmp_path):
    clip = tmp_path / "fixture.ogg"
    clip.write_bytes(b"profile alpha audio")
    for profile in ("alpha", "beta"):
        db = SessionDB(tmp_path / profile / "state.db")
        db.create_session("s", source="telegram")
        row = db.append_message("s", "assistant", "answer")
        if profile == "alpha":
            ident = retain_audio(db, "s", row, [clip])[0]
        db.close()
    assert client.get(f"/api/reply-audio/{ident}?profile=alpha").content == clip.read_bytes()
    assert client.get(f"/api/reply-audio/{ident}?profile=beta").status_code == 404
    assert client.get(f"/api/reply-audio/{ident}?token=wrong").status_code == 404
    client.headers.clear()
    assert client.get(f"/api/reply-audio/{ident}?profile=alpha").status_code == 401
