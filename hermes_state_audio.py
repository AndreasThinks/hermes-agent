"""Durable automatic speech, indexed by the actual assistant row, outside model context.

Files belong only to this store. Public callers use opaque IDs; source paths never
cross the API. Writes and pruning share the SessionDB transaction lock.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import tempfile
import time

RETENTION_SECONDS = 7 * 86400
MAX_FILE_BYTES = 50 * 1024 * 1024
MAX_TOTAL_BYTES = 256 * 1024 * 1024
MAX_PARTS = 32
MAX_ATTACHMENTS = 1024
MIME = {".ogg": "audio/ogg", ".opus": "audio/ogg", ".mp3": "audio/mpeg",
        ".wav": "audio/wav", ".m4a": "audio/mp4", ".aac": "audio/aac", ".flac": "audio/flac"}
_PENDING = re.compile(r"\.pending-[a-z0-9_]{8}\Z")
_ID = re.compile(r"[0-9a-f]{64}\.(?:ogg|opus|mp3|wav|m4a|aac|flac)\Z")


def _root(db):
    return Path(db.db_path).parent / "reply-audio"


def _owned_path(db, ident):
    if not isinstance(ident, str) or not _ID.fullmatch(ident):
        raise ValueError("Invalid audio ID")
    root = _root(db)
    if root.is_symlink():
        raise ValueError("Invalid audio store")
    path = root / ident
    if path.is_symlink():
        raise ValueError("Invalid audio file")
    return path


def _prune(db, conn, now):
    groups = conn.execute("""SELECT session_id, message_id, SUM(size) AS size, COUNT(*) AS count,
        MIN(expires_at) AS expires_at FROM reply_audio GROUP BY session_id, message_id
        ORDER BY MAX(created_at) DESC, message_id DESC""").fetchall()
    total = count = 0
    for group in groups:
        if (group["expires_at"] > now and total + group["size"] <= MAX_TOTAL_BYTES
                and count + group["count"] <= MAX_ATTACHMENTS):
            total += group["size"]
            count += group["count"]
            continue
        key = (group["session_id"], group["message_id"])
        for row in conn.execute("SELECT id FROM reply_audio WHERE session_id=? AND message_id=?", key):
            _owned_path(db, row["id"]).unlink(missing_ok=True)
        conn.execute("DELETE FROM reply_audio WHERE session_id=? AND message_id=?", key)
    # Recover only files created by this store if a process died between rename and commit.
    root = _root(db)
    if root.exists() and not root.is_symlink():
        known = {r[0] for r in conn.execute("SELECT id FROM reply_audio")}
        for path in root.iterdir():
            if (_ID.fullmatch(path.name) and path.name not in known) or _PENDING.fullmatch(path.name):
                path.unlink(missing_ok=True)


def prune_audio(db, *, now=None):
    now = time.time() if now is None else now
    db._execute_write(lambda conn: _prune(db, conn, now))


def retain_audio(db, session_id, message_id, paths, *, now=None):
    """Copy actual synthesis output before transport cleanup; first committed result wins.

    All parts are validated before publication. The owning row must already be
    durable. Callers without that receipt must skip attachment, never guess a row.
    """
    now = time.time() if now is None else now
    paths = list(paths)
    if not paths:
        return []
    if len(paths) > MAX_PARTS:
        raise ValueError("Too many audio parts")
    parts = []
    total_bytes = 0
    for source in paths:
        source = Path(source)
        if source.suffix.lower() not in MIME or source.is_symlink() or not source.is_file():
            raise ValueError("Invalid synthesized audio")
        with source.open("rb") as stream:
            data = stream.read(MAX_FILE_BYTES + 1)
        if not data or len(data) > MAX_FILE_BYTES:
            raise ValueError("Invalid audio size")
        total_bytes += len(data)
        if total_bytes > MAX_TOTAL_BYTES:
            raise ValueError("Reply audio exceeds retention budget")
        parts.append((source.suffix.lower(), data))

    def write(conn):
        row = conn.execute("SELECT role, tool_calls FROM messages WHERE session_id=? AND id=?",
                           (session_id, message_id)).fetchone()
        if row is None or row["role"] != "assistant" or row["tool_calls"] not in (None, "", "[]", "null"):
            raise ValueError("Audio requires an exact final assistant row")
        _prune(db, conn, now)
        existing = conn.execute("SELECT id FROM reply_audio WHERE session_id=? AND message_id=? ORDER BY part",
                                (session_id, message_id)).fetchall()
        if existing:
            return [r[0] for r in existing]
        # Discovery must move even if a transcript timestamp or heartbeat led wall time.
        tip_id = db.resolve_resume_session_id(session_id)
        previous = conn.execute("""SELECT MAX(MAX(COALESCE(last_activity_at,0),
            COALESCE((SELECT MAX(timestamp) FROM messages WHERE session_id=sessions.id),0)))
            FROM sessions WHERE id IN (?, ?)""", (session_id, tip_id)).fetchone()[0]
        published_at = max(now, previous + 0.01)
        root = _root(db)
        if root.is_symlink():
            raise ValueError("Invalid audio store")
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        ids = []
        for part, (suffix, data) in enumerate(parts):
            ident = hashlib.sha256(f"{session_id}:{message_id}:{part}:".encode() + data).hexdigest() + suffix
            target = _owned_path(db, ident)
            fd, temp = tempfile.mkstemp(prefix=".pending-", dir=root)
            try:
                with os.fdopen(fd, "wb") as out:
                    out.write(data)
                    out.flush()
                    os.fsync(out.fileno())
                os.replace(temp, target)
            finally:
                Path(temp).unlink(missing_ok=True)
            conn.execute("INSERT INTO reply_audio VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                         (ident, session_id, message_id, part, MIME[suffix], len(data), published_at, now + RETENTION_SECONDS))
            ids.append(ident)
        if os.name != "nt":
            directory = os.open(root, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        # The dashboard lists the resume tip, even when synthesis belongs to its ancestor.
        conn.execute("UPDATE sessions SET last_activity_at=MAX(COALESCE(last_activity_at,0)+0.01, ?) "
                     "WHERE id IN (?, ?)", (published_at, session_id, tip_id))
        _prune(db, conn, now)
        return ids
    return db._execute_write(write)


def message_audio(db, session_id, message_ids, *, now=None, include_ancestors=False):
    now = time.time() if now is None else now
    wanted = set(message_ids)
    result = {}
    session_ids = db._resume_lineage_ids(session_id) if include_ancestors else [session_id]
    placeholders = ",".join("?" for _ in session_ids)
    rows = db._read_all(f"SELECT * FROM reply_audio WHERE session_id IN ({placeholders}) "
                        "AND expires_at>? ORDER BY message_id, part", (*session_ids, now))
    groups = {}
    for row in rows:
        if row["message_id"] in wanted:
            groups.setdefault(row["message_id"], []).append(row)
    for message_id, parts in groups.items():
        try:
            available = all(_owned_path(db, row["id"]).is_file() for row in parts)
        except ValueError:
            available = False
        if available:
            result[message_id] = [{key: row[key] for key in
                ("id", "mime_type", "created_at", "expires_at", "part")} for row in parts]
    return result


def audio_file(db, ident, *, now=None):
    path = _owned_path(db, ident)
    now = time.time() if now is None else now
    row = db._read_one("SELECT id FROM reply_audio WHERE id=? AND expires_at>?", (ident, now))
    if row is None or not path.is_file():
        raise FileNotFoundError("Audio expired or unavailable")
    return path


def audio_messages(db, session_id, *, now=None, include_ancestors=False):
    """Retained reply rows outside a transcript page, bounded by the store's attachment cap."""
    now = time.time() if now is None else now
    session_ids = db._resume_lineage_ids(session_id) if include_ancestors else [session_id]
    placeholders = ",".join("?" for _ in session_ids)
    rows = db._read_all(f"""SELECT m.* FROM messages m
        WHERE m.session_id IN ({placeholders}) AND (m.active=1 OR m.compacted=1)
        AND EXISTS (SELECT 1 FROM reply_audio a WHERE a.message_id=m.id AND a.expires_at>?)
        ORDER BY m.id""", (*session_ids, now))
    return [db._row_to_message_dict(row, warn_context="reply audio", summary_flag=True) for row in rows]


def prune_live_reply_audio():
    """Hourly gateway maintenance, borrowing each live profile's existing database."""
    from hermes_state_registry import borrow_live_shared_session_dbs
    with borrow_live_shared_session_dbs() as databases:
        for db in databases:
            prune_audio(db)
