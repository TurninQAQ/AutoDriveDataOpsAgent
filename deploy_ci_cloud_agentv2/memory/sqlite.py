"""Lightweight SQLite durability for audit, checkpoints and WRITE capabilities."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .codec import CheckpointCodecError, decode, encode
from ..agent.events import Event, EventIntegrityError, EventProvenance
from ..agent.immutable import canonical_snapshot, thaw_value
from ..agent.state import AgentState
from ..safety.approval import (
    ApprovalDecision,
    ApprovalRecord,
    ApprovalRecordConflict,
    _same_approval_authority,
)
from ..safety.locks import (
    ExecutionClaim,
    ExecutionClaimAlreadyExists,
    MutationAttemptAlreadyConsumed,
)
from ..safety.write_transaction import WriteTransaction


class CheckpointIntegrityError(RuntimeError):
    pass


class DurableConcurrencyError(RuntimeError):
    """A stale worker attempted to extend a thread from a non-tail event."""


def _connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")
    return conn


def _thread_tail(conn: sqlite3.Connection, thread_id: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT event_id, sequence_no FROM events WHERE thread_id=? ORDER BY sequence_no DESC LIMIT 1",
        (thread_id,),
    ).fetchone()


def _assert_expected_tail(
    conn: sqlite3.Connection,
    thread_id: str,
    expected_event_id: str | None,
) -> None:
    tail = _thread_tail(conn, thread_id)
    actual = None if tail is None else tail["event_id"]
    if actual != expected_event_id:
        raise DurableConcurrencyError(
            f"stale durable thread tail: expected {expected_event_id!r}, found {actual!r}"
        )


def _checkpoint_payload(state: AgentState, last_event_id: str) -> tuple[str, str, str, str]:
    candidate = dict(state)
    candidate["last_event_id"] = last_event_id
    encoded = encode(candidate)
    text = json.dumps(encoded, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return candidate["thread_id"], last_event_id, text, digest


def _upsert_checkpoint(
    conn: sqlite3.Connection,
    state: AgentState,
    last_event_id: str,
) -> None:
    thread_id, event_id, text, digest = _checkpoint_payload(state, last_event_id)
    conn.execute(
        "INSERT INTO checkpoints(thread_id,last_event_id,state_json,digest,updated_at) VALUES(?,?,?,?,?) "
        "ON CONFLICT(thread_id) DO UPDATE SET "
        "last_event_id=excluded.last_event_id,state_json=excluded.state_json,"
        "digest=excluded.digest,updated_at=excluded.updated_at",
        (thread_id, event_id, text, digest, datetime.now(timezone.utc).isoformat()),
    )


class SQLiteEventStore:
    """Append-only event log with duplicate-id integrity and optional tail CAS."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._lock = threading.RLock()
        self._init()

    def _connect(self):
        return _connect(self.path)

    def _init(self):
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS events(
                sequence_no INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                request_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                causation_id TEXT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                provenance_json TEXT NOT NULL
            )"""
            )

    @staticmethod
    def _serialized(payload: dict[str, Any], provenance: EventProvenance) -> tuple[str, str]:
        canonical_payload = canonical_snapshot(dict(payload))
        payload_json = json.dumps(
            thaw_value(canonical_payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        provenance_json = json.dumps(
            asdict(provenance), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        return payload_json, provenance_json

    def _append_in_connection(
        self,
        conn: sqlite3.Connection,
        *,
        event_type: str,
        request_id: str,
        thread_id: str,
        payload: dict[str, Any],
        provenance: EventProvenance,
        causation_id: str | None,
        event_id: str,
    ) -> Event:
        payload_json, provenance_json = self._serialized(payload, provenance)
        row = conn.execute("SELECT * FROM events WHERE event_id=?", (event_id,)).fetchone()
        if row is not None:
            if not self._row_matches(
                row, event_type, request_id, thread_id, causation_id, payload_json, provenance_json
            ):
                raise EventIntegrityError(
                    f"event_id {event_id} was already appended with different content"
                )
            return self._event(row)
        timestamp = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO events(event_id,request_id,thread_id,causation_id,timestamp,event_type,payload_json,provenance_json) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (
                event_id,
                request_id,
                thread_id,
                causation_id,
                timestamp,
                event_type,
                payload_json,
                provenance_json,
            ),
        )
        row = conn.execute("SELECT * FROM events WHERE event_id=?", (event_id,)).fetchone()
        return self._event(row)

    def append(
        self,
        *,
        event_type,
        request_id,
        thread_id,
        payload,
        provenance,
        causation_id=None,
        event_id=None,
    ):
        stable_id = event_id or f"evt_{uuid.uuid4().hex}"
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                event = self._append_in_connection(
                    conn,
                    event_type=event_type,
                    request_id=request_id,
                    thread_id=thread_id,
                    payload=payload,
                    provenance=provenance,
                    causation_id=causation_id,
                    event_id=stable_id,
                )
                conn.execute("COMMIT")
                return event
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def append_with_checkpoint(
        self,
        checkpointer: "SQLiteCheckpointer",
        state: AgentState,
        *,
        event_type: str,
        request_id: str,
        thread_id: str,
        payload: dict[str, Any],
        provenance: EventProvenance,
        causation_id: str | None = None,
        event_id: str | None = None,
    ) -> Event:
        """Atomically extend one thread event tail and its checkpoint projection."""
        if not isinstance(checkpointer, SQLiteCheckpointer) or checkpointer.path != self.path:
            raise CheckpointIntegrityError(
                "checkpoint and event store must share one SQLite durability boundary"
            )
        stable_id = event_id or f"evt_{uuid.uuid4().hex}"
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                existing = conn.execute(
                    "SELECT * FROM events WHERE event_id=?", (stable_id,)
                ).fetchone()
                if existing is None:
                    _assert_expected_tail(conn, thread_id, causation_id)
                    event = self._append_in_connection(
                        conn,
                        event_type=event_type,
                        request_id=request_id,
                        thread_id=thread_id,
                        payload=payload,
                        provenance=provenance,
                        causation_id=causation_id,
                        event_id=stable_id,
                    )
                else:
                    payload_json, provenance_json = self._serialized(payload, provenance)
                    if not self._row_matches(
                        existing,
                        event_type,
                        request_id,
                        thread_id,
                        causation_id,
                        payload_json,
                        provenance_json,
                    ):
                        raise EventIntegrityError(
                            f"event_id {stable_id} was already appended with different content"
                        )
                    tail = _thread_tail(conn, thread_id)
                    if tail is None or tail["event_id"] != stable_id:
                        raise DurableConcurrencyError(
                            "idempotent event replay is stale relative to the current durable tail"
                        )
                    event = self._event(existing)
                _upsert_checkpoint(conn, state, event.event_id)
                conn.execute("COMMIT")
                return event
            except Exception:
                conn.execute("ROLLBACK")
                raise

    @staticmethod
    def _row_matches(
        row, event_type, request_id, thread_id, causation_id, payload_json, provenance_json
    ):
        return (
            row["event_type"] == event_type
            and row["request_id"] == request_id
            and row["thread_id"] == thread_id
            and row["causation_id"] == causation_id
            and row["payload_json"] == payload_json
            and row["provenance_json"] == provenance_json
        )

    def _event(self, row):
        prov = EventProvenance(**json.loads(row["provenance_json"]))
        return Event(
            event_id=row["event_id"],
            sequence_no=row["sequence_no"],
            request_id=row["request_id"],
            thread_id=row["thread_id"],
            causation_id=row["causation_id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            event_type=row["event_type"],
            payload=canonical_snapshot(json.loads(row["payload_json"])),
            provenance=prov,
        )

    def for_thread(self, thread_id: str):
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM events WHERE thread_id=? ORDER BY sequence_no", (thread_id,)
            ).fetchall()
        return tuple(self._event(row) for row in rows)

    def all(self):
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT * FROM events ORDER BY sequence_no").fetchall()
        return tuple(self._event(row) for row in rows)

    def readable_trace(self, thread_id: str):
        return [
            {
                "sequence_no": e.sequence_no,
                "event_type": e.event_type,
                "event_id": e.event_id,
                "causation_id": e.causation_id,
                "timestamp": e.timestamp.isoformat(),
                "payload": thaw_value(e.payload),
                "provenance": asdict(e.provenance),
            }
            for e in self.for_thread(thread_id)
        ]


class SQLiteCheckpointer:
    """Safe tagged-JSON checkpoint store with integrity digest."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._lock = threading.RLock()
        self._init()

    def _connect(self):
        return _connect(self.path)

    def _init(self):
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS checkpoints(
                thread_id TEXT PRIMARY KEY,
                last_event_id TEXT,
                state_json TEXT NOT NULL,
                digest TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )"""
            )

    def save(self, state: AgentState) -> None:
        last_event_id = state.get("last_event_id")
        if not last_event_id:
            raise CheckpointIntegrityError("durable checkpoint requires a last_event_id")
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                _upsert_checkpoint(conn, state, last_event_id)
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def save_consistent(self, event_store: SQLiteEventStore, state: AgentState) -> None:
        if not isinstance(event_store, SQLiteEventStore) or event_store.path != self.path:
            raise CheckpointIntegrityError(
                "checkpoint and event store must share one SQLite durability boundary"
            )
        thread_id = state["thread_id"]
        last_event_id = state.get("last_event_id")
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                tail = _thread_tail(conn, thread_id)
                if tail is None or tail["event_id"] != last_event_id:
                    raise CheckpointIntegrityError(
                        "checkpoint candidate does not match durable event tail"
                    )
                _upsert_checkpoint(conn, state, last_event_id)
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def load(self, thread_id: str) -> AgentState | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM checkpoints WHERE thread_id=?", (thread_id,)
            ).fetchone()
        if row is None:
            return None
        text = row["state_json"]
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest != row["digest"]:
            raise CheckpointIntegrityError("checkpoint digest mismatch")
        try:
            decoded = decode(json.loads(text))
        except (json.JSONDecodeError, CheckpointCodecError) as exc:
            raise CheckpointIntegrityError("checkpoint cannot be safely decoded") from exc
        if type(decoded) is not dict or decoded.get("thread_id") != thread_id:
            raise CheckpointIntegrityError("checkpoint identity mismatch")
        if decoded.get("last_event_id") != row["last_event_id"]:
            raise CheckpointIntegrityError("checkpoint last_event_id mismatch")
        return AgentState(**decoded)


class SQLiteApprovalRecordStore:
    """Durable idempotent ApprovalRecord authority."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._init()

    def _connect(self):
        return _connect(self.path)

    def _init(self):
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS approval_records(
                approval_request_id TEXT PRIMARY KEY,
                approval_id TEXT NOT NULL UNIQUE,
                transaction_id TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                operator_id TEXT NOT NULL,
                trust_domain TEXT NOT NULL,
                decision TEXT NOT NULL,
                approved_at TEXT NOT NULL
            )"""
            )

    @staticmethod
    def _from_row(row: sqlite3.Row) -> ApprovalRecord:
        return ApprovalRecord(
            approval_id=row["approval_id"],
            approval_request_id=row["approval_request_id"],
            transaction_id=row["transaction_id"],
            fingerprint=row["fingerprint"],
            operator_id=row["operator_id"],
            trust_domain=row["trust_domain"],
            decision=ApprovalDecision(row["decision"]),
            approved_at=datetime.fromisoformat(row["approved_at"]),
        )

    def record(self, candidate: ApprovalRecord) -> ApprovalRecord:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                existing = conn.execute(
                    "SELECT * FROM approval_records WHERE approval_request_id=?",
                    (candidate.approval_request_id,),
                ).fetchone()
                if existing is not None:
                    record = self._from_row(existing)
                    if not _same_approval_authority(record, candidate):
                        raise ApprovalRecordConflict(
                            "approval request was already resolved with different authorization facts"
                        )
                    conn.execute("COMMIT")
                    return record
                conn.execute(
                    "INSERT INTO approval_records(approval_request_id,approval_id,transaction_id,fingerprint,operator_id,trust_domain,decision,approved_at) "
                    "VALUES(?,?,?,?,?,?,?,?)",
                    (
                        candidate.approval_request_id,
                        candidate.approval_id,
                        candidate.transaction_id,
                        candidate.fingerprint,
                        candidate.operator_id,
                        candidate.trust_domain,
                        candidate.decision.value,
                        candidate.approved_at.isoformat(),
                    ),
                )
                conn.execute("COMMIT")
                return candidate
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def get(self, approval_request_id: str) -> ApprovalRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM approval_records WHERE approval_request_id=?",
                (approval_request_id,),
            ).fetchone()
        return None if row is None else self._from_row(row)

    def record_with_event(
        self,
        event_store: SQLiteEventStore,
        candidate: ApprovalRecord,
        *,
        request_id: str,
        thread_id: str,
        provenance: EventProvenance,
        causation_id: str | None,
    ) -> tuple[ApprovalRecord, Event]:
        if event_store.path != self.path:
            raise CheckpointIntegrityError("approval store and event store must share SQLite")
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                existing = conn.execute(
                    "SELECT * FROM approval_records WHERE approval_request_id=?",
                    (candidate.approval_request_id,),
                ).fetchone()
                if existing is None:
                    _assert_expected_tail(conn, thread_id, causation_id)
                    record = candidate
                    conn.execute(
                        "INSERT INTO approval_records(approval_request_id,approval_id,transaction_id,fingerprint,operator_id,trust_domain,decision,approved_at) "
                        "VALUES(?,?,?,?,?,?,?,?)",
                        (
                            record.approval_request_id,
                            record.approval_id,
                            record.transaction_id,
                            record.fingerprint,
                            record.operator_id,
                            record.trust_domain,
                            record.decision.value,
                            record.approved_at.isoformat(),
                        ),
                    )
                else:
                    record = self._from_row(existing)
                    if not _same_approval_authority(record, candidate):
                        raise ApprovalRecordConflict(
                            "approval request was already resolved with different authorization facts"
                        )
                event_type = (
                    "ApprovalGranted"
                    if record.decision is ApprovalDecision.APPROVE
                    else "ApprovalRejected"
                )
                stable_event_id = f"evt_approval_{record.approval_request_id}"
                row = conn.execute(
                    "SELECT * FROM events WHERE event_id=?", (stable_event_id,)
                ).fetchone()
                payload = {
                    "approval_id": record.approval_id,
                    "approval_request_id": record.approval_request_id,
                    "transaction_id": record.transaction_id,
                    "fingerprint": record.fingerprint,
                    "operator_id": record.operator_id,
                    "trust_domain": record.trust_domain,
                    "decision": record.decision.value,
                }
                if row is None:
                    # New record path already checked the tail. An existing
                    # record without its audit event is an integrity failure.
                    if existing is not None:
                        raise CheckpointIntegrityError(
                            "durable approval record exists without approval audit event"
                        )
                    event = event_store._append_in_connection(
                        conn,
                        event_type=event_type,
                        request_id=request_id,
                        thread_id=thread_id,
                        payload=payload,
                        provenance=provenance,
                        causation_id=causation_id,
                        event_id=stable_event_id,
                    )
                else:
                    payload_json, provenance_json = event_store._serialized(payload, provenance)
                    if not event_store._row_matches(
                        row,
                        event_type,
                        request_id,
                        thread_id,
                        causation_id,
                        payload_json,
                        provenance_json,
                    ):
                        raise EventIntegrityError("approval audit event conflicts with durable record")
                    tail = _thread_tail(conn, thread_id)
                    if tail is None or tail["event_id"] != stable_event_id:
                        raise DurableConcurrencyError("approval replay is stale")
                    event = event_store._event(row)
                conn.execute("COMMIT")
                return record, event
            except Exception:
                conn.execute("ROLLBACK")
                raise


class SQLiteExecutionClaimStore:
    """Cross-process single-claim/single-attempt capability store."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._init()

    def _connect(self):
        return _connect(self.path)

    def _init(self):
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS execution_claims(
                transaction_id TEXT PRIMARY KEY,
                claim_id TEXT NOT NULL UNIQUE,
                approval_id TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                claimed_at TEXT NOT NULL,
                attempt_id TEXT UNIQUE,
                attempted_at TEXT
            )"""
            )

    @staticmethod
    def _claim_from_row(row: sqlite3.Row) -> ExecutionClaim:
        return ExecutionClaim(
            row["claim_id"],
            row["transaction_id"],
            row["approval_id"],
            row["fingerprint"],
            datetime.fromisoformat(row["claimed_at"]),
        )

    @staticmethod
    def _validate_approval(transaction: WriteTransaction, approval: ApprovalRecord) -> None:
        if approval.decision is not ApprovalDecision.APPROVE:
            raise ValueError("only an approval grant may create an ExecutionClaim")
        if (
            approval.transaction_id != transaction.transaction_id
            or approval.fingerprint != transaction.fingerprint
        ):
            raise ValueError("approval does not authorize this transaction")

    def claim(self, transaction: WriteTransaction, approval: ApprovalRecord) -> ExecutionClaim:
        self._validate_approval(transaction, approval)
        claim = ExecutionClaim(
            claim_id=f"claim_{uuid.uuid4().hex}",
            transaction_id=transaction.transaction_id,
            approval_id=approval.approval_id,
            fingerprint=transaction.fingerprint,
            claimed_at=datetime.now(timezone.utc),
        )
        with self._connect() as conn:
            try:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    "INSERT INTO execution_claims(transaction_id,claim_id,approval_id,fingerprint,claimed_at) VALUES(?,?,?,?,?)",
                    (
                        claim.transaction_id,
                        claim.claim_id,
                        claim.approval_id,
                        claim.fingerprint,
                        claim.claimed_at.isoformat(),
                    ),
                )
                conn.execute("COMMIT")
            except sqlite3.IntegrityError as exc:
                conn.execute("ROLLBACK")
                raise ExecutionClaimAlreadyExists("execution claim already exists") from exc
        return claim

    def claim_with_event(
        self,
        event_store: SQLiteEventStore,
        transaction: WriteTransaction,
        approval: ApprovalRecord,
        *,
        request_id: str,
        thread_id: str,
        provenance: EventProvenance,
        causation_id: str | None,
    ) -> tuple[ExecutionClaim, Event]:
        self._validate_approval(transaction, approval)
        if event_store.path != self.path:
            raise CheckpointIntegrityError("claim store and event store must share SQLite")
        claim = ExecutionClaim(
            claim_id=f"claim_{uuid.uuid4().hex}",
            transaction_id=transaction.transaction_id,
            approval_id=approval.approval_id,
            fingerprint=transaction.fingerprint,
            claimed_at=datetime.now(timezone.utc),
        )
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                _assert_expected_tail(conn, thread_id, causation_id)
                existing = conn.execute(
                    "SELECT * FROM execution_claims WHERE transaction_id=?",
                    (transaction.transaction_id,),
                ).fetchone()
                if existing is not None:
                    raise ExecutionClaimAlreadyExists("execution claim already exists")
                conn.execute(
                    "INSERT INTO execution_claims(transaction_id,claim_id,approval_id,fingerprint,claimed_at) VALUES(?,?,?,?,?)",
                    (
                        claim.transaction_id,
                        claim.claim_id,
                        claim.approval_id,
                        claim.fingerprint,
                        claim.claimed_at.isoformat(),
                    ),
                )
                event = event_store._append_in_connection(
                    conn,
                    event_type="ExecutionClaimed",
                    request_id=request_id,
                    thread_id=thread_id,
                    payload={
                        "transaction_id": transaction.transaction_id,
                        "claim_id": claim.claim_id,
                        "approval_id": claim.approval_id,
                    },
                    provenance=provenance,
                    causation_id=causation_id,
                    event_id=f"evt_claim_{transaction.transaction_id}",
                )
                conn.execute("COMMIT")
                return claim, event
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def consume_attempt(self, claim: ExecutionClaim) -> str:
        attempt_id = f"attempt_{uuid.uuid4().hex}"
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT * FROM execution_claims WHERE transaction_id=?",
                    (claim.transaction_id,),
                ).fetchone()
                if (
                    row is None
                    or row["claim_id"] != claim.claim_id
                    or row["approval_id"] != claim.approval_id
                    or row["fingerprint"] != claim.fingerprint
                ):
                    raise ValueError("execution claim is not authoritative")
                if row["attempt_id"] is not None:
                    raise MutationAttemptAlreadyConsumed("mutation attempt already consumed")
                cursor = conn.execute(
                    "UPDATE execution_claims SET attempt_id=?, attempted_at=? "
                    "WHERE transaction_id=? AND attempt_id IS NULL",
                    (
                        attempt_id,
                        datetime.now(timezone.utc).isoformat(),
                        claim.transaction_id,
                    ),
                )
                if cursor.rowcount != 1:
                    raise MutationAttemptAlreadyConsumed("mutation attempt already consumed")
                conn.execute("COMMIT")
                return attempt_id
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def consume_attempt_with_event(
        self,
        event_store: SQLiteEventStore,
        claim: ExecutionClaim,
        *,
        request_id: str,
        thread_id: str,
        tool_name: str,
        fingerprint: str,
        provenance: EventProvenance,
        causation_id: str | None,
    ) -> tuple[str, Event]:
        if event_store.path != self.path:
            raise CheckpointIntegrityError("claim store and event store must share SQLite")
        attempt_id = f"attempt_{uuid.uuid4().hex}"
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                _assert_expected_tail(conn, thread_id, causation_id)
                row = conn.execute(
                    "SELECT * FROM execution_claims WHERE transaction_id=?",
                    (claim.transaction_id,),
                ).fetchone()
                if (
                    row is None
                    or row["claim_id"] != claim.claim_id
                    or row["approval_id"] != claim.approval_id
                    or row["fingerprint"] != claim.fingerprint
                ):
                    raise ValueError("execution claim is not authoritative")
                if row["attempt_id"] is not None:
                    raise MutationAttemptAlreadyConsumed("mutation attempt already consumed")
                cursor = conn.execute(
                    "UPDATE execution_claims SET attempt_id=?, attempted_at=? "
                    "WHERE transaction_id=? AND attempt_id IS NULL",
                    (
                        attempt_id,
                        datetime.now(timezone.utc).isoformat(),
                        claim.transaction_id,
                    ),
                )
                if cursor.rowcount != 1:
                    raise MutationAttemptAlreadyConsumed("mutation attempt already consumed")
                event = event_store._append_in_connection(
                    conn,
                    event_type="MutationStarted",
                    request_id=request_id,
                    thread_id=thread_id,
                    payload={
                        "transaction_id": claim.transaction_id,
                        "execution_attempt_id": attempt_id,
                        "claim_id": claim.claim_id,
                        "tool_name": tool_name,
                        "fingerprint": fingerprint,
                    },
                    provenance=provenance,
                    causation_id=causation_id,
                    event_id=f"evt_mutation_started_{attempt_id}",
                )
                conn.execute("COMMIT")
                return attempt_id, event
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def claim_for(self, transaction_id: str) -> ExecutionClaim | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM execution_claims WHERE transaction_id=?", (transaction_id,)
            ).fetchone()
        return None if row is None else self._claim_from_row(row)

    def attempt_for(self, transaction_id: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT attempt_id FROM execution_claims WHERE transaction_id=?",
                (transaction_id,),
            ).fetchone()
        return None if row is None else row["attempt_id"]
