"""Deterministic in-memory audit trail, metadata only, hash chained."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

ZERO = "0" * 64


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


@dataclass(frozen=True)
class AuditEntry:
    sequence: int
    role: str
    action: str
    details: dict
    previous_hash: str
    hash: str


class AuditTrail:
    def __init__(self) -> None:
        self.entries: list[AuditEntry] = []

    def record(self, role: str, action: str, details: dict | None = None) -> None:
        if role not in {"planner", "researcher", "verifier", "reporter", "engine"}:
            raise ValueError("unknown audit role")
        details = details or {}
        payload = {"sequence": len(self.entries), "role": role,
                   "action": action, "details": details,
                   "previous_hash": self.entries[-1].hash if self.entries else ZERO}
        self.entries.append(AuditEntry(**payload, hash=digest(payload)))

    @staticmethod
    def verify(entries: list[dict]) -> bool:
        previous = ZERO
        for index, entry in enumerate(entries):
            if set(entry) != {"sequence", "role", "action", "details", "previous_hash", "hash"}:
                return False
            payload = {key: value for key, value in entry.items() if key != "hash"}
            if entry["sequence"] != index or entry["previous_hash"] != previous or entry["hash"] != digest(payload):
                return False
            previous = entry["hash"]
        return True

    def to_dict(self) -> list[dict]:
        return [asdict(entry) for entry in self.entries]
