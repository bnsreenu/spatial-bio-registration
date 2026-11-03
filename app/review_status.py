"""Review status per sample (task 17), as plain functions on plain dicts.

state["review"][sample_id] = {"status": "unreviewed" | "accepted" | "rejected", "bulk": bool}

"Needs attention" is not stored: it is how an unreviewed sample is shown when
its confidence is low or its registration failed. Accepting or rejecting a
sample replaces it. No Qt imports."""
from __future__ import annotations

from typing import Dict, List, Optional

UNREVIEWED, ACCEPTED, REJECTED = "unreviewed", "accepted", "rejected"
NEEDS_ATTENTION = "needs attention"


def default_entry() -> Dict:
    return {"status": UNREVIEWED, "bulk": False}


def entry(review: Dict, sid: str) -> Dict:
    return review.get(sid) or default_entry()


def needs_attention(result: Dict) -> bool:
    return result.get("status") == "failed" or (result.get("status") == "ok" and result.get("confidence") == "low")


def label(result: Dict, item: Dict) -> str:
    """The status text shown in lists and written to batch_summary.csv."""
    status = item["status"]
    if status == ACCEPTED:
        return "accepted (bulk)" if item.get("bulk") else "accepted"
    if status == REJECTED:
        return "rejected"
    return NEEDS_ATTENTION if needs_attention(result) else UNREVIEWED


def set_status(review: Dict, sid: str, status: str, bulk: bool = False) -> None:
    review[sid] = {"status": status, "bulk": bool(bulk and status == ACCEPTED)}


def bulk_candidates(results: Dict[str, Dict], review: Dict) -> List[str]:
    """Unreviewed samples with high confidence: what "Accept all high-confidence" would accept.
    Rejected samples are left alone."""
    return [sid for sid, res in results.items()
            if res.get("status") == "ok" and res.get("confidence") == "high"
            and entry(review, sid)["status"] == UNREVIEWED]


def accept_bulk(results: Dict[str, Dict], review: Dict) -> List[str]:
    ids = bulk_candidates(results, review)
    for sid in ids:
        set_status(review, sid, ACCEPTED, bulk=True)
    return ids


def to_write(results: Dict[str, Dict], review: Dict, include_all: bool) -> List[str]:
    """Samples an "Export accepted" would write."""
    return [sid for sid in results if include_all or entry(review, sid)["status"] == ACCEPTED]


def clean(review: Optional[Dict]) -> Dict:
    """A review entry read from a session file, with anything unexpected reset to the default."""
    if not isinstance(review, dict) or review.get("status") not in (UNREVIEWED, ACCEPTED, REJECTED):
        return default_entry()
    return {"status": review["status"], "bulk": bool(review.get("bulk")) and review["status"] == ACCEPTED}
