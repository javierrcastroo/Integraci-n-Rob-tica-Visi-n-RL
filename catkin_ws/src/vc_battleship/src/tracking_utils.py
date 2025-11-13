"""Utilidades de tracking compartidas entre tablero y munición."""
from __future__ import annotations

from typing import Dict, Iterable, Tuple


TRACK_STABLE_HITS = 5


def update_tracks(
    tracked: Dict[int, Dict],
    detections: Iterable[Dict],
    next_id: int,
    *,
    max_dist: float = 35.0,
    max_miss: int = 10,
    stable_hits: int = TRACK_STABLE_HITS,
) -> Tuple[Dict[int, Dict], int]:
    """Actualiza un diccionario de tracks con nuevos puntos detectados."""
    for oid in list(tracked.keys()):
        tracked[oid]["updated"] = False

    for det in detections:
        cx, cy = det["pt"]
        label = det.get("label")
        best_oid = None
        best_dist = 1e9
        for oid, data in tracked.items():
            px, py = data["pt"]
            dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_oid = oid
        if best_oid is not None and best_dist < max_dist:
            tracked[best_oid]["pt"] = (cx, cy)
            tracked[best_oid]["miss"] = 0
            tracked[best_oid]["updated"] = True
            tracked[best_oid]["label"] = label
            hits = tracked[best_oid].get("hits", 0) + 1
            tracked[best_oid]["hits"] = min(hits, stable_hits)
        else:
            tracked[next_id] = {
                "pt": (cx, cy),
                "miss": 0,
                "updated": True,
                "label": label,
                "hits": 1,
                "stable": False,
            }
            next_id += 1

    for oid in list(tracked.keys()):
        if not tracked[oid].get("updated", False):
            tracked[oid]["miss"] += 1
            hits = tracked[oid].get("hits", 0)
            tracked[oid]["hits"] = max(hits - 1, 0)
        tracked[oid]["stable"] = tracked[oid].get("hits", 0) >= stable_hits
        if tracked[oid]["miss"] > max_miss:
            del tracked[oid]

    return tracked, next_id
