"""Gestión del tracking de munición fuera del tablero."""
from __future__ import annotations

from typing import Dict, List, Optional

import cv2

import board_state
import object_tracker
from tracking_utils import TRACK_STABLE_HITS, update_tracks


AMMO_STATE: Dict[str, Dict] = {
    "tracked": {},
    "next_id": 1,
    "selected_id": None,
}


def process_ammo_sources(frame_bgr, vis_img, cm_per_pix):
    result = {"mask": None, "list": [], "selected": None}

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    detections, mask = object_tracker.detect_ammo_in_scene(hsv)
    result["mask"] = mask

    AMMO_STATE["tracked"], AMMO_STATE["next_id"] = update_tracks(
        AMMO_STATE["tracked"],
        detections,
        AMMO_STATE["next_id"],
        max_dist=80,
        max_miss=20,
        stable_hits=TRACK_STABLE_HITS,
    )

    origin = board_state.GLOBAL_ORIGIN
    stable_infos: List[Dict[str, Optional[float]]] = []
    for oid, data in sorted(AMMO_STATE["tracked"].items()):
        if not data.get("stable", False):
            continue
        px, py = data["pt"]
        info = {
            "id": int(oid),
            "label": "ammo",
            "px": int(px),
            "py": int(py),
            "dx_cm": None,
            "dy_cm": None,
        }

        if origin is not None and cm_per_pix is not None:
            gx, gy = origin
            dx_pix = px - gx
            dy_pix = gy - py
            info["dx_cm"] = float(dx_pix * cm_per_pix)
            info["dy_cm"] = float(dy_pix * cm_per_pix)

        stable_infos.append(info)

    valid_ids = {info["id"] for info in stable_infos}
    previous = AMMO_STATE.get("selected_id")
    if previous in valid_ids:
        selected_id = previous
    elif stable_infos:
        selected_id = min(valid_ids)
    else:
        selected_id = None
    AMMO_STATE["selected_id"] = selected_id

    selected_info = None
    y_base = vis_img.shape[0] - 80
    for idx, info in enumerate(stable_infos):
        color = (0, 255, 0) if info["id"] == selected_id else (0, 200, 255)
        cv2.circle(vis_img, (info["px"], info["py"]), 10, color, 2)
        label = f"M{info['id']}"
        cv2.putText(
            vis_img,
            label,
            (info["px"] + 10, info["py"] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

        text = label
        if info["dx_cm"] is not None and info["dy_cm"] is not None:
            text += f" -> ({info['dx_cm']:.1f}, {info['dy_cm']:.1f}) cm"
        else:
            text += " -> sin escala"

        y_text = max(30, min(vis_img.shape[0] - 10, y_base + idx * 18))
        cv2.putText(
            vis_img,
            text,
            (10, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

        if info["id"] == selected_id:
            selected_info = info

    result["list"] = stable_infos
    result["selected"] = selected_info

    return result
