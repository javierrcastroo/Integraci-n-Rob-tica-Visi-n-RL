"""Funciones de apoyo para convertir detecciones de tablero en coordenadas."""
from __future__ import annotations

from typing import Dict, List, Optional

import cv2
import numpy as np

import board_state
import board_tracker


def collect_objects_info(vis_img, warp_img, H_warp, quad, slot):
    slot["cm_per_pix"] = None

    tl, tr, br, bl = quad
    top_mid = (tl + tr) / 2.0
    bot_mid = (bl + br) / 2.0
    board_height_px = float(np.linalg.norm(top_mid - bot_mid))
    board_height_cm = board_tracker.BOARD_SQUARES * board_tracker.SQUARE_SIZE_CM
    if board_height_px < 1e-3:
        return []
    cm_per_pix = board_height_cm / board_height_px
    slot["cm_per_pix"] = cm_per_pix

    origin = board_state.GLOBAL_ORIGIN
    y_off = 120 if slot["name"] == "T1" else 220

    infos: List[Dict] = []

    for oid, data in slot["tracked"].items():
        if not data.get("stable", False):
            continue
        obj_x_pix, obj_y_pix = data["pt"]
        label = data.get("label")

        obj_x_warp, obj_y_warp = cv2.perspectiveTransform(
            np.array([[[obj_x_pix, obj_y_pix]]], dtype=np.float32),
            H_warp,
        )[0, 0]

        n_cells = board_tracker.BOARD_SQUARES
        cell_size_px = warp_img.shape[0] / float(n_cells)
        col = int(obj_x_warp // cell_size_px)
        row = int(obj_y_warp // cell_size_px)
        col = max(0, min(n_cells - 1, col))
        row = max(0, min(n_cells - 1, row))
        cell_label = f"{chr(ord('A') + col)}{row + 1}"

        info: Dict[str, Optional[float]] = {
            "slot": slot["name"],
            "object_id": oid,
            "cell": cell_label,
            "col": col + 1,
            "row": row + 1,
            "dx_cm": None,
            "dy_cm": None,
            "has_origin": origin is not None,
            "ship_type": label,
        }

        type_suffix = f" [{label}]" if label else ""
        text = f"{slot['name']}-O{oid}: {cell_label}{type_suffix}"

        if origin is not None:
            gx_pix, gy_pix = origin
            dx_pix = obj_x_pix - gx_pix
            dy_pix = gy_pix - obj_y_pix
            dx_cm = float(dx_pix * cm_per_pix)
            dy_cm = float(dy_pix * cm_per_pix)
            info["dx_cm"] = dx_cm
            info["dy_cm"] = dy_cm
            text += f" ({dx_cm:.1f},{dy_cm:.1f})cm"

        cv2.putText(
            vis_img,
            text,
            (10, y_off),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
        )
        y_off += 15

        base_y = 25 + oid * 22
        cv2.rectangle(warp_img, (10, base_y - 15), (320, base_y + 5), (0, 0, 0), -1)
        cv2.putText(
            warp_img,
            text,
            (15, base_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        if origin is not None:
            print(
                f"[{slot['name']}] O{oid} -> {cell_label}{type_suffix} | Global ({info['dx_cm']:.1f}, {info['dy_cm']:.1f}) cm"
            )
        else:
            print(
                f"[{slot['name']}] O{oid} -> {cell_label}{type_suffix} | Global (sin ArUco)"
            )

        infos.append(info)

    return infos


def compute_global_scale(boards_state_list):
    values = [slot.get("cm_per_pix") for slot in boards_state_list if slot.get("cm_per_pix")]
    if not values:
        return None
    return sum(values) / len(values)


def draw_quad(img, quad, color=(0, 255, 255)):
    if quad is None:
        return
    q = np.array(quad, dtype=np.int32)
    cv2.polylines(img, [q], True, color, 2)
