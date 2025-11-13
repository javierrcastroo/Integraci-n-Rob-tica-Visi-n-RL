# board_processing.py
import cv2
import numpy as np
import board_tracker
import object_tracker
import board_ui
import board_state
import board_ammo
from board_geometry import collect_objects_info, compute_global_scale, draw_quad
from tracking_utils import TRACK_STABLE_HITS, update_tracks


TRACK_STABLE_HITS = 5

AMMO_STATE = {
    "tracked": {},
    "next_id": 1,
    "selected_id": None,
    "mask": None,
}


def process_all_boards(frame, boards_state_list, cam_mtx=None, dist=None, max_boards=2, warp_size=500):
    """
    Detecta varios tableros, los asigna a los slots existentes (T1, T2),
    procesa cada uno y devuelve todo para mostrar.
    """
    vis_all, boards_found, mask_board = board_tracker.detect_multiple_boards(
        frame,
        camera_matrix=cam_mtx,
        dist_coeffs=dist,
        max_boards=max_boards,
    )

    # dibujar ROI y HUD
    board_ui.draw_board_roi(vis_all)
    board_ui.draw_board_hud(vis_all)

    # asignar detecciones a slots por cercanía
    assignments = _assign_detections_to_slots(boards_found, boards_state_list)

    obj_mask_show = None
    all_objects_info = []

    for slot_idx, slot in enumerate(boards_state_list):
        det_idx = assignments.get(slot_idx, None)
        if det_idx is not None:
            binfo = boards_found[det_idx]
            quad = binfo["quad"]
            slot["last_quad"] = quad
            slot["miss"] = 0
            obj_mask, slot_objects = process_single_board(
                vis_all, frame, quad, slot, warp_size
            )
            if obj_mask is not None:
                obj_mask_show = obj_mask
            if slot_objects:
                all_objects_info.extend(slot_objects)
        else:
            fallback_or_decay(slot, vis_all)

    cm_per_pix = compute_global_scale(boards_state_list)
    ammo_result = board_ammo.process_ammo_sources(frame, vis_all, cm_per_pix)

    return vis_all, mask_board, obj_mask_show, all_objects_info, ammo_result


def _assign_detections_to_slots(boards_found, boards_state_list):
    """
    Empareja detecciones de tableros con los slots (T1, T2) por proximidad.
    Así no cambian de nombre cuando el contorno baila.
    """
    assignments = {}
    if not boards_found:
        return assignments

    # centros de detección
    det_centers = []
    for b in boards_found:
        quad = b["quad"]
        cx = np.mean(quad[:, 0])
        cy = np.mean(quad[:, 1])
        det_centers.append((cx, cy))

    used = set()
    for slot_idx, slot in enumerate(boards_state_list):
        best_det = None
        best_dist = 1e9

        if slot["last_quad"] is not None:
            sq = slot["last_quad"]
            sx = np.mean(sq[:, 0])
            sy = np.mean(sq[:, 1])
            slot_center = (sx, sy)
        else:
            slot_center = None

        for det_idx, (dx, dy) in enumerate(det_centers):
            if det_idx in used:
                continue
            if slot_center is None:
                best_det = det_idx
                break
            dist = ((dx - slot_center[0]) ** 2 + (dy - slot_center[1]) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_det = det_idx

        if best_det is not None:
            assignments[slot_idx] = best_det
            used.add(best_det)

    return assignments


def process_single_board(vis_img, frame_bgr, quad, slot, warp_size=500):
    """
    Procesa SOLO un tablero:
    - aplanado
    - detección de fichas dentro del tablero
    - tracking
    - transformación a coordenadas globales (marcador ArUco)
    - pintado en las dos ventanas
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # aplanar tablero
    src = np.array(quad, dtype=np.float32)
    dst = np.array(
        [
            [0, 0],
            [warp_size - 1, 0],
            [warp_size - 1, warp_size - 1],
            [0, warp_size - 1],
        ],
        dtype=np.float32,
    )
    H_warp = cv2.getPerspectiveTransform(src, dst)
    warp_img = cv2.warpPerspective(frame_bgr, H_warp, (warp_size, warp_size))

    # detectar fichas (color objeto) dentro del tablero
    obj_detections, obj_mask = object_tracker.detect_ships_in_board(
        hsv,
        quad,
        max_objs_per_type=4,
        min_area=40,
    )

    # dibujar en la vista principal
    for det in obj_detections:
        cx, cy = det["pt"]
        color = (0, 0, 255) if det["label"] == "ship2" else (0, 255, 255)
        cv2.circle(vis_img, (cx, cy), 6, color, -1)

    # tracking por tablero
    slot["tracked"], slot["next_id"] = update_tracks(
        slot["tracked"], obj_detections, slot["next_id"]
    )

    # si tenemos origen global (ArUco), pasamos todo a coordenadas globales
    objects_info = collect_objects_info(vis_img, warp_img, H_warp, quad, slot)

    # mostrar ventana del tablero aplanado
    cv2.imshow(f"{slot['name']} aplanado", warp_img)

    return obj_mask, objects_info

        if info["id"] == selected_id:
            selected_info = info

    result["list"] = stable_infos
    result["selected"] = selected_info

    return result


def fallback_or_decay(slot, vis_img):
    if slot["last_quad"] is not None and slot["miss"] <= 10:
        draw_quad(vis_img, slot["last_quad"])
        slot["miss"] += 1
    else:
        slot["miss"] += 1
        # purgar tracking de ese tablero
        for oid in list(slot["tracked"].keys()):
            data = slot["tracked"][oid]
            data["miss"] += 1
            hits = data.get("hits", 0)
            data["hits"] = max(hits - 1, 0)
            data["stable"] = data.get("hits", 0) >= TRACK_STABLE_HITS
            if data["miss"] > 15:
                del slot["tracked"][oid]
