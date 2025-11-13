# board_processing.py
import cv2
import numpy as np
import board_tracker
import object_tracker
import board_ui
import board_state


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

    cm_per_pix = _get_global_scale(boards_state_list)
    ammo_result = process_ammo_sources(frame, vis_all, cm_per_pix)

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


def update_tracks(
    tracked,
    detections,
    next_id,
    max_dist=35,
    max_miss=10,
    stable_hits=TRACK_STABLE_HITS,
):
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

    # purgar
    for oid in list(tracked.keys()):
        if not tracked[oid].get("updated", False):
            tracked[oid]["miss"] += 1
            hits = tracked[oid].get("hits", 0)
            tracked[oid]["hits"] = max(hits - 1, 0)
        tracked[oid]["stable"] = tracked[oid].get("hits", 0) >= stable_hits
        if tracked[oid]["miss"] > max_miss:
            del tracked[oid]

    return tracked, next_id


def collect_objects_info(vis_img, warp_img, H_warp, quad, slot):
    """
    Convierte las fichas del tablero a coordenadas globales (marcador ArUco).
    OJO: el origen ArUco está FUERA del tablero, así que no lo pasamos por la homografía.
    En vez de eso:
      1) medimos distancias en píxeles
      2) las convertimos a cm usando el tamaño físico del tablero
    """
    slot["cm_per_pix"] = None

    # 1. escala cm/píxel del tablero a partir de su altura en píxeles
    # quad: [tl, tr, br, bl]
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

    # 2. preparar desplazamiento vertical de texto para cada tablero
    y_off = 120 if slot["name"] == "T1" else 220

    infos = []

    for oid, data in slot["tracked"].items():
        if not data.get("stable", False):
            continue
        obj_x_pix, obj_y_pix = data["pt"]
        label = data.get("label")

        # proyectar al tablero aplanado para deducir la casilla
        obj_x_warp, obj_y_warp = cv2.perspectiveTransform(
            np.array([[[obj_x_pix, obj_y_pix]]], dtype=np.float32),
            H_warp
        )[0, 0]

        n_cells = board_tracker.BOARD_SQUARES
        cell_size_px = warp_img.shape[0] / float(n_cells)
        col = int(obj_x_warp // cell_size_px)
        row = int(obj_y_warp // cell_size_px)
        col = max(0, min(n_cells - 1, col))
        row = max(0, min(n_cells - 1, row))
        cell_label = f"{chr(ord('A') + col)}{row + 1}"

        info = {
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
            dy_pix = gy_pix - obj_y_pix  # invertimos Y para que "hacia abajo" sea positivo
            dx_cm = float(dx_pix * cm_per_pix)
            dy_cm = float(dy_pix * cm_per_pix)
            info["dx_cm"] = dx_cm
            info["dy_cm"] = dy_cm
            text += f" ({dx_cm:.1f},{dy_cm:.1f})cm"

        # pintar en la vista principal
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

        # pintar también en la ventana aplanada
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


def _get_global_scale(boards_state_list):
    values = [slot.get("cm_per_pix") for slot in boards_state_list if slot.get("cm_per_pix")]
    if not values:
        return None
    return sum(values) / len(values)


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
    stable_infos = []
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


def draw_quad(img, quad, color=(0, 255, 255)):
    if quad is None:
        return
    q = np.array(quad, dtype=np.int32)
    cv2.polylines(img, [q], True, color, 2)