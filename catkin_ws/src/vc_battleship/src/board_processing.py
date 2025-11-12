# board_processing.py
import cv2
import numpy as np
import board_tracker
import object_tracker
import board_ui
import board_state


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

    return vis_all, mask_board, obj_mask_show, all_objects_info


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
    obj_pts, obj_mask = object_tracker.detect_colored_points_in_board(
        hsv,
        quad,
        object_tracker.current_obj_lower,
        object_tracker.current_obj_upper,
        max_objs=4,
        min_area=40,
    )

    # dibujar en la vista principal
    for (cx, cy) in obj_pts:
        cv2.circle(vis_img, (cx, cy), 6, (0, 0, 255), -1)

    # tracking por tablero
    slot["tracked"], slot["next_id"] = update_tracks(
        slot["tracked"], obj_pts, slot["next_id"]
    )

    # si tenemos origen global (ArUco), pasamos todo a coordenadas globales
    objects_info = collect_objects_info(vis_img, warp_img, H_warp, quad, slot)

    # mostrar ventana del tablero aplanado
    cv2.imshow(f"{slot['name']} aplanado", warp_img)

    return obj_mask, objects_info


def update_tracks(tracked, detections, next_id, max_dist=35, max_miss=10):
    for oid in list(tracked.keys()):
        tracked[oid]["updated"] = False

    for (cx, cy) in detections:
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
        else:
            tracked[next_id] = {"pt": (cx, cy), "miss": 0, "updated": True}
            next_id += 1

    # purgar
    for oid in list(tracked.keys()):
        if not tracked[oid].get("updated", False):
            tracked[oid]["miss"] += 1
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

    origin = board_state.GLOBAL_ORIGIN

    # 2. preparar desplazamiento vertical de texto para cada tablero
    y_off = 120 if slot["name"] == "T1" else 220

    infos = []

    for oid, data in slot["tracked"].items():
        obj_x_pix, obj_y_pix = data["pt"]

        # proyectar al tablero aplanado para deducir la casilla
        obj_x_warp, obj_y_warp = cv2.perspectiveTransform(
            np.array([[[obj_x_pix, obj_y_pix]]], dtype=np.float32),
            H_warp
        )[0, 0]

        cell_size = board_tracker.SQUARE_SIZE_CM
        n_cells = board_tracker.BOARD_SQUARES
        col = int(obj_x_warp // cell_size)
        row = int(obj_y_warp // cell_size)
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
        }

        text = f"{slot['name']}-O{oid}: {cell_label}"

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
                f"[{slot['name']}] O{oid} -> {cell_label} | Global ({info['dx_cm']:.1f}, {info['dy_cm']:.1f}) cm"
            )
        else:
            print(f"[{slot['name']}] O{oid} -> {cell_label} | Global (sin ArUco)")

        infos.append(info)

    return infos


def fallback_or_decay(slot, vis_img):
    if slot["last_quad"] is not None and slot["miss"] <= 10:
        draw_quad(vis_img, slot["last_quad"])
        slot["miss"] += 1
    else:
        slot["miss"] += 1
        # purgar tracking de ese tablero
        for oid in list(slot["tracked"].keys()):
            slot["tracked"][oid]["miss"] += 1
            if slot["tracked"][oid]["miss"] > 15:
                del slot["tracked"][oid]


def draw_quad(img, quad, color=(0, 255, 255)):
    if quad is None:
        return
    q = np.array(quad, dtype=np.int32)
    cv2.polylines(img, [q], True, color, 2)