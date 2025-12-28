# board_processing.py
import cv2
import numpy as np
import board_tracker
import object_tracker
import board_state
import board_ui


def process_board(frame, board_state, cam_mtx=None, dist=None, warp_size=500, cell_cm=None):
    """
    Procesa un único tablero (T1). Si la cámara está rotada o invertida, la
    esquina superior-izquierda de lo que se ve en la imagen sigue siendo la
    casilla (0,0) gracias al reordenado robusto de esquinas en
    ``board_tracker.order_points``.
    """

    vis_all, boards_found, mask_board = board_tracker.detect_multiple_boards(
        frame,
        camera_matrix=cam_mtx,
        dist_coeffs=dist,
        max_boards=1,
    )

    # dibujar ROI y HUD
    board_ui.draw_board_roi(vis_all)
    board_ui.draw_board_hud(vis_all)

    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ammo_pts, ammo_mask_show = object_tracker.detect_colored_points_global(
        frame_hsv,
        object_tracker.current_ammo_ranges,
        max_objs=12,
        min_area=30,
    )
    for (cx, cy) in ammo_pts:
        cv2.circle(vis_all, (cx, cy), 6, (255, 0, 255), -1)

    ship_two_mask_show = None
    ship_one_mask_show = None
    layouts = []

    ratio_cm_per_pix = None
    if boards_found:
        ratio_cm_per_pix = boards_found[0].get("ratio")
        quad = boards_found[0]["quad"]


        print(f"[DBG] ratio(from boards_found[0]['ratio']) = {ratio_cm_per_pix}")
        print(f"[DBG] quad(px) = {quad}")

        # ancho en px (promedio borde superior e inferior)
        (x0, y0), (x1, y1), (x2, y2), (x3, y3) = quad
        top = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        bottom = ((x2 - x3) ** 2 + (y2 - y3) ** 2) ** 0.5
        width_px = 0.5 * (top + bottom)

        board_cm = board_tracker.BOARD_SQUARES * cell_cm  # 19.25 cm si 5x5
        cm_per_px_est = board_cm / width_px
        px_per_cm_est = width_px / board_cm

        print(f"[DBG] width_px≈{width_px:.2f} | board_cm={board_cm:.2f}")
        print(f"[DBG] derived cm/px≈{cm_per_px_est:.5f} | px/cm≈{px_per_cm_est:.2f}")


        board_state["last_quad"] = quad
        board_state["miss"] = 0
        ship_two_mask_show, ship_one_mask_show, layout_info = process_single_board(
            vis_all, frame, quad, board_state, warp_size
        )
        if layout_info is not None:
            layout_info["ammo_global_detections"] = _build_global_detections(
                ammo_pts, ratio_cm_per_pix
            )
            layout_info["ratio_cm_per_pix"] = ratio_cm_per_pix
            layout_info["board_corners_aruco"] = _board_quad_pixel_to_corners_aruco(quad, ratio_cm_per_pix)
            layouts.append(layout_info)
    else:
        fallback_or_decay(board_state, vis_all)

    return (
        vis_all,
        mask_board,
        ship_two_mask_show,
        ship_one_mask_show,
        ammo_mask_show,
        layouts,
    )

def _build_global_detections(ammo_pts, ratio_cm_per_pix):
    origin = board_state.GLOBAL_ORIGIN
    if origin is None:
        return []

    ox, oy = origin
    detections = []
    for (cx, cy) in ammo_pts or []:
        offset_x = float(cx) - float(ox)
        offset_y = float(cy) - float(oy)
        entry = {
            "pixel": (int(cx), int(cy)),
            "offset_from_origin": (offset_x, offset_y),
        }
        if ratio_cm_per_pix is not None:
            entry["xy_aruco"] = (
                (offset_x * ratio_cm_per_pix) / 100.0,
                (offset_y * ratio_cm_per_pix) / 100.0,
            )
        detections.append(entry)
    return detections

def _board_quad_pixel_to_corners_aruco(quad, ratio_cm_per_pix):
    """
    Convierte las 4 esquinas del tablero (quad en píxeles) a coordenadas XY en metros
    relativas al origen GLOBAL_ORIGIN (ArUco), usando ratio_cm_per_pix.
    """
    origin = board_state.GLOBAL_ORIGIN
    if origin is None or quad is None or ratio_cm_per_pix is None:
        return []

    ox, oy = origin
    corners_aruco = []
    for (px, py) in quad:
        offset_x = float(px) - float(ox)
        offset_y = float(py) - float(oy)
        x_m = (offset_x * float(ratio_cm_per_pix)) / 100.0
        y_m = (offset_y * float(ratio_cm_per_pix)) / 100.0
        corners_aruco.append((x_m, y_m))

    return corners_aruco


def process_single_board(vis_img, frame_bgr, quad, slot, warp_size=500):
    """
    Procesa un tablero individual detectando centros de barcos de dos y una casilla
    con el mismo pipeline basado en blobs que teníamos antes: calibras con un ROI,
    buscamos contornos del color elegido, calculamos su centroide y lo traducimos
    a una casilla (A1, B2, ...).
    """

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

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

    H_inv = np.linalg.inv(H_warp)
    n = board_tracker.BOARD_SQUARES
    cell_size = warp_size / n

    # centro de la celda (0,0) en coordenadas warp
    cxw = 0.5 * cell_size
    cyw = 0.5 * cell_size

    ctr_warp = np.array([[[cxw, cyw]]], dtype=np.float32)
    ctr_img = cv2.perspectiveTransform(ctr_warp, H_inv).reshape(-1, 2)[0]
    print(f"[DBG] cell(0,0) center in image px = ({ctr_img[0]:.2f}, {ctr_img[1]:.2f})")

    if board_state.GLOBAL_ORIGIN is not None:
        ox, oy = board_state.GLOBAL_ORIGIN
        dx_px = float(ctr_img[0]) - float(ox)
        dy_px = float(ctr_img[1]) - float(oy)
        print(f"[DBG] aruco->cell(0,0) center offset px = ({dx_px:.2f}, {dy_px:.2f})")

        # convierte a cm usando el ratio ya calculado
        # (si no lo tienes en este scope, pásalo o imprime solo px)


    warp_img = cv2.warpPerspective(frame_bgr, H_warp, (warp_size, warp_size))

    ship_two_pts, ship_two_mask = object_tracker.detect_colored_points_in_board(
        hsv,
        quad,
        object_tracker.current_ship_two_ranges,
        max_objs=2,
        min_area=40,
    )

    ship_one_pts, ship_one_mask = object_tracker.detect_colored_points_in_board(
        hsv,
        quad,
        object_tracker.current_ship_one_ranges,
        max_objs=3,
        min_area=40,
    )

    ammo_pts, ammo_mask = object_tracker.detect_colored_points_in_board(
        hsv,
        quad,
        object_tracker.current_ammo_ranges,
        max_objs=12,
        min_area=30,
    )

    _draw_points(vis_img, ship_two_pts, (0, 0, 255))
    _draw_points(vis_img, ship_one_pts, (0, 255, 255))
    _draw_points(vis_img, ammo_pts, (255, 0, 255))
    _draw_points_on_warp(warp_img, ship_two_pts, H_warp, (0, 0, 255))
    _draw_points_on_warp(warp_img, ship_one_pts, H_warp, (0, 255, 255))
    _draw_points_on_warp(warp_img, ammo_pts, H_warp, (255, 0, 255))

    ship_two_cells_raw, ship_two_labels, ship_two_pairs = _map_points_to_cells(
        ship_two_pts, H_warp, warp_size
    )
    ship_one_cells_raw, ship_one_labels, ship_one_pairs = _map_points_to_cells(
        ship_one_pts, H_warp, warp_size
    )
    ammo_cells_raw, ammo_labels, ammo_pairs = _map_points_to_cells(
        ammo_pts, H_warp, warp_size
    )

    slot["ship_two_cells"] = sorted(set(ship_two_cells_raw))
    slot["ship_one_cells"] = sorted(set(ship_one_cells_raw))
    slot["ammo_cells"] = sorted(set(ammo_cells_raw))

    ship_two_detections = _build_detection_entries(ship_two_pairs)
    ship_one_detections = _build_detection_entries(ship_one_pairs)
    ammo_detections = _build_detection_entries(ammo_pairs)

    display_entries = []
    for idx, label in enumerate(ship_two_labels, 1):
        display_entries.append((f"B2-{idx}", label))
    for idx, label in enumerate(ship_one_labels, 1):
        display_entries.append((f"B1-{idx}", label))
    for idx, label in enumerate(ammo_labels, 1):
        display_entries.append((f"M-{idx}", label))

    _annotate_detections(vis_img, warp_img, slot["name"], display_entries)

    layout_info = {
        "name": slot["name"],
        "ship_two_cells": slot["ship_two_cells"],
        "ship_one_cells": slot["ship_one_cells"],
        "ammo_cells": slot["ammo_cells"],
        "board_size": board_tracker.BOARD_SQUARES,
        "ship_two_detections": ship_two_detections,
        "ship_one_detections": ship_one_detections,
        "ammo_detections": ammo_detections,
        "board_quad_pixel": quad,
        "warp_size_px": warp_size,
    }

    if display_entries:
        for tag, label in display_entries:
            print(f"[{slot['name']}] {tag} -> {label}")

    cv2.imshow(f"{slot['name']} aplanado", warp_img)

    return ship_two_mask, ship_one_mask, layout_info


def fallback_or_decay(slot, vis_img):
    if slot["last_quad"] is not None and slot["miss"] <= 10:
        draw_quad(vis_img, slot["last_quad"])
        slot["miss"] += 1
    else:
        slot["miss"] += 1
        slot["ship_two_cells"] = []
        slot["ship_one_cells"] = []
        slot["ammo_cells"] = []


def draw_quad(img, quad, color=(0, 255, 255)):
    if quad is None:
        return
    q = np.array(quad, dtype=np.int32)
    cv2.polylines(img, [q], True, color, 2)


def _map_points_to_cells(points, H_warp, warp_size):
    if not points:
        return [], [], []

    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(pts, H_warp).reshape(-1, 2)
    n = board_tracker.BOARD_SQUARES
    if n <= 0:
        return [], [], []
    cell_size = warp_size / n

    cells = []
    labels = []
    point_cell_pairs = []
    for (wx, wy), (px, py) in zip(warped, points):
        col = _clip_cell_index(int(np.floor(wx / cell_size)), n)
        row_raw = _clip_cell_index(int(np.floor(wy / cell_size)), n)
        row = (n - 1) - row_raw
        cell = (row, col)
        cells.append(cell)
        labels.append(_format_cell_label(row, col))
        point_cell_pairs.append({"cell": cell, "pixel": (int(px), int(py))})
    return cells, labels, point_cell_pairs


def _build_detection_entries(point_cell_pairs):
    entries = []
    origin = board_state.GLOBAL_ORIGIN
    for pair in point_cell_pairs:
        cell = pair.get("cell")
        pixel = pair.get("pixel")
        if cell is None or pixel is None:
            continue
        offset = None
        if origin is not None:
            ox, oy = origin
            px, py = pixel
            offset = (px - ox, py - oy)
        entries.append(
            {
                "cell": cell,
                "pixel": pixel,
                "offset_from_origin": offset,
            }
        )
    return entries


def _draw_points(img, points, color):
    for (cx, cy) in points:
        cv2.circle(img, (int(cx), int(cy)), 6, color, -1)


def _draw_points_on_warp(warp_img, points, H_warp, color):
    if not points:
        return
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(pts, H_warp).reshape(-1, 2)
    for wx, wy in warped:
        cv2.circle(warp_img, (int(wx), int(wy)), 6, color, 2)


def _annotate_detections(vis_img, warp_img, slot_name, entries):
    if not entries:
        return

    y_offset = 120
    for tag, label in entries:
        text = f"{slot_name}-{tag}: {label}"
        cv2.putText(
            vis_img,
            text,
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )
        y_offset += 18

    for idx, (tag, label) in enumerate(entries):
        base_y = 25 + idx * 22
        cv2.rectangle(warp_img, (10, base_y - 15), (260, base_y + 5), (0, 0, 0), -1)
        cv2.putText(
            warp_img,
            f"{tag}: {label}",
            (15, base_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )


def _format_cell_label(row, col):
    return f"{chr(ord('A') + col)}{row + 1}"


def _clip_cell_index(idx, n):
    return max(0, min(n - 1, idx))