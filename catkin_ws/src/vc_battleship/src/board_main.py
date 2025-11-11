# board_main.py
import cv2
import os
import numpy as np

from board_config import USE_UNDISTORT_BOARD, BOARD_CAMERA_PARAMS_PATH, WARP_SIZE
import board_ui
import board_tracker
import object_tracker
import board_state
import board_processing as bp


def main():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara 1 (tablero)")

    # cargar calibración de cámara
    mtx = dist = None
    if USE_UNDISTORT_BOARD and os.path.exists(BOARD_CAMERA_PARAMS_PATH):
        data = np.load(BOARD_CAMERA_PARAMS_PATH)
        mtx = data["camera_matrix"]
        dist = data["dist_coeffs"]
        print("[INFO] Undistort activado")

    # dos tableros
    boards_state_list = [
        board_state.init_board_state("T1"),
        board_state.init_board_state("T2"),
    ]

    cv2.namedWindow("Tablero")
    cv2.setMouseCallback("Tablero", board_ui.board_mouse_callback)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # detectar el cubo verde (origen global) CADA FRAME
        _detect_global_origin(frame)

        # procesar todos los tableros con el origen global actual
        vis, mask_b, mask_o, _ = bp.process_all_boards(
            frame,
            boards_state_list,
            cam_mtx=mtx,
            dist=dist,
            max_boards=2,
            warp_size=WARP_SIZE,
        )

        # dibujar el origen global si lo tenemos
        if board_state.GLOBAL_ORIGIN is not None:
            gx, gy = board_state.GLOBAL_ORIGIN
            cv2.circle(vis, (int(gx), int(gy)), 10, (0, 255, 0), -1)
            cv2.putText(
                vis,
                "ORIGEN (verde)",
                (int(gx) + 10, int(gy) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        board_ui.draw_board_hud(vis)

        # mostrar
        cv2.imshow("Tablero", vis)
        cv2.imshow("Mascara tablero", mask_b)
        if mask_o is not None:
            cv2.imshow("Mascara objetos", mask_o)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

        handle_keys(key, frame)

    cap.release()
    cv2.destroyAllWindows()


def _detect_global_origin(frame):
    """
    Busca en TODO el frame el color que se calibró con 'r'
    y actualiza board_state.GLOBAL_ORIGIN cada frame.
    Si un frame no lo ve, aguanta unos cuantos.
    """
    if not hasattr(object_tracker, "current_origin_lower"):
        return

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # cuadrilátero que cubre toda la imagen
    h, w = frame.shape[:2]
    full_quad = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

    pts, _ = object_tracker.detect_colored_points_in_board(
        hsv,
        full_quad,
        object_tracker.current_origin_lower,
        object_tracker.current_origin_upper,
        max_objs=1,
        min_area=40,
    )

    if pts:
        board_state.GLOBAL_ORIGIN = pts[0]
        board_state.GLOBAL_ORIGIN_MISS = 0
    else:
        # si no lo ve este frame, aguanta unos cuantos
        board_state.GLOBAL_ORIGIN_MISS += 1
        if board_state.GLOBAL_ORIGIN_MISS > board_state.GLOBAL_ORIGIN_MAX_MISS:
            board_state.GLOBAL_ORIGIN = None


def handle_keys(key, frame):
    # calibrar color del tablero
    if key == ord("b"):
        if board_ui.board_roi_defined:
            x0, x1 = sorted([board_ui.bx_start, board_ui.bx_end])
            y0, y1 = sorted([board_ui.by_start, board_ui.by_end])
            roi_hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
            lo, up = board_tracker.calibrate_board_color_from_roi(roi_hsv)
            board_tracker.current_lower, board_tracker.current_upper = lo, up
            print("[INFO] calibrado TABLERO:", lo, up)
        else:
            print("[WARN] dibuja ROI en 'Tablero' primero")

    # calibrar color de las fichas
    elif key == ord("o"):
        if board_ui.board_roi_defined:
            x0, x1 = sorted([board_ui.bx_start, board_ui.bx_end])
            y0, y1 = sorted([board_ui.by_start, board_ui.by_end])
            roi_hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
            lo, up = object_tracker.calibrate_object_color_from_roi(roi_hsv)
            object_tracker.current_obj_lower, object_tracker.current_obj_upper = lo, up
            print("[INFO] calibrado OBJETO:", lo, up)
        else:
            print("[WARN] dibuja ROI sobre la ficha")

    # 🔴 ahora 'r' = calibrar el color del CUBO VERDE (origen global)
    elif key == ord("r"):
        if board_ui.board_roi_defined:
            x0, x1 = sorted([board_ui.bx_start, board_ui.bx_end])
            y0, y1 = sorted([board_ui.by_start, board_ui.by_end])
            roi_hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
            lo, up = object_tracker.calibrate_origin_color_from_roi(roi_hsv)
            object_tracker.current_origin_lower = lo
            object_tracker.current_origin_upper = up
            # forzamos a que la próxima iteración vuelva a buscarlo
            board_state.GLOBAL_ORIGIN = None
            board_state.GLOBAL_ORIGIN_MISS = 0
            print("[INFO] calibrado ORIGEN GLOBAL (cubo verde):", lo, up)
        else:
            print("[WARN] dibuja ROI sobre el cubo verde para el origen")


if __name__ == "__main__":
    main()