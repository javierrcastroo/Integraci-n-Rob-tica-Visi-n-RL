# board_main.py
import cv2
import os
import numpy as np

from board_config import USE_UNDISTORT_BOARD, BOARD_CAMERA_PARAMS_PATH, WARP_SIZE
import board_ui
import board_state
import board_processing as bp
import aruco_utils
import battleship_logic

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

        # detectar el ORIGEN con ArUco cada frame
        aruco_utils.update_global_origin_from_aruco(frame, aruco_id=2)

        # procesar todos los tableros con el origen global actual
        vis, mask_b, mask_ship2, mask_ship1, mask_m, layouts = bp.process_all_boards(
            frame,
            boards_state_list,
            cam_mtx=mtx,
            dist=dist,
            max_boards=2,
            warp_size=WARP_SIZE,
        )

        validation_map = {}
        for layout in layouts:
            ok, msg = battleship_logic.evaluate_board(layout)
            validation_map[layout["name"]] = (ok, msg)
            print(f"[{layout['name']}] {msg}")

        for slot in boards_state_list:
            if slot["name"] in validation_map and slot["last_quad"] is not None:
                ok, msg = validation_map[slot["name"]]
                board_ui.draw_validation_result(vis, slot["last_quad"], msg, ok)

        # dibujar el origen global si lo tenemos
        if board_state.GLOBAL_ORIGIN is not None:
            gx, gy = board_state.GLOBAL_ORIGIN
            cv2.circle(vis, (int(gx), int(gy)), 10, (0, 255, 0), -1)
            cv2.putText(
                vis,
                "ORIGEN (ArUco)",
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
        if mask_ship2 is not None:
            cv2.imshow("Mascara barco x2", mask_ship2)
        if mask_ship1 is not None:
            cv2.imshow("Mascara barco x1", mask_ship1)
        if mask_m is not None:
            cv2.imshow("Mascara municion", mask_m)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

        handle_keys(key, frame)

    cap.release()
    cv2.destroyAllWindows()


def handle_keys(key, frame):
   
    # b = color tablero
    # 2 = barco de dos casillas
    # 1 = barco de una casilla
    # m = color de la munición
    import board_tracker
    import object_tracker
    import board_ui as bu

    if key == ord("b"):
        if bu.board_roi_defined:
            x0, x1 = sorted([bu.bx_start, bu.bx_end])
            y0, y1 = sorted([bu.by_start, bu.by_end])
            roi_hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
            lo, up = board_tracker.calibrate_board_color_from_roi(roi_hsv)
            board_tracker.current_lower, board_tracker.current_upper = lo, up
            print("[INFO] calibrado TABLERO:", lo, up)
        else:
            print("[WARN] dibuja ROI en 'Tablero' primero")

    elif key == ord("2"):
        if bu.board_roi_defined:
            x0, x1 = sorted([bu.bx_start, bu.bx_end])
            y0, y1 = sorted([bu.by_start, bu.by_end])
            roi_hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
            lo, up = object_tracker.calibrate_ship_two_color_from_roi(roi_hsv)
            object_tracker.current_ship_two_lower, object_tracker.current_ship_two_upper = lo, up
            print("[INFO] calibrado BARCO x2:", lo, up)
        else:
            print("[WARN] dibuja ROI sobre el barco largo")

    elif key == ord("1"):
        if bu.board_roi_defined:
            x0, x1 = sorted([bu.bx_start, bu.bx_end])
            y0, y1 = sorted([bu.by_start, bu.by_end])
            roi_hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
            lo, up = object_tracker.calibrate_ship_one_color_from_roi(roi_hsv)
            object_tracker.current_ship_one_lower, object_tracker.current_ship_one_upper = lo, up
            print("[INFO] calibrado BARCO x1:", lo, up)
        else:
            print("[WARN] dibuja ROI sobre el barco corto")

    elif key == ord("m"):
        if bu.board_roi_defined:
            x0, x1 = sorted([bu.bx_start, bu.bx_end])
            y0, y1 = sorted([bu.by_start, bu.by_end])
            roi_hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
            lo, up = object_tracker.calibrate_ammo_color_from_roi(roi_hsv)
            object_tracker.current_ammo_lower, object_tracker.current_ammo_upper = lo, up
            print("[INFO] calibrada MUNICION:", lo, up)
        else:
            print("[WARN] dibuja ROI sobre la municion")



if __name__ == "__main__":
    main()