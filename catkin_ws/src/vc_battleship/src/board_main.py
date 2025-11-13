# board_main.py
import cv2
import os
import numpy as np

from board_config import USE_UNDISTORT_BOARD, BOARD_CAMERA_PARAMS_PATH, WARP_SIZE
import board_ui
from board_runtime import BoardRuntime


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

    map1 = map2 = None
    new_cam_mtx = None

    runtime = BoardRuntime(max_boards=2, warp_size=WARP_SIZE)

    cv2.namedWindow("Tablero")
    cv2.setMouseCallback("Tablero", board_ui.board_mouse_callback)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_proc = frame
        if mtx is not None and dist is not None:
            if map1 is None or map2 is None:
                h, w = frame.shape[:2]
                new_cam_mtx, _ = cv2.getOptimalNewCameraMatrix(
                    mtx, dist, (w, h), 0
                )
                map1, map2 = cv2.initUndistortRectifyMap(
                    mtx, dist, None, new_cam_mtx, (w, h), cv2.CV_16SC2
                )
                print(
                    f"[INFO] Mapas de undistort listos para resolución {w}x{h}"
                )
            frame_proc = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)

        # modo espejo para que los movimientos coincidan visualmente
        frame_proc = cv2.flip(frame_proc, 1)

        vis, mask_b, mask_o, _, ammo_data = runtime.process_frame(frame_proc)

        BoardRuntime.draw_origin_indicator(vis)

        board_ui.draw_board_hud(vis)

        # mostrar
        cv2.imshow("Tablero", vis)
        cv2.imshow("Mascara tablero", mask_b)
        if mask_o is not None:
            cv2.imshow("Mascara objetos", mask_o)
        ammo_mask = ammo_data.get("mask") if ammo_data else None
        if ammo_mask is not None:
            cv2.imshow("Mascara municion", ammo_mask)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

        handle_keys(key, frame_proc, runtime)

    cap.release()
    cv2.destroyAllWindows()


def handle_keys(key, frame, runtime: BoardRuntime):
    # calibrar color del tablero
    if key == ord("b"):
        if board_ui.board_roi_defined:
            x0, x1 = sorted([board_ui.bx_start, board_ui.bx_end])
            y0, y1 = sorted([board_ui.by_start, board_ui.by_end])
            roi_hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
            lo, up = runtime.calibrate_board_color(roi_hsv)
            print("[INFO] calibrado TABLERO:", lo, up)
        else:
            print("[WARN] dibuja ROI en 'Tablero' primero")

    # calibrar color de las fichas genéricas (modo legado)
    elif key == ord("o"):
        if board_ui.board_roi_defined:
            x0, x1 = sorted([board_ui.bx_start, board_ui.bx_end])
            y0, y1 = sorted([board_ui.by_start, board_ui.by_end])
            roi_hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
            lo, up = runtime.calibrate_object_color(roi_hsv)
            print("[INFO] calibrado OBJETO:", lo, up)
        else:
            print("[WARN] dibuja ROI sobre la ficha")

    # calibrar barco de tamaño 1
    elif key == ord("1"):
        if board_ui.board_roi_defined:
            x0, x1 = sorted([board_ui.bx_start, board_ui.bx_end])
            y0, y1 = sorted([board_ui.by_start, board_ui.by_end])
            roi_hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
            lo, up = runtime.calibrate_ship_color("ship1", roi_hsv)
            print("[INFO] calibrado BARCO x1:", lo, up)
        else:
            print("[WARN] dibuja ROI del barco tamaño 1")

    # calibrar barco de tamaño 2
    elif key == ord("2"):
        if board_ui.board_roi_defined:
            x0, x1 = sorted([board_ui.bx_start, board_ui.bx_end])
            y0, y1 = sorted([board_ui.by_start, board_ui.by_end])
            roi_hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
            lo, up = runtime.calibrate_ship_color("ship2", roi_hsv)
            print("[INFO] calibrado BARCO x2:", lo, up)
        else:
            print("[WARN] dibuja ROI del barco tamaño 2")

    # calibrar munición
    elif key == ord("m"):
        if board_ui.board_roi_defined:
            x0, x1 = sorted([board_ui.bx_start, board_ui.bx_end])
            y0, y1 = sorted([board_ui.by_start, board_ui.by_end])
            roi_hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
            lo, up = runtime.calibrate_ammo_color(roi_hsv)
            print("[INFO] calibrado MUNICIÓN:", lo, up)
        else:
            print("[WARN] dibuja ROI sobre la munición antes de pulsar 'm'")

    # reiniciar manualmente el origen global detectado por ArUco
    elif key == ord("r"):
        runtime.reset_origin()
        print(
            f"[INFO] Origen global reiniciado. Esperando ArUco ID {runtime.aruco_id}..."
        )


if __name__ == "__main__":
    main()

