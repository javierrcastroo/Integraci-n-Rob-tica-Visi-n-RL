#!/usr/bin/env python3
# board_main.py (ROS)

import os
import json
from collections import defaultdict, Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String

from board_config import USE_UNDISTORT_BOARD, BOARD_CAMERA_PARAMS_PATH, WARP_SIZE
import board_ui
import board_state
import board_processing as bp
import aruco_utils
import battleship_logic
import board_tracker


Cell = Tuple[int, int]


class LayoutAccumulator:
    """
    Acumulador (promedio temporal):
    - ship_two / ship_one por celdas (votes)
    - corners del tablero en aruco (ya viene en layout_info)
    - municion GLOBAL (xy_aruco) + pixel/offset para estabilizar IDs
    - ratio_cm_per_pix y board_quad_pixel (promedio)
    """

    def __init__(self, target_frames: int):
        self.target_frames = int(target_frames)
        self.reset()

    def reset(self) -> None:
        self.frame_count = 0
        self.data = defaultdict(
            lambda: {
                "board_size": None,
                "ship_two_counts": Counter(),
                "ship_one_counts": Counter(),
                "ship_two_offsets": defaultdict(list),
                "ship_one_offsets": defaultdict(list),
                "ship_two_pixels": defaultdict(list),
                "ship_one_pixels": defaultdict(list),

                # Global ammo
                "ammo_global_offsets": [],
                "ammo_global_pixels": [],
                "ammo_global_xy": [],

                # Board geometry
                "board_corners_aruco": [],       # lista de frames; cada frame: [(x,y)*4]
                "ratio_cm_per_pix_list": [],
                "board_quad_pixel_frames": [],   # lista de frames; cada frame: 4 puntos ordenados TL,TR,BR,BL
            }
        )

    def push(self, layouts: List[dict]) -> None:
        self.frame_count += 1
        for layout in layouts:
            name = layout.get("name", f"board_{len(self.data)}")
            entry = self.data[name]

            if layout.get("board_size") is not None:
                entry["board_size"] = layout["board_size"]

            for cell in layout.get("ship_two_cells", []):
                cell = tuple(cell)
                entry["ship_two_counts"][cell] += 1
            for cell in layout.get("ship_one_cells", []):
                cell = tuple(cell)
                entry["ship_one_counts"][cell] += 1

            # detections (pixel/offset) solo para estabilizar promedios/depuracion;
            # luego se eliminaran al minimizar JSON.
            for det in layout.get("ship_two_detections", []):
                cell = det.get("cell")
                if cell is None:
                    continue
                cell = tuple(cell)
                pixel = det.get("pixel")
                offset = det.get("offset_from_origin")
                if pixel is not None:
                    entry["ship_two_pixels"][cell].append(tuple(pixel))
                if offset is not None:
                    entry["ship_two_offsets"][cell].append(tuple(offset))

            for det in layout.get("ship_one_detections", []):
                cell = det.get("cell")
                if cell is None:
                    continue
                cell = tuple(cell)
                pixel = det.get("pixel")
                offset = det.get("offset_from_origin")
                if pixel is not None:
                    entry["ship_one_pixels"][cell].append(tuple(pixel))
                if offset is not None:
                    entry["ship_one_offsets"][cell].append(tuple(offset))

            # corners aruco
            corners = layout.get("board_corners_aruco")
            if corners and len(corners) == 4:
                entry["board_corners_aruco"].append([tuple(c) for c in corners])

            # global ammo detections
            global_ammo = layout.get("ammo_global_detections", [])
            if global_ammo:
                global_ammo = sorted(
                    global_ammo,
                    key=lambda det: (
                        det.get("pixel", det.get("offset_from_origin", (0, 0)))[0],
                        det.get("pixel", det.get("offset_from_origin", (0, 0)))[1],
                    ),
                )
                for idx, det in enumerate(global_ammo):
                    while len(entry["ammo_global_offsets"]) <= idx:
                        entry["ammo_global_offsets"].append([])
                        entry["ammo_global_pixels"].append([])
                        entry["ammo_global_xy"].append([])

                    pixel = det.get("pixel")
                    offset = det.get("offset_from_origin")
                    xy = det.get("xy_aruco")

                    if pixel is not None:
                        entry["ammo_global_pixels"][idx].append(tuple(pixel))
                    if offset is not None:
                        entry["ammo_global_offsets"][idx].append(tuple(offset))
                    if xy is not None:
                        entry["ammo_global_xy"][idx].append(tuple(xy))

            # ratio
            ratio = layout.get("ratio_cm_per_pix")
            if ratio is not None:
                try:
                    entry["ratio_cm_per_pix_list"].append(float(ratio))
                except Exception:
                    pass

            # quad pixel (ordenado estable)
            quad = layout.get("board_quad_pixel")
            if quad is not None and len(quad) == 4:
                try:
                    q = np.array(quad, dtype=np.float32)
                    q_ord = board_tracker.order_points(q)  # TL,TR,BR,BL
                    entry["board_quad_pixel_frames"].append([tuple(map(float, p)) for p in q_ord])
                except Exception:
                    pass

    def progress(self) -> float:
        if self.target_frames <= 0:
            return 1.0
        return min(1.0, float(self.frame_count) / float(self.target_frames))

    def ready(self) -> bool:
        return self.frame_count >= self.target_frames

    @staticmethod
    def _average_point(pts: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        if not pts:
            return None
        sx = sum(p[0] for p in pts)
        sy = sum(p[1] for p in pts)
        n = float(len(pts))
        return (sx / n, sy / n)

    @staticmethod
    def _cells_with_type(ship_two_cells: List[Cell], ship_one_cells: List[Cell]) -> List[dict]:
        # Importante: cell = (col,row)
        cells = []
        for c, r in ship_two_cells:
            cells.append({"col": c, "row": r, "type": "ship_two"})
        for c, r in ship_one_cells:
            cells.append({"col": c, "row": r, "type": "ship_one"})
        return cells

    def build_layouts(self) -> List[dict]:
        layouts: List[dict] = []
        threshold = max(1, int(self.target_frames * 0.6))

        for name, entry in self.data.items():
            ship_two_cells = [cell for cell, count in entry["ship_two_counts"].items() if count >= threshold]
            ship_one_cells = [cell for cell, count in entry["ship_one_counts"].items() if count >= threshold]

            ship_two_positions = []
            for cell in ship_two_cells:
                ship_two_positions.append(
                    {
                        "cell": cell,
                        "mean_pixel": self._average_point(entry["ship_two_pixels"].get(cell, [])),
                        "mean_offset_from_origin": self._average_point(entry["ship_two_offsets"].get(cell, [])),
                    }
                )

            ship_one_positions = []
            for cell in ship_one_cells:
                ship_one_positions.append(
                    {
                        "cell": cell,
                        "mean_pixel": self._average_point(entry["ship_one_pixels"].get(cell, [])),
                        "mean_offset_from_origin": self._average_point(entry["ship_one_offsets"].get(cell, [])),
                    }
                )

            ammo_global_positions = []
            for idx, xy_list in enumerate(entry["ammo_global_xy"]):
                mean_xy = self._average_point(xy_list)
                mean_pixel = self._average_point(entry["ammo_global_pixels"][idx])
                mean_offset = self._average_point(entry["ammo_global_offsets"][idx])
                if mean_xy is None and mean_pixel is None and mean_offset is None:
                    continue
                ammo_global_positions.append(
                    {
                        "id": idx,
                        "mean_pixel": mean_pixel,
                        "mean_offset_from_origin": mean_offset,
                        "xy_aruco": mean_xy,
                    }
                )

            # corners (promedio 4 esquinas)
            mean_board_corners: List[Tuple[float, float]] = []
            corners_frames = entry.get("board_corners_aruco", [])

            if corners_frames:
                for k in range(4):
                    pts_k = []
                    for frame_corners in corners_frames:
                        if frame_corners and len(frame_corners) == 4 and frame_corners[k] is not None:
                            pts_k.append(frame_corners[k])
                    mean_k = self._average_point(pts_k)
                    mean_board_corners.append(mean_k)

                if any(c is None for c in mean_board_corners):
                    mean_board_corners = []
            else:
                mean_board_corners = []

            # ratio promedio
            ratio_list = entry.get("ratio_cm_per_pix_list", [])
            mean_ratio = None
            if ratio_list:
                mean_ratio = sum(ratio_list) / float(len(ratio_list))

            # quad promedio
            quad_frames = entry.get("board_quad_pixel_frames", [])
            mean_quad = None
            if quad_frames:
                mean_quad = []
                for k in range(4):
                    xs = [q[k][0] for q in quad_frames]
                    ys = [q[k][1] for q in quad_frames]
                    mean_quad.append((sum(xs) / len(xs), sum(ys) / len(ys)))
                mean_quad = board_tracker.order_points(np.array(mean_quad, dtype=np.float32)).tolist()

            layout = {
                "name": name,
                "board_size": entry["board_size"],
                "ship_two_cells": sorted(ship_two_cells),
                "ship_one_cells": sorted(ship_one_cells),
                "cells": self._cells_with_type(ship_two_cells, ship_one_cells),
                "ship_two_positions": ship_two_positions,
                "ship_one_positions": ship_one_positions,

                # Global ammo (aruco frame)
                "ammo_points_aruco": ammo_global_positions,

                # Board plane
                "board_corners_aruco": mean_board_corners,

                # geometry debug needed for reconstructing centers
                "ratio_cm_per_pix": mean_ratio,
                "board_quad_pixel": mean_quad,
                "warp_size_px": WARP_SIZE,
            }
            layouts.append(layout)

        return layouts


class BoardMainNode:
    def __init__(self):
        self.bridge = CvBridge()
        self.last_frame = None

        self.capture_frames = 150
        self.capture_state = "WAIT_TRIGGER"
        self.capture_progress = 0.0
        self.status_lines = ["Ajusta HSV y pulsa 's' para enviar el tablero."]
        self.accumulator: Optional[LayoutAccumulator] = None
        self.capture_reason = "manual"
        self.debug = True

        # Igual que debug: cell_cm configurable
        self.cell_cm = float(rospy.get_param("~cell_cm", 3.85))

        image_topic = rospy.get_param("~image_topic", "/camara_tablero/usb_cam/image_raw")
        rospy.loginfo("[board_main] Suscribiendose a %s", image_topic)
        self.image_sub = rospy.Subscriber(image_topic, Image, self.cb_image, queue_size=1)

        self.board_pub = rospy.Publisher("battleship/board_layout", String, queue_size=10)
        self.request_sub = rospy.Subscriber("battleship/board_request", String, self.request_cb, queue_size=10)

        self.mtx = None
        self.dist = None
        self.new_mtx = None
        self.calib_size = None  # (w_calib, h_calib)

        if USE_UNDISTORT_BOARD and os.path.exists(BOARD_CAMERA_PARAMS_PATH):
            rospy.loginfo("[board_main] Cargando parametros de camara desde %s", BOARD_CAMERA_PARAMS_PATH)
            data = np.load(BOARD_CAMERA_PARAMS_PATH)

            self.mtx = data["camera_matrix"]
            self.dist = data["dist_coeffs"]

            # Si en el npz hemos guardado tambien el tamaño de calibracion:
            if "image_size" in data.files:
                # image_size = (w, h)
                self.calib_size = tuple(map(int, data["image_size"]))
                rospy.loginfo("[board_main] Tamaño de calibracion: %s", self.calib_size)
            else:
                rospy.logwarn("[board_main] El npz no tiene 'image_size'. Asumimos misma resolucion camara/calibracion.")


        self.board_state = board_state.init_board_state("T1")

        cv2.namedWindow("Tablero")
        cv2.setMouseCallback("Tablero", board_ui.board_mouse_callback)
        cv2.namedWindow("Mascara tablero")
        cv2.namedWindow("Mascara barco x2")
        cv2.namedWindow("Mascara barco x1")
        cv2.namedWindow("Mascara municion")

        self.loop_rate = rospy.Rate(rospy.get_param("~loop_rate", 30.0))

    def cb_image(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            rospy.logwarn("[board_main] Error CvBridge: %s", exc)
            return
        self.last_frame = frame

    def start_capture(self, reason: str):
        if self.capture_state == "CAPTURING":
            rospy.loginfo("[board_main] Captura ya en curso, ignorando")
            return

        rospy.loginfo("[board_main] Iniciando captura de layout (%s)", reason)
        self.capture_reason = reason
        self.accumulator = LayoutAccumulator(self.capture_frames)
        self.capture_state = "CAPTURING"
        self.capture_progress = 0.0
        self.status_lines = [
            f"Capturando {self.capture_frames} frames ({reason}).",
            "No muevas el tablero.",
        ]

    @staticmethod
    def json_default(obj: Any):
        if isinstance(obj, tuple):
            return list(obj)
        raise TypeError

    # -------------------------
    # Enriquecimiento (igual que debug)
    # -------------------------
    def _with_cartesian_coords(self, layout: dict) -> dict:
        """
        Genera cell_centers_aruco (col,row)->xy_aruco en metros:
        - Preferente: usando (GLOBAL_ORIGIN px) + (board_quad_pixel) + (ratio_cm_per_pix)
          reconstruyendo H_warp y proyectando centro de cada celda a imagen.
        - Fallback: rejilla ideal con self.cell_cm (NO geometrica real, solo aproximacion).
        """
        board_size = layout.get("board_size") or board_tracker.BOARD_SQUARES
        origin = board_state.GLOBAL_ORIGIN

        out = dict(layout)

        quad = out.get("board_quad_pixel")
        warp_size = float(out.get("warp_size_px", WARP_SIZE))
        ratio_cm_per_pix = out.get("ratio_cm_per_pix")

        cell_centers: List[dict] = []

        if origin is not None and quad is not None and len(quad) == 4 and ratio_cm_per_pix is not None:
            # --- 1) Ejes del tablero en píxeles (mismos que en board_processing) ---
            q = np.array(quad, dtype=np.float32)
            q = board_tracker.order_points(q)  # TL,TR,BR,BL
            (tl_x, tl_y), (tr_x, tr_y), (br_x, br_y), (bl_x, bl_y) = q

            vx1 = np.array([tr_x - tl_x, tr_y - tl_y], dtype=np.float32)
            vx2 = np.array([br_x - bl_x, br_y - bl_y], dtype=np.float32)
            vx = 0.5 * (vx1 + vx2)

            vy1 = np.array([bl_x - tl_x, bl_y - tl_y], dtype=np.float32)
            vy2 = np.array([br_x - tr_x, br_y - tr_y], dtype=np.float32)
            vy = 0.5 * (vy1 + vy2)

            norm_x = np.linalg.norm(vx)
            norm_y = np.linalg.norm(vy)
            if norm_x < 1e-6 or norm_y < 1e-6:
                print("WARNING: vectores de ejes del tablero degenerados, usando fallback de imagen.")
                ex = np.array([1.0, 0.0], dtype=np.float32)
                ey = np.array([0.0, 1.0], dtype=np.float32)
            else:
                ex = vx / norm_x
                ey = vy / norm_y
                cross_z = ex[0] * ey[1] - ex[1] * ey[0]
                if cross_z < 0:
                    ey = -ey

            # --- 2) Homografía para pasar de warp->imagen ---
            src = np.array(quad, dtype=np.float32)
            dst = np.array(
                [[0, 0],
                 [warp_size - 1, 0],
                 [warp_size - 1, warp_size - 1],
                 [0, warp_size - 1]],
                dtype=np.float32,
            )
            H_warp = cv2.getPerspectiveTransform(src, dst)
            H_inv = np.linalg.inv(H_warp)

            n = int(board_size)
            cell_size_px = warp_size / n
            ox, oy = origin

            for col in range(n):
                for row in range(n):
                    # centro de celda en warp
                    cxw = (col + 0.5) * cell_size_px
                    cyw = (row + 0.5) * cell_size_px

                    ctr_warp = np.array([[[cxw, cyw]]], dtype=np.float32)
                    ctr_img = cv2.perspectiveTransform(ctr_warp, H_inv).reshape(-1, 2)[0]

                    # vector desde ArUco a centro en coordenadas de imagen (px)
                    v = np.array(
                        [float(ctr_img[0]) - float(ox), float(ctr_img[1]) - float(oy)],
                        dtype=np.float32,
                    )

                    # Proyección de v sobre los ejes del tablero (px a lo largo de bordes)
                    proj_x_px = float(np.dot(v, ex))
                    proj_y_px = float(np.dot(v, ey))

                    # px -> cm -> m usando ratio_cm_per_pix
                    x_m = (proj_x_px * float(ratio_cm_per_pix)) / 100.0
                    y_m = (proj_y_px * float(ratio_cm_per_pix)) / 100.0

                    cell_centers.append(
                        {
                            "col": col,
                            "row": row,
                            "xy_aruco": [x_m, y_m],
                            # debug util para overlay; se elimina en minimizacion
                            "ctr_img_px": [float(ctr_img[0]), float(ctr_img[1])],
                        }
                    )
        else:
            # Fallback: no podemos reconstruir XY reales en frame ArUco porque falta geometria.
            rospy.logwarn("No podemos reconstruir XY reales en frame ArUco porque falta geometria.")

        out["cell_centers_aruco"] = cell_centers
        out["cell_size_m"] = self.cell_cm / 100.0
        return out

    def _minimize_layout_for_robot(self, layout: dict) -> dict:
        """
        Reduce el layout a lo estrictamente necesario para RobotAttackExecutor.
        Elimina pixeles, offsets y duplicados que no se consumen.
        """
        out = {
            "name": layout.get("name", "T1"),
            "board_size": int(layout.get("board_size") or 5),

            # MoveIt board plane (4 corners in meters, ArUco frame)
            "board_corners_aruco": layout.get("board_corners_aruco", []),

            # Triangulacion por celda: (col,row)->xy_aruco (meters)
            "cell_centers_aruco": [],

            # Obstaculos barcos por celdas
            "ship_two_cells": layout.get("ship_two_cells", []),
            "ship_one_cells": layout.get("ship_one_cells", []),

            # Municion fuera de tablero: lista de puntos en el frame ArUco (meters)
            "ammo_points_aruco": [],
        }

        # Recorta cell_centers_aruco: solo (col,row,xy_aruco)
        for e in layout.get("cell_centers_aruco", []) or []:
            try:
                col = int(e.get("col"))
                row = int(e.get("row"))
                xy = e.get("xy_aruco")
                if xy is None or len(xy) != 2:
                    continue
                x = float(xy[0])
                y = float(xy[1])
                out["cell_centers_aruco"].append({"col": col, "row": row, "xy_aruco": [x, y]})
            except Exception:
                continue

        # Recorta ammo_points_aruco: solo xy_aruco e id
        for a in layout.get("ammo_points_aruco", []) or []:
            try:
                xy = a.get("xy_aruco")
                if xy is None or len(xy) != 2:
                    continue
                entry = {"xy_aruco": [float(xy[0]), float(xy[1])]}
                if "id" in a:
                    entry["id"] = int(a["id"])
                out["ammo_points_aruco"].append(entry)
            except Exception:
                continue

        return out

    # -------------------------
    # Publicacion (ROS) - igual que debug pero publicando
    # -------------------------
    def publish_layouts(self, layouts: List[dict]) -> None:
        boards = []
        for l in layouts:
            l2 = self._with_cartesian_coords(l)
            l2 = self._minimize_layout_for_robot(l2)
            boards.append(l2)

        payload = {"boards": boards}
        msg = String()
        msg.data = json.dumps(payload, default=self.json_default)
        self.board_pub.publish(msg)
        rospy.loginfo("[board_main] Layouts enviados (%d tableros) tras captura.", len(boards))

    def update_capture_state(self, layouts: List[dict]):
        if self.capture_state == "CAPTURING":
            if self.accumulator is None:
                self.accumulator = LayoutAccumulator(self.capture_frames)

            self.accumulator.push(layouts)
            self.capture_progress = self.accumulator.progress()

            if self.accumulator.ready():
                averaged = self.accumulator.build_layouts()
                if averaged:
                    self.publish_layouts(averaged)
                    self.status_lines = [
                        f"Layout enviado ({self.capture_reason}).",
                        "Standby: esperando peticiones automaticas.",
                    ]
                else:
                    self.status_lines = [
                        "No se detecto un tablero estable durante la captura.",
                        "Repite la operacion cuando haya imagen.",
                    ]
                self.capture_state = "STANDBY"

        elif self.capture_state == "STANDBY":
            self.capture_progress = 0.0
            if not self.status_lines:
                self.status_lines = ["Standby: esperando peticion del juego."]
        else:
            self.capture_progress = 0.0
            if not self.status_lines:
                self.status_lines = ["Ajusta HSV y pulsa 's' para enviar el tablero."]

    def request_cb(self, msg):
        reason = msg.data if msg and msg.data else "peticion automatica"
        rospy.loginfo("[board_main] Peticion externa de layout: %s", reason)
        if self.capture_state == "WAIT_TRIGGER":
            self.status_lines = ["Peticion recibida, iniciando captura de tablero."]
        self.start_capture(reason)

    def handle_keys(self, key: int, frame: np.ndarray) -> None:
        import object_tracker
        import board_ui as bu

        if key == ord("b"):
            if bu.board_roi_defined:
                x0, x1 = sorted([bu.bx_start, bu.bx_end])
                y0, y1 = sorted([bu.by_start, bu.by_end])
                roi_hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
                lo, up = board_tracker.calibrate_board_color_from_roi(roi_hsv)
                board_tracker.current_ranges.append((lo, up))
                print(f"[INFO] calibrado TABLERO: {lo} {up} (rangos={len(board_tracker.current_ranges)})")
            else:
                print("[WARN] dibuja ROI en 'Tablero' primero")

        elif key == ord("2"):
            if bu.board_roi_defined:
                x0, x1 = sorted([bu.bx_start, bu.bx_end])
                y0, y1 = sorted([bu.by_start, bu.by_end])
                roi_hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
                lo, up = object_tracker.calibrate_ship_two_color_from_roi(roi_hsv)
                object_tracker.current_ship_two_ranges = [(lo, up)]
                print(f"[INFO] calibrado BARCO x2: {lo} {up}")
            else:
                print("[WARN] dibuja ROI sobre el barco largo")

        elif key == ord("1"):
            if bu.board_roi_defined:
                x0, x1 = sorted([bu.bx_start, bu.bx_end])
                y0, y1 = sorted([bu.by_start, bu.by_end])
                roi_hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
                lo, up = object_tracker.calibrate_ship_one_color_from_roi(roi_hsv)
                object_tracker.current_ship_one_ranges = [(lo, up)]
                print(f"[INFO] calibrado BARCO x1: {lo} {up}")
            else:
                print("[WARN] dibuja ROI sobre el barco corto")

        elif key == ord("m"):
            if bu.board_roi_defined:
                x0, x1 = sorted([bu.bx_start, bu.bx_end])
                y0, y1 = sorted([bu.by_start, bu.by_end])
                roi_hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
                lo, up = object_tracker.calibrate_ammo_color_from_roi(roi_hsv)
                object_tracker.current_ammo_ranges.append((lo, up))
                print(f"[INFO] calibrada MUNICION: {lo} {up} (rangos={len(object_tracker.current_ammo_ranges)})")
            else:
                print("[WARN] dibuja ROI sobre la municion")

        elif key == ord("s"):
            self.start_capture("tecla 's'")

        elif key == ord("r"):
            board_tracker.current_ranges = []
            object_tracker.current_ship_two_ranges = []
            object_tracker.current_ship_one_ranges = []
            object_tracker.current_ammo_ranges = []
            print("[INFO] RESET: limpiados rangos HSV (tablero/barcos/municion)")
            
        elif key == ord("d"):
            self.debug = not self.debug

    def spin(self):
        while not rospy.is_shutdown():
            frame = self.last_frame
            if frame is None:
                self.loop_rate.sleep()
                continue
            
            if self.mtx is not None and self.dist is not None:
                h, w = frame.shape[:2]

                # 1) Escalamos la camera_matrix si la resolucion actual es distinta de la de calibracion
                if self.calib_size is not None:
                    w_calib, h_calib = self.calib_size
                    if (w, h) != (w_calib, h_calib):
                        sx = float(w) / float(w_calib)
                        sy = float(h) / float(h_calib)
                        S = np.array([[sx, 0,  0],
                                      [0,  sy, 0],
                                      [0,  0,  1]], dtype=np.float32)
                        mtx_scaled = S @ self.mtx
                        # opcional: loguear una sola vez
                        if self.new_mtx is None:
                            rospy.logwarn("[board_main] Escalando camera_matrix de %sx%s a %sx%s", 
                                          w_calib, h_calib, w, h)
                    else:
                        mtx_scaled = self.mtx
                else:
                    # No sabemos el tamaño de calibracion, asumimos que coincide
                    mtx_scaled = self.mtx

                # 2) Calculamos newCameraMatrix una sola vez
                if self.new_mtx is None:
                    # alpha=0 -> menos zonas negras, recorta un poco; alpha=1 -> conserva todo
                    self.new_mtx, roi = cv2.getOptimalNewCameraMatrix(
                        mtx_scaled, self.dist, (w, h), alpha=0, newImgSize=(w, h)
                    )
                    rospy.loginfo("[board_main] Calculada newCameraMatrix para undistort. ROI=%s", roi)

                # 3) Undistort usando la matriz escalada y la new_mtx
                frame = cv2.undistort(frame, mtx_scaled, self.dist, None, self.new_mtx)


            # Actualiza GLOBAL_ORIGIN (ArUco) en coordenadas pixel
            aruco_utils.update_global_origin_from_aruco(frame, aruco_id=2)

            vis, mask_b, mask_ship2, mask_ship1, mask_m, layouts = bp.process_board(
                frame,
                self.board_state,
                cam_mtx=self.mtx,
                dist=self.dist,
                warp_size=WARP_SIZE,
                cell_cm=self.cell_cm
            )
            
            if self.debug :

                # =========================
                # VISUAL DEBUG
                # =========================
                try:
                    if layouts and board_state.GLOBAL_ORIGIN is not None:
                        # usa el layout "ya enriquecido"
                        l_dbg = self._with_cartesian_coords(layouts[0])

                        gx, gy = map(int, board_state.GLOBAL_ORIGIN)
                        # pinta el origen
                        cv2.circle(vis, (gx, gy), 6, (0, 255, 0), -1)

                        centers = l_dbg.get("cell_centers_aruco", [])
                        for e in centers:
                            col = int(e["col"]);
                            row = int(e["row"])
                            x_m, y_m = e["xy_aruco"]
                            px = e.get("ctr_img_px")
                            if not px:
                                continue
                            cx, cy = int(px[0]), int(px[1])

                            # punto centro
                            cv2.circle(vis, (cx, cy), 3, (255, 255, 255), -1)

                            # vector desde ArUco a centro
                            cv2.line(vis, (gx, gy), (cx, cy), (200, 200, 200), 1)

                            # etiqueta compacta: (c,r) y (x,y) en cm
                            txt1 = f"(c,r)=({col},{row})"
                            txt2 = f"x{(x_m * 100):.2f}"
                            txt3 = f"y{(y_m * 100):.2f}"

                            cv2.putText(vis, txt1, (cx + 3, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)
                            cv2.putText(vis, txt2, (cx + 3, cy + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)
                            cv2.putText(vis, txt3, (cx + 3, cy + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)

                    ammo_global = layouts[0].get("ammo_global_detections", []) if layouts else []
                    if ammo_global and board_state.GLOBAL_ORIGIN is not None:
                        gx, gy = map(int, board_state.GLOBAL_ORIGIN)

                        # orden estable para que el id sea consistente (izq->der, arriba->abajo)
                        ammo_global_sorted = sorted(
                            ammo_global,
                            key=lambda det: (
                                det.get("pixel", det.get("offset_from_origin", (0, 0)))[0],
                                det.get("pixel", det.get("offset_from_origin", (0, 0)))[1],
                            ),
                        )

                        for idx, det in enumerate(ammo_global_sorted):
                            px = det.get("pixel")  # (x_px, y_px) en imagen original
                            xy = det.get("xy_aruco")  # (x_m, y_m) respecto al ArUco (en metros)

                            if px is None:
                                continue

                            ax, ay = int(px[0]), int(px[1])

                            # punto municion
                            cv2.circle(vis, (ax, ay), 5, (255, 255, 255), -1)

                            # vector ArUco -> municion
                            cv2.line(vis, (gx, gy), (ax, ay), (200, 200, 200), 1)

                            # texto: id + xy en cm si existe
                            txt1 = f"AMMO#{idx}"
                            if xy is not None and len(xy) == 2:
                                x_cm = float(xy[0]) * 100.0
                                y_cm = float(xy[1]) * 100.0
                                txt2 = f"x{x_cm:.2f}"
                                txt3 = f"y{y_cm:.2f}"
                            else:
                                txt2 = "x?"
                                txt3 = "y?"

                            cv2.putText(vis, txt1, (ax + 3, ay - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)
                            cv2.putText(vis, txt2, (ax + 3, ay + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)
                            cv2.putText(vis, txt3, (ax + 3, ay + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)
                    # =========================
                    # ESQUINAS DEL TABLERO: overlay ArUco -> esquinas
                    # =========================
                    quad = layouts[0].get("board_quad_pixel") if layouts else None
                    corners_xy = layouts[0].get("board_corners_aruco") if layouts else None

                    if (
                            quad is not None
                            and len(quad) == 4
                            and corners_xy is not None
                            and len(corners_xy) == 4
                            and board_state.GLOBAL_ORIGIN is not None
                    ):
                        gx, gy = map(int, board_state.GLOBAL_ORIGIN)

                        q = np.array(quad, dtype=np.float32)
                        q = board_tracker.order_points(q)  # TL,TR,BR,BL

                        corner_names = ["TL", "TR", "BR", "BL"]

                        for name, (px, py), (x_m, y_m) in zip(corner_names, q, corners_xy):
                            cx, cy = int(px), int(py)

                            # punto esquina
                            cv2.circle(vis, (cx, cy), 6, (255, 255, 255), 2)

                            # vector ArUco -> esquina (solo visual)
                            cv2.line(vis, (gx, gy), (cx, cy), (200, 200, 200), 1)

                            # texto (3 líneas compactas como casillas)
                            txt1 = f"{name}"
                            txt2 = f"x{(x_m * 100):.2f}"
                            txt3 = f"y{(y_m * 100):.2f}"

                            cv2.putText(vis, txt1, (cx + 3, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                                        (255, 255, 255), 1)
                            cv2.putText(vis, txt2, (cx + 3, cy + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                                        (255, 255, 255), 1)
                            cv2.putText(vis, txt3, (cx + 3, cy + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                                        (255, 255, 255), 1)
                except Exception as exc:
                    print("[WARN] overlay debug failed:", exc)

            validation_map = {}
            # Evaluacion (igual que antes)
            for layout in layouts:
                ok, msg = battleship_logic.evaluate_board(layout)
                validation_map[layout.get("name", "?")] = (ok, msg)
                print(f"[{layout.get('name','?')}] {msg}")

            if self.board_state["name"] in validation_map and self.board_state["last_quad"] is not None:
                ok, msg = validation_map[self.board_state["name"]]
                board_ui.draw_validation_result(vis, self.board_state["last_quad"], msg, ok)

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

            self.update_capture_state(layouts)

            board_ui.draw_board_hud(vis)
            board_ui.draw_capture_status(vis, self.capture_state, self.capture_progress, self.status_lines)

            cv2.imshow("Tablero", vis)
            if mask_b is not None:
                cv2.imshow("Mascara tablero", mask_b)
            if mask_ship2 is not None:
                cv2.imshow("Mascara barco x2", mask_ship2)
            if mask_ship1 is not None:
                cv2.imshow("Mascara barco x1", mask_ship1)
            if mask_m is not None:
                cv2.imshow("Mascara municion", mask_m)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                rospy.loginfo("[board_main] Saliendo por ESC/q")
                break

            self.handle_keys(key, frame)

            self.loop_rate.sleep()

        cv2.destroyAllWindows()


def main():
    rospy.init_node("board_main_viewer", anonymous=False)
    node = BoardMainNode()
    rospy.loginfo("[board_main] Nodo de tablero iniciado.")
    node.spin()


if __name__ == "__main__":
    main()
