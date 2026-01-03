#!/usr/bin/env python3
"""
board_main_debug.py

Versión SIN ROS (sin rospy). Lee cámara con cv2.VideoCapture(1) y, al pulsar 's',
captura N frames, promedia detecciones (igual que tu pipeline actual) e imprime
por terminal el JSON que "se enviaría" por ROS.

Teclas:
- b: calibrar color del tablero (ROI)
- 2: calibrar barco x2 (ROI)
- 1: calibrar barco x1 (ROI)
- m: calibrar munición (ROI)
- s: iniciar captura/promedio y volcar JSON por terminal
- r: resetear calibraciones
- q / ESC: salir
"""

import os
import json
from collections import defaultdict, Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from board_config import USE_UNDISTORT_BOARD, BOARD_CAMERA_PARAMS_PATH, WARP_SIZE
import board_ui
import board_state
import board_processing as bp
import aruco_utils
import battleship_logic
import board_tracker


Cell = Tuple[int, int]


class LayoutAccumulator:
    def __init__(self, target_frames: int):
        self.target_frames = int(target_frames)
        self.reset()
        self.cell_cm = 3.85

    def reset(self) -> None:
        self.frame_count = 0
        self.data = defaultdict(
            lambda: {
                "board_size": None,
                "ship_two_counts": Counter(),
                "ship_one_counts": Counter(),
                "ammo_counts": Counter(),
                "ship_two_offsets": defaultdict(list),
                "ship_one_offsets": defaultdict(list),
                "ammo_offsets": defaultdict(list),
                "ship_two_pixels": defaultdict(list),
                "ship_one_pixels": defaultdict(list),
                "ammo_pixels": defaultdict(list),
                "ammo_global_offsets": [],
                "ammo_global_pixels": [],
                "ammo_global_xy": [],
                "board_corners_aruco": [],  # lista de frames; cada frame: [(x,y)*4]
                "ratio_cm_per_pix_list": [],
                "board_quad_pixel_frames": [],
            }
        )

    def push(self, layouts: List[dict]) -> None:
        self.frame_count += 1
        for layout in layouts:
            name = layout.get("name", f"board_{len(self.data)}")
            entry = self.data[name]

            if layout.get("board_size") is not None:
                entry["board_size"] = layout["board_size"]

            n = int(layout.get("board_size") or entry.get("board_size") or 5)
            for cell in layout.get("ship_two_cells", []):
                cell = tuple(cell)
                entry["ship_two_counts"][cell] += 1
            for cell in layout.get("ship_one_cells", []):
                cell = tuple(cell)
                entry["ship_one_counts"][cell] += 1

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

            corners = layout.get("board_corners_aruco")
            if corners and len(corners) == 4:
                entry["board_corners_aruco"].append([tuple(c) for c in corners])

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

            # quad en píxel (4 puntos)
            quad = layout.get("board_quad_pixel")
            if quad is not None and len(quad) == 4:
                try:
                    q = np.array(quad, dtype=np.float32)
                    # fuerza orden canónico (TL, TR, BR, BL)
                    q_ord = board_tracker.order_points(q)
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
    def _cells_with_type(
        ship_two_cells: List[Cell], ship_one_cells: List[Cell]
    ) -> List[dict]:
        cells = []
        for c, r in ship_two_cells:
            cells.append({"col": c, "row": r, "type": "ship_two"})
        for c, r in ship_one_cells:
            cells.append({"col": c, "row": r,"type": "ship_one"})
        return cells

    def build_layouts(self) -> List[dict]:
        layouts: List[dict] = []
        threshold = max(1, int(self.target_frames * 0.6))

        for name, entry in self.data.items():
            ship_two_cells = [
                cell for cell, count in entry["ship_two_counts"].items() if count >= threshold
            ]
            ship_one_cells = [
                cell for cell, count in entry["ship_one_counts"].items() if count >= threshold
            ]

            ship_two_positions = []
            for cell in ship_two_cells:
                ship_two_positions.append(
                    {
                        "cell": cell,
                        "mean_pixel": self._average_point(entry["ship_two_pixels"].get(cell, [])),
                        "mean_offset_from_origin": self._average_point(
                            entry["ship_two_offsets"].get(cell, [])
                        ),
                    }
                )

            ship_one_positions = []
            for cell in ship_one_cells:
                ship_one_positions.append(
                    {
                        "cell": cell,
                        "mean_pixel": self._average_point(entry["ship_one_pixels"].get(cell, [])),
                        "mean_offset_from_origin": self._average_point(
                            entry["ship_one_offsets"].get(cell, [])
                        ),
                    }
                )

            ammo_positions = []

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

            # Promedio de esquinas del tablero (4 esquinas)
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

            # promedio del ratio
            ratio_list = entry.get("ratio_cm_per_pix_list", [])
            mean_ratio = None
            if ratio_list:
                mean_ratio = sum(ratio_list) / float(len(ratio_list))

            # promedio del quad
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
                "ammo_positions": ammo_positions,
                "ammo_points_aruco": ammo_global_positions,
                "board_corners_aruco": mean_board_corners,
                "ratio_cm_per_pix": mean_ratio,
                "board_quad_pixel": mean_quad,
                "warp_size_px": WARP_SIZE,  # o el warp_size real si lo pasas
            }
            layouts.append(layout)

        return layouts


class BoardMainDebug:
    def __init__(self, cam_index: int = 1):
        self.cap = cv2.VideoCapture(int(cam_index))
        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la cámara con índice {cam_index}")

        self.capture_frames = 150
        self.capture_state = "WAIT_TRIGGER"  # WAIT_TRIGGER | CAPTURING | STANDBY
        self.capture_progress = 0.0
        self.status_lines = ["Ajusta HSV y pulsa 's' para capturar/volcar JSON."]
        self.accumulator: Optional[LayoutAccumulator] = None
        self.capture_reason = "manual"
        self.cell_cm = 3.85

        self.mtx = None
        self.dist = None
        if USE_UNDISTORT_BOARD and os.path.exists(BOARD_CAMERA_PARAMS_PATH):
            data = np.load(BOARD_CAMERA_PARAMS_PATH)
            self.mtx = data["camera_matrix"]
            self.dist = data["dist_coeffs"]

        self.board_state = board_state.init_board_state("T1")

        cv2.namedWindow("Tablero")
        cv2.setMouseCallback("Tablero", board_ui.board_mouse_callback)
        cv2.namedWindow("Mascara tablero")
        cv2.namedWindow("Mascara barco x2")
        cv2.namedWindow("Mascara barco x1")
        cv2.namedWindow("Mascara municion")

    @staticmethod
    def json_default(obj: Any):
        if isinstance(obj, tuple):
            return list(obj)
        raise TypeError

    def start_capture(self, reason: str) -> None:
        if self.capture_state == "CAPTURING":
            print("[board_main_debug] Captura ya en curso, ignorando")
            return

        print(f"[board_main_debug] Iniciando captura de layout ({reason})")
        self.capture_reason = reason
        self.accumulator = LayoutAccumulator(self.capture_frames)
        self.capture_state = "CAPTURING"
        self.capture_progress = 0.0
        self.status_lines = [
            f"Capturando {self.capture_frames} frames ({reason}).",
            "No muevas el tablero.",
        ]


    def _with_cartesian_coords(self, layout: dict) -> dict:
        board_size = layout.get("board_size") or board_tracker.BOARD_SQUARES
        origin = board_state.GLOBAL_ORIGIN

        layout = dict(layout)

        quad = layout.get("board_quad_pixel")
        warp_size = float(layout.get("warp_size_px", WARP_SIZE))
        ratio_cm_per_pix = layout.get("ratio_cm_per_pix")  # ya lo metes en process_board()

        cell_centers = []

        if origin is not None and quad is not None and ratio_cm_per_pix is not None:
            # reconstruye H_warp igual que en process_single_board
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
                    dx_px = float(ctr_img[0]) - float(ox)
                    dy_px = float(ctr_img[1]) - float(oy)

                    # px -> cm -> m
                    x_m = (dx_px * float(ratio_cm_per_pix)) / 100.0
                    y_m = (dy_px * float(ratio_cm_per_pix)) / 100.0

                    cell_centers.append({"col": col, "row": row, "xy_aruco": [x_m, y_m], "ctr_img_px": [float(ctr_img[0]), float(ctr_img[1])]})

        else:
            # fallback: tu rejilla ideal (lo que tenías), pero ya NO la llames "aruco"
            n = int(board_size)
            cell_size_m = self.cell_cm / 100.0
            for col in range(n):
                for row in range(n):
                    cell_centers.append(
                        {"col": col, "row": row, "xy_aruco": [(col + 0.5) * cell_size_m, (row + 0.5) * cell_size_m]})

        layout["cell_centers_aruco"] = cell_centers
        layout["cell_size_m"] = self.cell_cm / 100.0
        return layout

        # listas de pares [c,r]
        for key in ("ship_two_cells", "ship_one_cells"):
            out[key] = [list(rot(int(c), int(r))) for c, r in out.get(key, [])]

        # lista "cells" (dict)
        new_cells = []
        for e in out.get("cells", []) or []:
            try:
                c, r = int(e["col"]), int(e["row"])
                c2, r2 = rot(c, r)
                ne = dict(e);
                ne["col"] = c2;
                ne["row"] = r2
                new_cells.append(ne)
            except Exception:
                new_cells.append(e)
        out["cells"] = new_cells

        # positions con "cell"
        for key in ("ship_two_positions", "ship_one_positions", "ammo_positions"):
            new_lst = []
            for e in out.get(key, []) or []:
                try:
                    c, r = map(int, e["cell"])
                    c2, r2 = rot(c, r)
                    ne = dict(e);
                    ne["cell"] = [c2, r2]
                    new_lst.append(ne)
                except Exception:
                    new_lst.append(e)
            out[key] = new_lst

        # cell_centers_aruco: solo indices
        new_centers = []
        for e in out.get("cell_centers_aruco", []) or []:
            try:
                c, r = int(e["col"]), int(e["row"])
                c2, r2 = rot(c, r)
                ne = dict(e);
                ne["col"] = c2;
                ne["row"] = r2
                new_centers.append(ne)
            except Exception:
                new_centers.append(e)
        out["cell_centers_aruco"] = new_centers

        return out

    def emit_payload_to_terminal(self, layouts: List[dict]) -> None:
        boards = []
        for l in layouts:
            l2 = self._with_cartesian_coords(l)
            l2 = self._minimize_layout_for_robot(l2)
            boards.append(l2)
        payload = {"boards": boards}
        print("\n" + "=" * 25 + " JSON QUE SE ENVIARIA " + "=" * 25)
        print(json.dumps(payload, indent=2, default=self.json_default))
        print("=" * 78 + "\n")

    def _minimize_layout_for_robot(self, layout: dict) -> dict:
        """
        Reduce el layout a lo estrictamente necesario para RobotAttackExecutor.
        Elimina píxeles, offsets y duplicados que no se consumen.
        """
        out = {
            "name": layout.get("name", "T1"),
            "board_size": int(layout.get("board_size") or 5),

            # MoveIt board plane (4 corners in meters, ArUco frame)
            "board_corners_aruco": layout.get("board_corners_aruco", []),

            # Triangulación por celda: (col,row)->xy_aruco (meters)
            "cell_centers_aruco": [],

            # Obstáculos barcos por celdas
            "ship_two_cells": layout.get("ship_two_cells", []),
            "ship_one_cells": layout.get("ship_one_cells", []),

            # Munición fuera de tablero: lista de puntos en el frame ArUco (meters)
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

        # Recorta ammo_points_aruco: solo xy_aruco (mantengo id si existe para debug estable)
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

    def update_capture_state(self, layouts: List[dict]) -> None:
        if self.capture_state == "CAPTURING":
            if self.accumulator is None:
                self.accumulator = LayoutAccumulator(self.capture_frames)

            self.accumulator.push(layouts)
            self.capture_progress = self.accumulator.progress()

            if self.accumulator.ready():
                averaged = self.accumulator.build_layouts()
                if averaged:
                    # En esta versión debug: NO publicamos por ROS, imprimimos JSON
                    self.emit_payload_to_terminal(averaged)
                    self.status_lines = [
                        f"JSON impreso ({self.capture_reason}).",
                        "Standby.",
                    ]
                else:
                    self.status_lines = [
                        "No se detectó un tablero estable durante la captura.",
                        "Repite la operación cuando haya imagen.",
                    ]
                self.capture_state = "STANDBY"

        elif self.capture_state == "STANDBY":
            self.capture_progress = 0.0
            if not self.status_lines:
                self.status_lines = ["Standby."]
        else:
            self.capture_progress = 0.0
            if not self.status_lines:
                self.status_lines = ["Ajusta HSV y pulsa 's' para capturar/volcar JSON."]

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
            print("[INFO] RESET: limpiados rangos HSV (tablero/barcos/munición)")

    def spin(self) -> None:
        while True:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                print("[board_main_debug] No se pudo leer frame de cámara")
                break

            if self.mtx is not None and self.dist is not None:
                frame = cv2.undistort(frame, self.mtx, self.dist)

            # Actualiza GLOBAL_ORIGIN (ArUco) en coordenadas pixel
            aruco_utils.update_global_origin_from_aruco(frame, aruco_id=2)
            print(f"[DBG] GLOBAL_ORIGIN(px) = {board_state.GLOBAL_ORIGIN}")

            vis, mask_b, mask_ship2, mask_ship1, mask_m, layouts = bp.process_board(
                frame,
                self.board_state,
                cam_mtx=self.mtx,
                dist=self.dist,
                warp_size=WARP_SIZE,
                cell_cm=self.cell_cm
            )

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

                        # punto munición
                        cv2.circle(vis, (ax, ay), 5, (255, 255, 255), -1)

                        # vector ArUco -> munición
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
                ratio = layouts[0].get("ratio_cm_per_pix") if layouts else None

                if quad is not None and len(quad) == 4 and board_state.GLOBAL_ORIGIN is not None and ratio is not None:
                    gx, gy = map(int, board_state.GLOBAL_ORIGIN)
                    ox, oy = board_state.GLOBAL_ORIGIN

                    # Si tu quad ya viene ordenado por board_tracker.order_points, perfecto.
                    # Si no, lo ordenamos para tener TL,TR,BR,BL estable:
                    q = np.array(quad, dtype=np.float32)
                    q = board_tracker.order_points(q)  # TL,TR,BR,BL

                    corner_names = ["TL", "TR", "BR", "BL"]

                    for name, (px, py) in zip(corner_names, q):
                        px_f, py_f = float(px), float(py)

                        # offset px desde el ArUco
                        dx_px = px_f - float(ox)
                        dy_px = py_f - float(oy)

                        # a metros (xy_aruco)
                        x_m = (dx_px * float(ratio)) / 100.0
                        y_m = (dy_px * float(ratio)) / 100.0

                        cx, cy = int(px_f), int(py_f)

                        # punto esquina
                        cv2.circle(vis, (cx, cy), 6, (255, 255, 255), 2)

                        # vector ArUco -> esquina
                        cv2.line(vis, (gx, gy), (cx, cy), (200, 200, 200), 1)

                        # texto (3 líneas compactas como casillas)
                        txt1 = f"{name}"
                        txt2 = f"x{(x_m * 100):.2f}"
                        txt3 = f"y{(y_m * 100):.2f}"

                        cv2.putText(vis, txt1, (cx + 3, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)
                        cv2.putText(vis, txt2, (cx + 3, cy + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)
                        cv2.putText(vis, txt3, (cx + 3, cy + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)
            except Exception as exc:
                print("[WARN] overlay debug failed:", exc)

            # Evaluación (igual que antes)
            for layout in layouts:
                ok_board, msg = battleship_logic.evaluate_board(layout)
                print(f"[{layout.get('name','?')}] {msg}")

            # Debug: dibujar origen ArUco
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
                print("[board_main_debug] Saliendo por ESC/q")
                break

            self.handle_keys(key, frame)

        self.cap.release()
        cv2.destroyAllWindows()

    def _flip_rows_layout(self, layout: dict) -> dict:
        """
        Fuerza convención: (0,0) arriba-izquierda de pantalla.
        Si actualmente row está invertida (sale 4 cuando debería 0), aplicamos:
            row_new = (N-1) - row_old
        Se aplica a ship_*_cells, cells[] y a *_positions[].cell
        También rehace cell_centers_aruco en coherencia.
        """
        n = int(layout.get("board_size") or board_tracker.BOARD_SQUARES or 5)

        out = dict(layout)

        # Listas de celdas
        for key in ("ship_two_cells", "ship_one_cells"):
            cells = out.get(key, [])
            out[key] = [tuple(c) for c in cells]

        # Lista "cells" con type
        if "cells" in out and isinstance(out["cells"], list):
            new_cells = []
            for e in out["cells"]:
                try:
                    new_cells.append({**e, "row": n - 1 - int(e["row"])})
                except Exception:
                    new_cells.append(e)
            out["cells"] = new_cells

        # Positions: ship_two_positions, ship_one_positions, ammo_positions
        for key in ("ship_two_positions", "ship_one_positions", "ammo_positions"):
            lst = out.get(key, [])
            new_lst = []
            for e in lst:
                try:
                    cell = e.get("cell")
                    new_e = dict(e)
                    new_e["cell"] = cell
                    new_lst.append(new_e)
                except Exception:
                    new_lst.append(e)
            out[key] = new_lst

        # cell_centers_aruco (si existe): flip row del índice, pero OJO:
        # xy_aruco ya está en metros en el frame del ArUco; no debe cambiar.
        # Solo cambiamos el "row" asociado a ese xy.
        centers = out.get("cell_centers_aruco", [])
        if centers:
            new_centers = []
            for e in centers:
                try:
                    new_centers.append({**e, "row": n - 1 - int(e["row"])})
                except Exception:
                    new_centers.append(e)
            out["cell_centers_aruco"] = new_centers

        return out




def main() -> None:
    # Nota: si tu cámara no es el índice 1, cambia aquí: BoardMainDebug(cam_index=0/2/...)
    node = BoardMainDebug(cam_index=1)
    node.spin()


if __name__ == "__main__":
    main()
