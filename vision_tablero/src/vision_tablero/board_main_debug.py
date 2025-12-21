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
                entry["ship_two_counts"][tuple(cell)] += 1
            for cell in layout.get("ship_one_cells", []):
                entry["ship_one_counts"][tuple(cell)] += 1
            for cell in layout.get("ammo_cells", []):
                entry["ammo_counts"][tuple(cell)] += 1

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

            for det in layout.get("ammo_detections", []):
                cell = det.get("cell")
                if cell is None:
                    continue
                cell = tuple(cell)
                pixel = det.get("pixel")
                offset = det.get("offset_from_origin")
                if pixel is not None:
                    entry["ammo_pixels"][cell].append(tuple(pixel))
                if offset is not None:
                    entry["ammo_offsets"][cell].append(tuple(offset))

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
        ship_two_cells: List[Cell], ship_one_cells: List[Cell], ammo_cells: List[Cell]
    ) -> List[dict]:
        cells = []
        for r, c in ship_two_cells:
            cells.append({"row": r, "col": c, "type": "ship_two"})
        for r, c in ship_one_cells:
            cells.append({"row": r, "col": c, "type": "ship_one"})
        for r, c in ammo_cells:
            cells.append({"row": r, "col": c, "type": "ammo"})
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
            ammo_cells = [
                cell for cell, count in entry["ammo_counts"].items() if count >= threshold
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
            for cell in ammo_cells:
                ammo_positions.append(
                    {
                        "cell": cell,
                        "mean_pixel": self._average_point(entry["ammo_pixels"].get(cell, [])),
                        "mean_offset_from_origin": self._average_point(
                            entry["ammo_offsets"].get(cell, [])
                        ),
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

            layout = {
                "name": name,
                "board_size": entry["board_size"],
                "ship_two_cells": sorted(ship_two_cells),
                "ship_one_cells": sorted(ship_one_cells),
                "ammo_cells": sorted(ammo_cells),
                "cells": self._cells_with_type(ship_two_cells, ship_one_cells, ammo_cells),
                "ship_two_positions": ship_two_positions,
                "ship_one_positions": ship_one_positions,
                "ammo_positions": ammo_positions,
                "ammo_points_aruco": ammo_global_positions,
                "board_corners_aruco": mean_board_corners,
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
        cell_size_m = float(getattr(board_tracker, "SQUARE_SIZE_CM", 3.7)) / 100.0

        cell_centers = []
        for row in range(int(board_size)):
            for col in range(int(board_size)):
                x_aruco = (col + 0.5) * cell_size_m
                y_aruco = (row + 0.5) * cell_size_m
                cell_centers.append({"row": row, "col": col, "xy_aruco": [x_aruco, y_aruco]})

        ammo_centers = []
        for cell in layout.get("ammo_cells", []):
            try:
                row, col = int(cell[0]), int(cell[1])
            except Exception:
                continue
            x_aruco = (col + 0.5) * cell_size_m
            y_aruco = (row + 0.5) * cell_size_m
            ammo_centers.append({"row": row, "col": col, "xy_aruco": [x_aruco, y_aruco]})

        layout = dict(layout)
        layout["cell_centers_aruco"] = cell_centers
        layout["ammo_centers_aruco"] = ammo_centers
        layout["ammo_points_aruco"] = layout.get("ammo_points_aruco", [])
        layout["board_corners_aruco"] = layout.get("board_corners_aruco", [])
        layout["cell_size_m"] = cell_size_m
        return layout

    def emit_payload_to_terminal(self, layouts: List[dict]) -> None:
        payload = {"boards": [self._with_cartesian_coords(l) for l in layouts]}
        print("\n" + "=" * 25 + " JSON QUE SE ENVIARIA " + "=" * 25)
        print(json.dumps(payload, indent=2, default=self.json_default))
        print("=" * 78 + "\n")

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
            aruco_utils.update_global_origin_from_aruco(frame, aruco_id=3)

            vis, mask_b, mask_ship2, mask_ship1, mask_m, layouts = bp.process_board(
                frame,
                self.board_state,
                cam_mtx=self.mtx,
                dist=self.dist,
                warp_size=WARP_SIZE,
            )

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


def main() -> None:
    # Nota: si tu cámara no es el índice 1, cambia aquí: BoardMainDebug(cam_index=0/2/...)
    node = BoardMainDebug(cam_index=1)
    node.spin()


if __name__ == "__main__":
    main()
