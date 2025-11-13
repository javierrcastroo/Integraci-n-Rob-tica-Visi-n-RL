"""Runtime compartido entre la versión local y el nodo ROS del tablero."""
from __future__ import annotations

from typing import List, Tuple

import cv2

import aruco_util
import board_processing as bp
import board_state
import board_tracker
import object_tracker
from board_config import WARP_SIZE


class BoardRuntime:
    def __init__(self, max_boards: int = 2, warp_size: int = WARP_SIZE, aruco_id: int = aruco_util.ARUCO_ORIGIN_ID):
        self.max_boards = max_boards
        self.warp_size = warp_size
        self.aruco_id = aruco_id
        self.boards_state: List[dict] = [
            board_state.init_board_state(f"T{i + 1}") for i in range(max_boards)
        ]

    def process_frame(self, frame, *, update_origin: bool = True):
        if update_origin:
            aruco_util.update_global_origin_from_aruco(frame, aruco_id=self.aruco_id)
        return bp.process_all_boards(
            frame,
            self.boards_state,
            cam_mtx=None,
            dist=None,
            max_boards=self.max_boards,
            warp_size=self.warp_size,
        )

    @staticmethod
    def draw_origin_indicator(image):
        if board_state.GLOBAL_ORIGIN is None:
            return
        gx, gy = board_state.GLOBAL_ORIGIN
        cv2.circle(image, (int(gx), int(gy)), 10, (0, 255, 0), -1)
        cv2.putText(
            image,
            "ORIGEN (ArUco)",
            (int(gx) + 10, int(gy) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    @staticmethod
    def reset_origin():
        board_state.GLOBAL_ORIGIN = None
        board_state.GLOBAL_ORIGIN_MISS = board_state.GLOBAL_ORIGIN_MAX_MISS + 1

    @staticmethod
    def calibrate_board_color(roi_hsv) -> Tuple:
        lo, up = board_tracker.calibrate_board_color_from_roi(roi_hsv)
        board_tracker.current_lower, board_tracker.current_upper = lo, up
        return lo, up

    @staticmethod
    def calibrate_object_color(roi_hsv) -> Tuple:
        lo, up = object_tracker.calibrate_object_color_from_roi(roi_hsv)
        object_tracker.current_obj_lower, object_tracker.current_obj_upper = lo, up
        return lo, up

    @staticmethod
    def calibrate_ship_color(ship_type: str, roi_hsv) -> Tuple:
        return object_tracker.calibrate_ship_color_from_roi(ship_type, roi_hsv)

    @staticmethod
    def calibrate_ammo_color(roi_hsv) -> Tuple:
        return object_tracker.calibrate_ammo_color_from_roi(roi_hsv)
