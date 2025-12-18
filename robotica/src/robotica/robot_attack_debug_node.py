#!/usr/bin/env python3
"""Versión debug del ejecutor: acepta celdas manuales y usa el layout simulado."""

import json
import math
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import rospy
from geometry_msgs.msg import Pose
from std_msgs.msg import String
from tf.transformations import euler_from_quaternion, quaternion_inverse, quaternion_matrix

from control_robot import ControlRobot

Cell = Tuple[int, int]


class RobotAttackDebug:
    def __init__(self) -> None:
        rospy.init_node("robot_attack_debug", anonymous=True)

        # --- Configuración fija del ArUco respecto a base_link ---
        self.aruco_origin_x = float(rospy.get_param("~aruco_origin_x", 0.30))
        self.aruco_origin_y = float(rospy.get_param("~aruco_origin_y", 0.45))
        self.aruco_yaw = float(rospy.get_param("~aruco_yaw", 0.0))
        self._load_aruco_pose_from_param()

        self.board_origin_dx = float(rospy.get_param("~board_origin_dx", 0.0))
        self.board_origin_dy = float(rospy.get_param("~board_origin_dy", 0.0))

        self.cell_size = float(rospy.get_param("~cell_size", 0.037))
        self.hover_z = float(rospy.get_param("~hover_z", 0.15))
        self.board_surface_z = float(rospy.get_param("~board_surface_z", 0.0))
        self.ship_box_size = float(rospy.get_param("~ship_box_size", 0.025))
        self.ammo_box_size = float(rospy.get_param("~ammo_box_size", self.ship_box_size))
        self.move_to_initial = bool(rospy.get_param("~move_to_initial", False))

        self.control = ControlRobot(init_ros_node=False)

        self.board_request_pub = rospy.Publisher(
            "battleship/board_request", String, queue_size=10
        )
        self.board_layout_sub = rospy.Subscriber(
            "battleship/board_layout", String, self.board_layout_cb, queue_size=10
        )
        self.target_sub = rospy.Subscriber(
            "battleship/debug_target", String, self.target_cb, queue_size=10
        )

        self.ship_boxes: Set[str] = set()
        self.ammo_boxes: Set[str] = set()
        self._last_ship_cells: Set[Cell] = set()
        self._last_ammo_cells: Set[Cell] = set()
        self.cell_xy_base: Dict[Cell, Tuple[float, float]] = {}

        if self.move_to_initial:
            self._move_to_initial_position()

        initial_target = rospy.get_param("~target_cell", None)
        if isinstance(initial_target, str) and initial_target:
            rospy.sleep(0.5)
            self._handle_target(initial_target.strip())

        rospy.loginfo(
            "[robot_attack_executor] SIN TF. ArUco fijo en base_link: (%.3f, %.3f), yaw=%.3f rad "
            "(cell=%.3f, hover_z=%.3f)",
            self.aruco_origin_x,
            self.aruco_origin_y,
            self.aruco_yaw,
            self.cell_size,
            self.hover_z,
        )

    # ------------------------- Helpers de transformación -------------------------
    def _board_to_base_xy(self, x_board: float, y_board: float) -> Tuple[float, float]:
        x_local = x_board + self.board_origin_dx
        y_local = y_board + self.board_origin_dy

        return self._aruco_to_base_xy(x_local, y_local)

    def _aruco_to_base_xy(self, x_aruco: float, y_aruco: float) -> Tuple[float, float]:
        """Transforma coordenadas en el frame del ArUco al frame base_link."""
        c = math.cos(self.aruco_yaw)
        s = math.sin(self.aruco_yaw)

        x_base = self.aruco_origin_x + (c * x_aruco - s * y_aruco)
        y_base = self.aruco_origin_y + (s * x_aruco + c * y_aruco)
        return x_base, y_base

    def _load_aruco_pose_from_param(self) -> None:
        pose_param = rospy.get_param("Pose_Actual", None)
        if not isinstance(pose_param, dict):
            rospy.logwarn("[robot_attack_executor] No se encontró Pose_Actual en parámetros, se usan valores por defecto")
            return

        pose_dict = pose_param.get("pose", {})
        pos_dict = pose_dict.get("position", {})
        ori_dict = pose_dict.get("orientation", {})

        try:
            t_robot_in_aruco = np.array(
                [
                    float(pos_dict.get("x", 0.0)),
                    float(pos_dict.get("y", 0.0)),
                    float(pos_dict.get("z", 0.0)),
                ]
            )

            quat_aruco_to_robot = [
                float(ori_dict.get("x", 0.0)),
                float(ori_dict.get("y", 0.0)),
                float(ori_dict.get("z", 0.0)),
                float(ori_dict.get("w", 1.0)),
            ]

            rotation_matrix = quaternion_matrix(quat_aruco_to_robot)[:3, :3]
            t_aruco_in_robot = -rotation_matrix.T @ t_robot_in_aruco

            quat_robot_to_aruco = quaternion_inverse(quat_aruco_to_robot)
            _, _, yaw = euler_from_quaternion(quat_robot_to_aruco)

            self.aruco_origin_x = float(t_aruco_in_robot[0])
            self.aruco_origin_y = float(t_aruco_in_robot[1])
            self.aruco_yaw = yaw
        except Exception as exc:
            rospy.logwarn(
                "[robot_attack_executor] No se pudo triangular Pose_Actual, se mantienen valores previos (error: %s)",
                exc,
            )
            return

        rospy.loginfo(
            "[robot_attack_executor] Pose del ArUco triangulada: (x=%.3f, y=%.3f, yaw=%.3f)",
            self.aruco_origin_x,
            self.aruco_origin_y,
            self.aruco_yaw,
        )

    def _cell_to_hover_pose(self, cell: Cell) -> Pose:
        x_base, y_base = self._cell_xy_base(cell)

        pose = self.control.pose_actual()
        pose.position.x = x_base
        pose.position.y = y_base
        pose.position.z = self.hover_z
        return pose

    def _cell_to_box_pose(self, cell: Cell, size: float) -> Pose:
        x_base, y_base = self._cell_xy_base(cell)

        pose = Pose()
        pose.position.x = x_base
        pose.position.y = y_base
        pose.position.z = self.board_surface_z + size / 2.0
        pose.orientation.w = 1.0
        return pose

    def _cell_xy_base(self, cell: Cell) -> Tuple[float, float]:
        if cell in self.cell_xy_base:
            return self.cell_xy_base[cell]

        row, col = cell
        x_board = col * self.cell_size
        y_board = row * self.cell_size
        return self._board_to_base_xy(x_board, y_board)

    # ------------------------- callbacks -------------------------
    def board_layout_cb(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except Exception as exc:
            rospy.logwarn("[robot_attack_executor] Error parseando layout: %s", exc)
            return
        boards = data.get("boards")
        if not boards:
            return
        layout = boards[0]
        self.cell_xy_base = self._build_cell_base_map(
            layout.get("cell_centers_aruco", []), layout.get("cell_size_m")
        )
        ship_cells = self._extract_cells(layout.get("ship_two_cells", []))
        ship_cells.update(self._extract_cells(layout.get("ship_one_cells", [])))
        ammo_cells = self._extract_cells(layout.get("ammo_cells", []))
        self._update_obstacles(ship_cells, ammo_cells)

    def target_cb(self, msg: String) -> None:
        text = msg.data.strip()
        if not text:
            return
        self._handle_target(text)

    # ------------------------- lógica principal -------------------------
    def _handle_target(self, raw_target: str) -> None:
        cell = self._parse_cell(raw_target)
        if cell is None:
            rospy.logwarn("[robot_attack_executor] Celda de destino inválida: %s", raw_target)
            return

        x_board = cell[1] * self.cell_size
        y_board = cell[0] * self.cell_size
        x_base, y_base = self._cell_xy_base(cell)

        self._log_triangulation(
            x_board=x_board,
            y_board=y_board,
            x_base=x_base,
            y_base=y_base,
        )

        target_pose = self._cell_to_hover_pose(cell)

        rospy.loginfo(
            "[robot_attack_executor] Moviendo a celda (r=%s, c=%s) -> (x=%.3f, y=%.3f, z=%.3f)",
            cell[0],
            cell[1],
            target_pose.position.x,
            target_pose.position.y,
            target_pose.position.z,
        )
        success = self.control.mover_en_linea_recta(
            target_pose,
            wait=True,
            pasos=200,
            z_constante=target_pose.position.z,
            eef_step=0.0075,
            intentos=4,
        )
        if not success:
            rospy.logwarn("[robot_attack_executor] No se pudo planificar el movimiento lineal")
            return

        self.board_request_pub.publish(String("post_robot_attack"))
        rospy.loginfo("[robot_attack_executor] Petición de captura enviada tras mover el robot")

    def _extract_cells(self, cells: Iterable[Sequence[int]]) -> Set[Cell]:
        result: Set[Cell] = set()
        for cell in cells:
            try:
                row, col = cell
                result.add((int(row), int(col)))
            except Exception:
                continue
        return result

    def _build_cell_base_map(
        self, centers_aruco: Iterable[dict], cell_size_m: Optional[float]
    ) -> Dict[Cell, Tuple[float, float]]:
        result: Dict[Cell, Tuple[float, float]] = {}
        if cell_size_m is not None:
            self.cell_size = cell_size_m

        for entry in centers_aruco or []:
            try:
                row = int(entry.get("row"))
                col = int(entry.get("col"))
                xy = entry.get("xy_aruco")
                x_aruco = float(xy[0])
                y_aruco = float(xy[1])
            except Exception:
                continue

            x_base, y_base = self._aruco_to_base_xy(x_aruco, y_aruco)
            result[(row, col)] = (x_base, y_base)

        return result

    def _update_obstacles(self, ship_cells: Set[Cell], ammo_cells: Set[Cell]) -> None:
        if ship_cells == self._last_ship_cells and ammo_cells == self._last_ammo_cells:
            return

        for name in self.ship_boxes:
            self.control.scene.remove_world_object(name)
        for name in self.ammo_boxes:
            self.control.scene.remove_world_object(name)
        self.ship_boxes.clear()
        self.ammo_boxes.clear()

        for row, col in ship_cells:
            name = f"ship_r{row}_c{col}"
            pose_caja = self._cell_to_box_pose((row, col), self.ship_box_size)
            self.control.añadir_caja_a_escena_de_planificacion(
                pose_caja, name, tamaño=(self.ship_box_size,) * 3
            )
            self.ship_boxes.add(name)

        for row, col in ammo_cells:
            name = f"ammo_r{row}_c{col}"
            pose_caja = self._cell_to_box_pose((row, col), self.ammo_box_size)
            self.control.añadir_caja_a_escena_de_planificacion(
                pose_caja, name, tamaño=(self.ammo_box_size,) * 3
            )
            self.ammo_boxes.add(name)

        rospy.loginfo(
            "[robot_attack_executor] Obstáculos actualizados: %s barcos, %s munición",
            len(self.ship_boxes),
            len(self.ammo_boxes),
        )
        self._last_ship_cells = set(ship_cells)
        self._last_ammo_cells = set(ammo_cells)

    def _move_to_initial_position(self) -> None:
        parametros = rospy.get_param("Pos_Inicial", None)
        joints: Optional[List[float]] = None

        if isinstance(parametros, dict):
            joints = parametros.get("joints")

        if not isinstance(joints, list) or len(joints) != 6:
            rospy.logwarn(
                "[robot_attack_executor] No se encontró Pos_Inicial/joints válido, se omite el movimiento inicial"
            )
            return

        rospy.loginfo("[robot_attack_executor] Moviendo a posición inicial: %s", joints)
        self.control.mover_articulaciones(joints, wait=True)

    # ------------------------- utils -------------------------
    @staticmethod
    def _parse_cell(text: str) -> Optional[Cell]:
        if not text:
            return None
        t = text.strip().upper()
        if len(t) < 2:
            return None
        letter = t[0]
        number = t[1:]
        if not letter.isalpha() or not number.isdigit():
            return None
        row = ord(letter) - ord("A")
        col = int(number) - 1
        if row < 0 or col < 0:
            return None
        return row, col

    def _log_triangulation(
        self, *, x_board: float, y_board: float, x_base: float, y_base: float
    ) -> None:
        """Emite trazas con las coordenadas relevantes para depuración."""

        rospy.loginfo(
            "[robot_attack_executor][debug] robot->aruco: (x=%.3f, y=%.3f, yaw=%.3f rad)",
            self.aruco_origin_x,
            self.aruco_origin_y,
            self.aruco_yaw,
        )

        rospy.loginfo(
            "[robot_attack_executor][debug] aruco->ficha: (x=%.3f, y=%.3f)",
            x_board,
            y_board,
        )

        rospy.loginfo(
            "[robot_attack_executor][debug] robot->ficha: (x=%.3f, y=%.3f)",
            x_base,
            y_base,
        )


def main() -> None:
    _node = RobotAttackDebug()
    rospy.spin()


if __name__ == "__main__":
    main()
