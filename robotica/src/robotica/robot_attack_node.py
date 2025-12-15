#!/usr/bin/env python3
"""Ejecutor de ataques: mueve el robot a la celda validada por ``game_logic_node``.

Versión SIN TF: el ArUco se asume fijo respecto a base_link y se configura por parámetros.
Incluye yaw opcional (rotación alrededor de Z) para alinear tablero y robot.
"""

import json
import math
from typing import Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import rospy
from geometry_msgs.msg import Pose
from std_msgs.msg import String
from tf.transformations import euler_from_quaternion, quaternion_inverse, quaternion_matrix

from control_robot import ControlRobot

Cell = Tuple[int, int]


class RobotAttackExecutor:
    def __init__(self) -> None:
        rospy.init_node("robot_attack_executor", anonymous=True)

        # --- Configuración fija del ArUco respecto a base_link (SIN TF) ---
        # Offset del centro del ArUco expresado en base_link (metros)
        self.aruco_origin_x = float(rospy.get_param("~aruco_origin_x", 0.30))
        self.aruco_origin_y = float(rospy.get_param("~aruco_origin_y", 0.45))

        # Yaw del tablero/aruco respecto a base_link (rad). 0.0 => sin rotación.
        self.aruco_yaw = float(rospy.get_param("~aruco_yaw", 0.0))

        # Carga inicial del ArUco desde poseAruco.yaml (parámetro Pose_Actual)
        self._load_aruco_pose_from_param()

        # Si tu (0,0) del tablero NO coincide con el centro del ArUco, añade offsets:
        # (por defecto 0.0, 0.0)
        self.board_origin_dx = float(rospy.get_param("~board_origin_dx", 0.0))
        self.board_origin_dy = float(rospy.get_param("~board_origin_dy", 0.0))

        # Tamaño de celda, alturas y obstáculos
        self.cell_size = float(rospy.get_param("~cell_size", 0.037))
        self.hover_z = float(rospy.get_param("~hover_z", 0.15))
        self.board_surface_z = float(rospy.get_param("~board_surface_z", 0.0))
        self.ship_box_size = float(rospy.get_param("~ship_box_size", 0.025))
        self.move_to_initial = bool(rospy.get_param("~move_to_initial", False))

        # Control
        self.control = ControlRobot(init_ros_node=False)

        # Subs/Pubs
        self.attack_result_sub = rospy.Subscriber(
            "battleship/attack_result", String, self.attack_result_cb, queue_size=10
        )
        self.board_layout_sub = rospy.Subscriber(
            "battleship/board_layout", String, self.board_layout_cb, queue_size=10
        )
        self.board_request_pub = rospy.Publisher(
            "battleship/board_request", String, queue_size=10
        )

        self.ship_boxes: Set[str] = set()

        if self.move_to_initial:
            self._move_to_initial_position()

        rospy.loginfo(
            "[robot_attack_executor] SIN TF. ArUco fijo en base_link: (%.3f, %.3f), yaw=%.3f rad "
            "(cell=%.3f, hover_z=%.3f)",
            self.aruco_origin_x,
            self.aruco_origin_y,
            self.aruco_yaw,
            self.cell_size,
            self.hover_z,
        )

    # -------------------------
    # Helpers de transformación (tablero -> base_link)
    # -------------------------

    def _board_to_base_xy(self, x_board: float, y_board: float) -> Tuple[float, float]:
        """
        Convierte coordenadas (x_board, y_board) expresadas en el frame del tablero/aruco
        a coordenadas (x_base, y_base) en base_link, usando:
          - traslación fija (aruco_origin_x, aruco_origin_y)
          - yaw fijo (aruco_yaw)
        """
        # Si el origen (0,0) del tablero no coincide con el centro del ArUco,
        # aplicamos el desplazamiento local (en frame tablero) antes de rotar.
        x_local = x_board + self.board_origin_dx
        y_local = y_board + self.board_origin_dy

        c = math.cos(self.aruco_yaw)
        s = math.sin(self.aruco_yaw)

        # Rotación 2D + traslación
        x_base = self.aruco_origin_x + (c * x_local - s * y_local)
        y_base = self.aruco_origin_y + (s * x_local + c * y_local)
        return x_base, y_base

    def _load_aruco_pose_from_param(self) -> None:
        """Carga la pose inicial del ArUco desde ``Pose_Actual``.

        El fichero ``poseAruco.yaml`` se carga en el parámetro ``Pose_Actual``.
        Se usa su ``position`` como traslación del ArUco respecto a ``base_link``
        y se extrae el yaw de su orientación para las rotaciones del tablero.
        """

        pose_param = rospy.get_param("Pose_Actual", None)
        if not isinstance(pose_param, dict):
            rospy.logwarn(
                "[robot_attack_executor] No se encontró Pose_Actual en parámetros, se usan valores por defecto"
            )
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
        # Coordenadas de celda en frame del tablero (convención: col->X, row->Y)
        row, col = cell
        x_board = col * self.cell_size
        y_board = row * self.cell_size

        x_base, y_base = self._board_to_base_xy(x_board, y_board)

        pose = self.control.pose_actual()
        pose.position.x = x_base
        pose.position.y = y_base
        pose.position.z = self.hover_z
        return pose

    def _cell_to_box_pose(self, cell: Cell) -> Pose:
        row, col = cell
        x_board = col * self.cell_size
        y_board = row * self.cell_size

        x_base, y_base = self._board_to_base_xy(x_board, y_board)

        pose = Pose()
        pose.position.x = x_base
        pose.position.y = y_base
        pose.position.z = self.board_surface_z + self.ship_box_size / 2.0
        return pose

    # -------------------------
    # Callbacks
    # -------------------------

    def attack_result_cb(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except Exception as exc:
            rospy.logwarn("[robot_attack_executor] Error parseando ataque: %s", exc)
            return

        if data.get("status") != "OK":
            return

        if not data.get("board_valid", False):
            rospy.loginfo("[robot_attack_executor] Tablero no válido, no se mueve el robot")
            return

        cell_info = data.get("cell")
        if not cell_info:
            return

        result = data.get("result")
        if result in {"board_invalid", "invalid_attack", "invalid_gestures", "out_of_bounds", "repeated"}:
            rospy.loginfo("[robot_attack_executor] Jugada sin movimiento (%s)", result)
            return

        row = cell_info.get("row")
        col = cell_info.get("col")
        if row is None or col is None:
            return

        row = int(row)
        col = int(col)

        # Coordenadas tablero (Aruco -> ficha) y triangulación hasta el robot
        x_board = col * self.cell_size
        y_board = row * self.cell_size
        x_base, y_base = self._board_to_base_xy(x_board, y_board)

        self._log_triangulation(
            x_board=x_board,
            y_board=y_board,
            x_base=x_base,
            y_base=y_base,
        )

        target_pose = self._cell_to_hover_pose((row, col))

        rospy.loginfo(
            "[robot_attack_executor] Moviendo a celda (r=%s, c=%s) -> (x=%.3f, y=%.3f, z=%.3f)",
            row,
            col,
            target_pose.position.x,
            target_pose.position.y,
            target_pose.position.z,
        )

        success = self.control.mover_trayectoria(
            [target_pose], wait=True, pasos=200, z_constante=target_pose.position.z
        )
        if not success:
            rospy.logwarn("[robot_attack_executor] No se pudo planificar el movimiento lineal")
            return

        self.board_request_pub.publish(String("post_robot_attack"))
        rospy.loginfo("[robot_attack_executor] Petición de captura enviada tras mover el robot")

    # -------------------------
    # Debug helpers
    # -------------------------

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
        ship_cells = self._extract_cells(layout.get("ship_two_cells", []))
        ship_cells.extend(self._extract_cells(layout.get("ship_one_cells", [])))

        self._update_obstacles(ship_cells)

    def _extract_cells(self, cells: Iterable[Sequence[int]]) -> List[Cell]:
        result: List[Cell] = []
        for cell in cells:
            try:
                row, col = cell
                result.append((int(row), int(col)))
            except Exception:
                continue
        return result

    def _update_obstacles(self, ship_cells: List[Cell]) -> None:
        for name in self.ship_boxes:
            self.control.scene.remove_world_object(name)
        self.ship_boxes.clear()

        for row, col in ship_cells:
            name = f"ship_r{row}_c{col}"
            pose_caja = self._cell_to_box_pose((row, col))
            self.control.añadir_caja_a_escena_de_planificacion(
                pose_caja, name, tamaño=(self.ship_box_size,) * 3
            )
            self.ship_boxes.add(name)

        rospy.loginfo("[robot_attack_executor] Obstáculos actualizados: %s celdas", len(self.ship_boxes))

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


def main() -> None:
    _executor = RobotAttackExecutor()
    rospy.spin()


if __name__ == "__main__":
    main()

