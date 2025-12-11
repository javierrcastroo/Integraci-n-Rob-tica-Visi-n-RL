#!/usr/bin/env python3
"""Ejecutor de ataques: mueve el robot a la celda validada por ``game_logic_node``.

Este nodo espera los resultados publicados en ``battleship/attack_result`` para
mover el brazo solo cuando la jugada es válida. Además, proyecta los barcos
recibidos en ``battleship/board_layout`` como obstáculos en la escena de
planificación de MoveIt para evitar colisiones.

Parámetros relevantes:

- ``~board_origin_x`` y ``~board_origin_y``: coordenadas de la celda (0,0).
- ``~cell_size``: tamaño de cada celda.
- ``~hover_z``: altura de aproximación al atacar.
- ``~board_surface_z``: altura del plano del tablero para colocar obstáculos.
- ``~ship_box_size``: tamaño de los cubos que representan los barcos.
- ``~move_to_initial``: si es True, mueve al robot a ``/Pos_Inicial/joints`` al iniciar.
"""

import json
from typing import Iterable, List, Optional, Sequence, Set, Tuple

import rospy
from geometry_msgs.msg import Pose
from std_msgs.msg import String

from control_robot import ControlRobot


Cell = Tuple[int, int]


class RobotAttackExecutor:
    def __init__(self) -> None:
        rospy.init_node("robot_attack_executor", anonymous=True)

        self.origin_x = rospy.get_param("~board_origin_x", 0.3)
        self.origin_y = rospy.get_param("~board_origin_y", 0.0)
        self.cell_size = rospy.get_param("~cell_size", 0.05)
        self.hover_z = rospy.get_param("~hover_z", 0.15)
        self.board_surface_z = rospy.get_param("~board_surface_z", 0.0)
        self.ship_box_size = rospy.get_param("~ship_box_size", 0.025)
        self.move_to_initial = rospy.get_param("~move_to_initial", False)

        # Reutilizamos ControlRobot sin re-inicializar el nodo ROS.
        self.control = ControlRobot(init_ros_node=False)

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
            "[robot_attack_executor] Esperando resultados en 'battleship/attack_result' "
            "(origen=(%.3f, %.3f), paso=%.3f, z=%.3f)",
            self.origin_x,
            self.origin_y,
            self.cell_size,
            self.hover_z,
        )

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
        # Solo actuamos para jugadas que tienen sentido físico sobre el tablero.
        if result in {"board_invalid", "invalid_attack", "invalid_gestures", "out_of_bounds", "repeated"}:
            rospy.loginfo("[robot_attack_executor] Jugada sin movimiento (%s)", result)
            return

        row = cell_info.get("row")
        col = cell_info.get("col")
        if row is None or col is None:
            return

        target_pose = self._cell_to_hover_pose((int(row), int(col)))

        rospy.loginfo(
            "[robot_attack_executor] Moviendo a celda (r=%s, c=%s) -> (x=%.3f, y=%.3f, z=%.3f)",
            row,
            col,
            target_pose.position.x,
            target_pose.position.y,
            target_pose.position.z,
        )

        success = self.control.mover_trayectoria([target_pose], wait=True)
        if not success:
            rospy.logwarn("[robot_attack_executor] No se pudo planificar el movimiento lineal")
            return

        self.board_request_pub.publish(String("post_robot_attack"))
        rospy.loginfo("[robot_attack_executor] Petición de captura enviada tras mover el robot")

    def board_layout_cb(self, msg: String) -> None:
        """Crea obstáculos en la escena para cada barco detectado."""

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

    def _cell_to_hover_pose(self, cell: Cell) -> Pose:
        pose = self.control.pose_actual()
        row, col = cell
        pose.position.x = self.origin_x + col * self.cell_size
        pose.position.y = self.origin_y + row * self.cell_size
        pose.position.z = self.hover_z
        return pose

    def _cell_to_box_pose(self, cell: Cell) -> Pose:
        pose = Pose()
        row, col = cell
        pose.position.x = self.origin_x + col * self.cell_size
        pose.position.y = self.origin_y + row * self.cell_size
        pose.position.z = self.board_surface_z + self.ship_box_size / 2.0
        return pose

    def _update_obstacles(self, ship_cells: List[Cell]) -> None:
        # Elimina cajas anteriores
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

        rospy.loginfo(
            "[robot_attack_executor] Obstáculos actualizados: %s celdas", len(self.ship_boxes)
        )

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

        rospy.loginfo(
            "[robot_attack_executor] Moviendo a posición inicial definida en parámetros: %s",
            joints,
        )
        self.control.mover_articulaciones(joints, wait=True)

def main() -> None:
    executor = RobotAttackExecutor()
    rospy.spin()


if __name__ == "__main__":
    main()
