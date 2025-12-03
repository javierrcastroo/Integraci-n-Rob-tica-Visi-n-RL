#!/usr/bin/env python3
"""Ejecutor de ataques: mueve el robot a la celda indicada por el usuario.

Este nodo escucha los ataques que se publican en ``battleship/attack`` (mensajes
JSON con dos gestos) y convierte la celda solicitada a coordenadas X/Y en el
espacio del robot. Para simplificar, se usa una cuadrícula parametrizable:

- ``~board_origin_x`` y ``~board_origin_y`` marcan la celda (0,0).
- ``~cell_size`` define la separación entre celdas.
- ``~hover_z`` es la altura de aproximación a la celda.

Ejemplo de configuración:

.. code-block:: bash

    rosrun robotica robot_attack_node.py _board_origin_x:=0.35 _board_origin_y:=0.0 _cell_size:=0.05

"""

import json
import rospy
from std_msgs.msg import String

from control_robot import ControlRobot


def _gesture_to_digit(label: str) -> int:
    """Convierte etiquetas tipo ``"3dedos"`` en el dígito correspondiente."""

    for ch in label:
        if ch.isdigit():
            return int(ch)
    raise ValueError(f"No se encontró dígito en la etiqueta de gesto: '{label}'")


class RobotAttackExecutor:
    def __init__(self) -> None:
        rospy.init_node("robot_attack_executor", anonymous=True)

        self.origin_x = rospy.get_param("~board_origin_x", 0.3)
        self.origin_y = rospy.get_param("~board_origin_y", 0.0)
        self.cell_size = rospy.get_param("~cell_size", 0.05)
        self.hover_z = rospy.get_param("~hover_z", 0.15)

        # Reutilizamos ControlRobot sin re-inicializar el nodo ROS.
        self.control = ControlRobot(init_ros_node=False)

        self.attack_sub = rospy.Subscriber(
            "battleship/attack", String, self.attack_cb, queue_size=10
        )
        self.board_request_pub = rospy.Publisher(
            "battleship/board_request", String, queue_size=10
        )

        rospy.loginfo(
            "[robot_attack_executor] Esperando ataques en 'battleship/attack' "
            "(origen=(%.3f, %.3f), paso=%.3f, z=%.3f)",
            self.origin_x,
            self.origin_y,
            self.cell_size,
            self.hover_z,
        )

    def attack_cb(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except Exception as exc:
            rospy.logwarn("[robot_attack_executor] Error parseando ataque: %s", exc)
            return

        gestures = data.get("gestures", [])
        if len(gestures) != 2:
            rospy.logwarn("[robot_attack_executor] Se esperaban 2 gestos, llegó: %s", gestures)
            return

        try:
            row = _gesture_to_digit(gestures[0])
            col = _gesture_to_digit(gestures[1])
        except ValueError as exc:
            rospy.logwarn("[robot_attack_executor] Gestos inválidos: %s", exc)
            return

        target_pose = self.control.pose_actual()
        target_pose.position.x = self.origin_x + col * self.cell_size
        target_pose.position.y = self.origin_y + row * self.cell_size
        target_pose.position.z = self.hover_z

        rospy.loginfo(
            "[robot_attack_executor] Moviendo a celda (r=%s, c=%s) -> (x=%.3f, y=%.3f, z=%.3f)",
            row,
            col,
            target_pose.position.x,
            target_pose.position.y,
            target_pose.position.z,
        )

        success = self.control.mover_a_pose(target_pose, wait=True)
        if not success:
            rospy.logwarn("[robot_attack_executor] No se pudo planificar el movimiento")
            return

        self.board_request_pub.publish(String("post_robot_attack"))
        rospy.loginfo("[robot_attack_executor] Petición de captura enviada tras mover el robot")


def main() -> None:
    executor = RobotAttackExecutor()
    rospy.spin()


if __name__ == "__main__":
    main()
