#!/usr/bin/env python3
"""Mueve el robot a una posición inicial leída desde un YAML."""

from typing import Any, Dict

import rospy
from geometry_msgs.msg import Pose

from control_robot import ControlRobot


def dict_to_pose(data: Dict[str, Any]) -> Pose:
    pose = Pose()
    try:
        position = data["position"]
        orientation = data["orientation"]

        pose.position.x = float(position["x"])
        pose.position.y = float(position["y"])
        pose.position.z = float(position["z"])

        pose.orientation.x = float(orientation["x"])
        pose.orientation.y = float(orientation["y"])
        pose.orientation.z = float(orientation["z"])
        pose.orientation.w = float(orientation["w"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Estructura inválida para la pose inicial") from exc

    return pose


class InitialPositionMover:
    def __init__(self) -> None:
        rospy.init_node("initial_position_mover", anonymous=True)

        self.control = ControlRobot(init_ros_node=False)

        parametros_posicion = rospy.get_param("Pos_Inicial", None)
        if not isinstance(parametros_posicion, dict):
            rospy.logerr(
                "[initial_position_mover] No se encontró el parámetro 'Pos_Inicial'"
            )
            return

        try:
            objetivo = dict_to_pose(parametros_posicion)
        except ValueError:
            rospy.logerr(
                "[initial_position_mover] Parámetros inválidos para 'Pos_Inicial'"
            )
            return

        rospy.loginfo(
            "[initial_position_mover] Moviendo a posición inicial "
            "(x=%.3f, y=%.3f, z=%.3f)",
            objetivo.position.x,
            objetivo.position.y,
            objetivo.position.z,
        )

        rospy.sleep(1.0)

        exito = self.control.mover_a_pose(objetivo, wait=True)
        if not exito:
            rospy.logwarn(
                "[initial_position_mover] Falló la trayectoria cartesiana inicial, "
                "reintentando con planificación directa"
            )
            self.control.move_group.set_pose_target(objetivo)
            exito = self.control.move_group.go(wait=True)
            self.control.move_group.stop()
            self.control.move_group.clear_pose_targets()

        if not exito:
            rospy.logwarn("[initial_position_mover] No se pudo ejecutar el movimiento inicial")

        rospy.signal_shutdown("Posición inicial alcanzada")


def main() -> None:
    InitialPositionMover()
    rospy.spin()


if __name__ == "__main__":
    main()
