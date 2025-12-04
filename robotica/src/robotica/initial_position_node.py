#!/usr/bin/env python3
"""Mueve el robot a una posición inicial leída desde un YAML."""

import rospy

from control_robot import ControlRobot


class InitialPositionMover:
    def __init__(self) -> None:
        rospy.init_node("initial_position_mover", anonymous=True)

        self.control = ControlRobot(init_ros_node=False)

        parametros_posicion = rospy.get_param("start_position", None)
        if not isinstance(parametros_posicion, dict):
            rospy.logerr(
                "[initial_position_mover] No se encontró el parámetro 'start_position'"
            )
            return

        try:
            objetivo = self.control.pose_actual()
            objetivo.position.x = float(parametros_posicion.get("x"))
            objetivo.position.y = float(parametros_posicion.get("y"))
            objetivo.position.z = float(parametros_posicion.get("z"))
        except (TypeError, ValueError):
            rospy.logerr(
                "[initial_position_mover] Parámetros inválidos para 'start_position'"
            )
            return

        rospy.loginfo(
            "[initial_position_mover] Moviendo a posición inicial x=%.3f y=%.3f z=%.3f",
            objetivo.position.x,
            objetivo.position.y,
            objetivo.position.z,
        )

        exito = self.control.mover_a_pose(objetivo, wait=True)
        if not exito:
            rospy.logwarn("[initial_position_mover] No se pudo ejecutar el movimiento inicial")

        rospy.signal_shutdown("Posición inicial alcanzada")


def main() -> None:
    InitialPositionMover()
    rospy.spin()


if __name__ == "__main__":
    main()
