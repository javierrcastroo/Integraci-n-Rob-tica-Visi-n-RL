#!/usr/bin/env python3
"""Mueve el robot a una posición inicial definida por articulaciones en el YAML."""

from typing import Any, Dict, List
import rospy

from control_robot import ControlRobot


class InitialPositionMover:
    def __init__(self) -> None:
        rospy.init_node("initial_position_mover", anonymous=True)

        self.control = ControlRobot(init_ros_node=False)

        # Leemos "/"Pos_Inicial/joints"
        parametros = rospy.get_param("Pos_Inicial", None)

        if not isinstance(parametros, dict) or "joints" not in parametros:
            rospy.logerr("[initial_position_mover] No se encontró 'Pos_Inicial/joints' en parámetros.")
            return

        joints = parametros["joints"]

        if not isinstance(joints, list) or len(joints) != 6:
            rospy.logerr("[initial_position_mover] 'joints' debe ser una lista de 6 valores.")
            return

        rospy.loginfo(
            "[initial_position_mover] Moviendo a posición inicial con articulaciones: %s",
            joints,
        )

        rospy.sleep(1.0)

        exito = self.control.mover_articulaciones(joints, wait=True)

        if not exito:
            rospy.logwarn("[initial_position_mover] Falló el movimiento a la posición inicial.")

        rospy.signal_shutdown("Posición inicial alcanzada")


def main() -> None:
    InitialPositionMover()
    rospy.spin()


if __name__ == "__main__":
    main()
