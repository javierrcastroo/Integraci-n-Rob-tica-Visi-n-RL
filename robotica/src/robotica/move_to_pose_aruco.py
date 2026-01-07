#!/usr/bin/env python3
"""
Mueve el robot a una pose cartesiana definida en el YAML,
usando el método mover_a_pose de ControlRobot.
"""

from typing import Any, Dict
import rospy
from geometry_msgs.msg import Pose

from control_robot import ControlRobot


class MoveToPoseAruco:
    def __init__(self) -> None:
        rospy.init_node("move_to_pose_aruco", anonymous=True)

        # No volvemos a inicializar el nodo dentro de ControlRobot
        self.control = ControlRobot(init_ros_node=False)

        # Leemos "Pose_Actual/pose" del servidor de parámetros
        params = rospy.get_param("Pose_Actual", None)

        if not isinstance(params, dict) or "pose" not in params:
            rospy.logerr("[move_to_pose_aruco] No se encontró 'Pose_Actual/pose' en parámetros.")
            rospy.signal_shutdown("Parámetros inválidos")
            return

        pose_dict = params["pose"]
        if not isinstance(pose_dict, dict):
            rospy.logerr("[move_to_pose_aruco] 'Pose_Actual/pose' debe ser un diccionario.")
            rospy.signal_shutdown("Parámetros inválidos")
            return

        # Esperamos subcampos 'position' y 'orientation'
        position = pose_dict.get("position")
        orientation = pose_dict.get("orientation")

        if not isinstance(position, dict) or not isinstance(orientation, dict):
            rospy.logerr("[move_to_pose_aruco] 'position' u 'orientation' no encontrados o con formato incorrecto.")
            rospy.signal_shutdown("Parámetros inválidos")
            return

        pose_goal = Pose()
        try:
            pose_goal.position.x = float(position["x"])
            pose_goal.position.y = float(position["y"])
            pose_goal.position.z = float(position["z"])
            pose_goal.orientation.x = float(orientation["x"])
            pose_goal.orientation.y = float(orientation["y"])
            pose_goal.orientation.z = float(orientation["z"])
            pose_goal.orientation.w = float(orientation["w"])
        except (KeyError, ValueError, TypeError) as e:
            rospy.logerr("[move_to_pose_aruco] Error al parsear la pose desde parámetros: %s", e)
            rospy.signal_shutdown("Parámetros inválidos")
            return

        rospy.loginfo("[move_to_pose_aruco] Moviendo a pose cartesiana (planificación estándar): %s", pose_goal)

        rospy.sleep(1.0)

        success = self.control.mover_a_pose(pose_goal, wait=True)

        if not success:
            rospy.logwarn("[move_to_pose_aruco] Falló el movimiento a la pose cartesiana objetivo.")

        rospy.signal_shutdown("Movimiento terminado")


def main() -> None:
    MoveToPoseAruco()
    rospy.spin()


if __name__ == "__main__":
    main()
