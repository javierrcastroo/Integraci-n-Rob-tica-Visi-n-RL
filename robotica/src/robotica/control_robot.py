#!/usr/bin/python3

import sys
import copy
from typing import List

import numpy as np
import rospy
from actionlib import SimpleActionClient
from control_msgs.msg import GripperCommandAction, GripperCommandGoal, GripperCommandResult
from geometry_msgs.msg import Pose, PoseStamped
import moveit_msgs.msg
from moveit_commander import MoveGroupCommander, RobotCommander, PlanningSceneInterface, roscpp_initialize
from moveit_commander.conversions import pose_to_list
from math import pi, tau, dist, fabs, cos
from std_msgs.msg import String

class ControlRobot:
    def __init__(self, *, init_ros_node: bool = True, node_name: str = "control_robot") -> None:
        """Inicializa el controlador del robot.

        Args:
            init_ros_node: Si es True y ROS no está inicializado, crea el nodo con
                ``node_name``. Esto permite reutilizar la clase desde otros nodos
                que ya hayan llamado a ``rospy.init_node``.
            node_name: Nombre del nodo ROS en caso de inicializarlo aquí.
        """

        roscpp_initialize(sys.argv)
        if init_ros_node and not rospy.core.is_initialized():
            rospy.init_node(node_name, anonymous=True)
        self.robot = RobotCommander()
        self.scene = PlanningSceneInterface()
        self.group_name = "robot"
        self.move_group = MoveGroupCommander(self.group_name)
        self.gripper_action_client = SimpleActionClient("rg2_action_server", GripperCommandAction)
        self.añadir_suelo()

    def articulaciones_actuales(self) -> list:
        return self.move_group.get_current_joint_values()
    
    def mover_articulaciones(self, joint_goal: List[float], wait: bool= True) -> bool:
        return self.move_group.go(joint_goal, wait=wait)
    
    def pose_actual(self) -> Pose:
        return self.move_group.get_current_pose().pose
    
    def pose_a_stamped(self, pose: Pose) -> PoseStamped:
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = "base_link"
        pose_stamped.pose = pose
        return pose_stamped
    
    def mover_a_pose(self, pose_goal: Pose, wait: bool = True) -> bool:
        """
        Mueve el robot a una pose objetivo usando la planificación estándar de MoveIt
        (no trayectoria cartesiana).

        Devuelve True si la ejecución ha sido correcta.
        """
        # Aseguramos que el estado inicial es el actual
        self.move_group.set_start_state_to_current_state()
        self.move_group.set_pose_target(pose_goal)

        plan = self.move_group.plan()

        # plan puede ser una tupla o un objeto, según versión; comprobamos que tenga puntos
        try:
            trajectory = plan[1] if isinstance(plan, tuple) else plan
        except Exception:
            trajectory = plan

        if not trajectory or not hasattr(trajectory, "joint_trajectory") \
        or not trajectory.joint_trajectory.points:
            rospy.logerr("[ControlRobot] No se ha podido planificar una trayectoria a la pose objetivo.")
            self.move_group.clear_pose_targets()
            return False

        success = self.move_group.execute(trajectory, wait=wait)
        self.move_group.stop()
        self.move_group.clear_pose_targets()

        return bool(success)
    
    def añadir_caja_a_escena_de_planificacion(self, pose_caja: Pose, name: str,
                                  tamaño: tuple = (.1,.1,.1)) -> None:
        box_pose = PoseStamped()
        box_pose.header.frame_id = "base_link"
        box_pose.pose = pose_caja
        box_name = name
        self.scene.add_box(box_name, box_pose, size=tamaño)

    def mover_trayectoria(self, poses: List[Pose], wait: bool = True) -> bool:
        if not poses:
            return True

        trayecto_expandido: List[Pose] = []
        pose_previa = self.pose_actual()

        for pose_objetivo in poses:
            trayecto_expandido.extend(
                self._generar_puntos_intermedios(pose_previa, pose_objetivo)
            )
            pose_previa = pose_objetivo

        trayecto_expandido.insert(0, self.pose_actual())

        (plan, fraction) = self.move_group.compute_cartesian_path(trayecto_expandido, 0.01)

        if fraction != 1.0:
            return False

        return self.move_group.execute(plan, wait=wait)

    def añadir_suelo(self) -> None:
        pose_suelo = Pose()
        pose_suelo.position.z = -0.026
        self.añadir_caja_a_escena_de_planificacion(pose_suelo,"suelo",(2,2,.05))

    def _generar_puntos_intermedios(self, inicio: Pose, fin: Pose, pasos: int = 100) -> List[Pose]:
        """Genera ``pasos`` poses entre ``inicio`` y ``fin`` usando numpy."""

        inicio_pos = np.array([inicio.position.x, inicio.position.y, inicio.position.z], dtype=float)
        fin_pos = np.array([fin.position.x, fin.position.y, fin.position.z], dtype=float)
        desplazamiento = fin_pos - inicio_pos
        distancia = np.linalg.norm(desplazamiento)

        if distancia == 0:
            return [copy.deepcopy(fin)]

        fracciones = np.linspace(0.0, 1.0, pasos + 2)[1:]
        poses_intermedias: List[Pose] = []

        for fraccion in fracciones:
            pose_intermedia = copy.deepcopy(inicio)
            punto = inicio_pos + desplazamiento * fraccion
            pose_intermedia.position.x, pose_intermedia.position.y, pose_intermedia.position.z = punto
            pose_intermedia.orientation = fin.orientation
            poses_intermedias.append(pose_intermedia)

        return poses_intermedias
        
    def mover_pinza(self, anchura_dedos: float, fuerza: float) -> bool:
        goal = GripperCommandGoal()
        goal.command.position = anchura_dedos
        goal.command.max_effort = fuerza
        self.gripper_action_client.send_goal(goal)
        self.gripper_action_client.wait_for_result()
        result = self.gripper_action_client.get_result()
        
        return result.reached_goal

if __name__ == '__main__':
    # Crear el objeto de tipo robot
    control = ControlRobot()
    
    pi_medios = pi/2
    # Mover el robot a articulaciones iniciales
    control.mover_articulaciones([0,-pi_medios,-pi_medios,-pi_medios,pi_medios,0])
    
    # Mover el robot a una pose
    pose_actual = control.pose_actual()
    pose_actual.position.z -= 0.1
    control.mover_a_pose(pose_actual)
    
    # Mover el efector final del robot en línea recta a través de varias poses
    poses = [] # Lista de poses que va a recorrer
    
    # Pose 1
    pose_actual = control.pose_actual()
    pose_actual.position.z += 0.1
    poses.append(copy.deepcopy(pose_actual))
    
    # Pose 2
    pose_actual.position.y += 0.1
    poses.append(copy.deepcopy(pose_actual))
    
    # Pose 3
    pose_actual.position.x += 0.1
    poses.append(copy.deepcopy(pose_actual))
    
    # Pose 4
    pose_actual.position.x -= 0.1
    pose_actual.position.y -= 0.1
    pose_actual.position.z -= 0.1
    poses.append(copy.deepcopy(pose_actual))
    
    control.mover_trayectoria(poses)
