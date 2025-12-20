#!/usr/bin/python3

import sys
import copy
from typing import List, Optional

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
        self.floor_name = "suelo"
        self.floor_size = (2.0, 2.0, 0.05)   # (x, y, z)
        self.floor_center_z = -0.026
        self.añadir_suelo()

        # Parámetros para hacer la planificación más robusta
        self.move_group.allow_replanning(True)
        self.move_group.set_num_planning_attempts(10)
        self.move_group.set_planning_time(10.0)
        self.move_group.set_max_acceleration_scaling_factor(0.6)
        self.move_group.set_max_velocity_scaling_factor(0.8)

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
        if (
            pose_caja.orientation.x == 0.0
            and pose_caja.orientation.y == 0.0
            and pose_caja.orientation.z == 0.0
            and pose_caja.orientation.w == 0.0
        ):
            pose_caja.orientation.w = 1.0
        box_pose.pose = pose_caja
        box_name = name
        self.scene.add_box(box_name, box_pose, size=tamaño)

    def mover_trayectoria(
        self,
        poses: List[Pose],
        wait: bool = True,
        pasos: int = 100,
        z_constante: Optional[float] = None,
        eef_step: float = 0.01,
        intentos: int = 3,
    ) -> bool:
        if not poses:
            return True

        trayecto_expandido: List[Pose] = []
        inicio_trayectoria = copy.deepcopy(self.pose_actual())
        pose_previa = copy.deepcopy(inicio_trayectoria)

        if z_constante is not None:
            pose_previa.position.z = z_constante
            inicio_trayectoria.position.z = z_constante

        for pose_objetivo in poses:
            if z_constante is not None:
                pose_objetivo = copy.deepcopy(pose_objetivo)
                pose_objetivo.position.z = z_constante

            trayecto_expandido.extend(
                self._generar_puntos_intermedios(
                    pose_previa, pose_objetivo, pasos=pasos
                )
            )
            pose_previa = pose_objetivo

        trayecto_expandido.insert(0, inicio_trayectoria)

        for intento in range(intentos):
            self.move_group.set_start_state_to_current_state()
            paso_ef = eef_step * (0.5 ** intento)
            (plan, fraction) = self.move_group.compute_cartesian_path(
                trayecto_expandido, paso_ef, True
            )

            if fraction == 1.0:
                return self.move_group.execute(plan, wait=wait)

            rospy.logwarn(
                "[ControlRobot] compute_cartesian_path incompleto (fraction=%.3f, eef_step=%.4f); reintentando",
                fraction,
                paso_ef,
            )

        rospy.logerr(
            "[ControlRobot] No se pudo planificar trayecto cartesiano tras %s intentos",
            intentos,
        )
        return False

    def mover_en_linea_recta(
        self,
        pose_objetivo: Pose,
        wait: bool = True,
        pasos: int = 100,
        z_constante: Optional[float] = None,
        eef_step: float = 0.01,
        intentos: int = 3,
    ) -> bool:
        """
        Mueve el efector final en línea recta hasta ``pose_objetivo`` interpolando ``pasos`` puntos.

        Resulta útil para trayectorias más rectilíneas o cuando la planificación estándar falla.
        """

        return self.mover_trayectoria(
            [pose_objetivo],
            wait=wait,
            pasos=pasos,
            z_constante=z_constante,
            eef_step=eef_step,
            intentos=intentos,
        )

    def añadir_suelo(self) -> None:
        pose_suelo = Pose()
        pose_suelo.position.z = self.floor_center_z
        pose_suelo.orientation.w = 1.0
        self.añadir_caja_a_escena_de_planificacion(pose_suelo, self.floor_name, self.floor_size)

    def suelo_top_z(self) -> float:
        return float(self.floor_center_z + self.floor_size[2] / 2.0)

    def añadir_aruco_como_plano(self,*,x: float,y: float,name: str = "aruco_marker",size_xy: float = 0.03, thickness: float = 0.002,z_epsilon: float = 0.005) -> None:
        """
        Añade un obstáculo representando el ArUco como una caja muy delgada (plano).
        Se coloca sobre el suelo (ignorando z del ArUco).
        """

        # Si ya existe, lo eliminamos y recreamos (evita duplicados)
        try:
            self.scene.remove_world_object(name)
        except Exception:
            pass

        pose = Pose()
        pose.position.x = float(x)
        pose.position.y = float(y)

        z_top_suelo = self.suelo_top_z()
        pose.position.z = z_top_suelo + float(z_epsilon) + float(thickness) / 2.0

        pose.orientation.w = 1.0

        self.añadir_caja_a_escena_de_planificacion(
            pose,
            name,
            tamaño=(float(size_xy), float(size_xy), float(thickness)),
        )



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

    pi_medios = pi / 2

    # 1) Mover el robot a articulaciones iniciales
    control.mover_articulaciones([0, -pi_medios, -pi_medios, -pi_medios, pi_medios, 0])

    rospy.sleep(1.0)

    # 2) Definir pose objetivo ABSOLUTA en base_link
    pose_objetivo = Pose()
    pose_objetivo.position.x = -0.299
    pose_objetivo.position.y = -0.244
    pose_objetivo.position.z = 0.284

    # Orientación válida (identidad)
    pose_objetivo.orientation.w = 1.0

    rospy.loginfo(
        "[TEST] Moviendo a pose objetivo (x=%.3f, y=%.3f, z=%.3f)",
        pose_objetivo.position.x,
        pose_objetivo.position.y,
        pose_objetivo.position.z,
    )

    # 3) Movimiento directo a la pose (NO cartesiano)
    ok = control.mover_a_pose(pose_objetivo, wait=True)

    if ok:
        rospy.loginfo("[TEST] Movimiento completado correctamente")
    else:
        rospy.logerr("[TEST] No se pudo planificar el movimiento a la pose objetivo")
