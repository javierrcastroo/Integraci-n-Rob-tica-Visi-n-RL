#!/usr/bin/env python3
"""Ejecutor de ataques: mueve el robot a la celda validada por ``game_logic_node``.

Versión SIN TF: el ArUco se asume fijo respecto a base_link y se configura por parámetros.
Incluye yaw opcional (rotación alrededor de Z) para alinear tablero y robot.
"""

import json
import math
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import rospy
from geometry_msgs.msg import Pose
from std_msgs.msg import String
from tf.transformations import euler_from_quaternion, quaternion_from_euler

from control_robot import ControlRobot

Cell = Tuple[int, int]


class RobotAttackExecutor:
    def __init__(self) -> None:
        rospy.init_node("robot_attack_executor", anonymous=True)

        # --- Configuración fija del ArUco respecto a base_link (SIN TF) ---
        # Valores iniciales que se rellenan al cargar poseAruco.yaml
        self.aruco_origin_x = 0.0
        self.aruco_origin_y = 0.0
        self.aruco_origin_z = 0.0
        self.aruco_yaw = 0.0

        # Carga inicial del ArUco desde poseAruco.yaml (parámetro Pose_Actual)
        self._load_aruco_pose_from_param()

        # Si tu (0,0) del tablero NO coincide con el centro del ArUco, añade offsets:
        # (por defecto 0.0, 0.0)
        self.board_origin_dx = float(rospy.get_param("~board_origin_dx", 0.0))
        self.board_origin_dy = float(rospy.get_param("~board_origin_dy", 0.0))

        # Tamaño de celda, alturas y obstáculos
        self.board_surface_z = float(rospy.get_param("~board_surface_z", 0.0))
        self.ship_box_size = float(rospy.get_param("~ship_box_size", 0.025))
        self.ammo_box_size = float(rospy.get_param("~ammo_box_size", self.ship_box_size))
        self.move_to_initial = bool(rospy.get_param("~move_to_initial", True))

        self.aruco_obstacle_size_xy = float(rospy.get_param("~aruco_obstacle_size_xy", 0.03))
        self.aruco_obstacle_thickness = float(rospy.get_param("~aruco_obstacle_thickness", 0.002))
        self.aruco_obstacle_z_epsilon = float(rospy.get_param("~aruco_obstacle_z_epsilon", 0.005))

        self.board_obstacle_thickness = float(rospy.get_param("~board_obstacle_thickness", 0.002))
        self.board_obstacle_z_epsilon = float(rospy.get_param("~board_obstacle_z_epsilon", 0.005))
        self.board_obstacle_name = str(rospy.get_param("~board_obstacle_name", "board_plane"))

        self.cell_xy_aruco: Dict[Cell, Tuple[float, float]] = {}

        # -------------------------
        # Pinza y alturas de pick/place
        # -------------------------
        # Anchuras típicas RG2: ajusta a tu pinza real si hace falta
        self.gripper_open_width = float(rospy.get_param("~gripper_open_width", 75))
        self.gripper_closed_width = float(rospy.get_param("~gripper_closed_width", 2))
        self.gripper_force = float(rospy.get_param("~gripper_force", 20.0))

        # Pick munición: bajar "casi hasta el suelo"
        self.pick_clearance_m = float(rospy.get_param("~pick_clearance_m", 0.03))  # 3 cm sobre el suelo

        # Place en tablero: dejar 5 cm sobre el suelo 
        self.place_clearance_m = float(rospy.get_param("~place_clearance_m", 0.05))  # 5 cm sobre el suelo

        # Control
        self.control = ControlRobot(init_ros_node=False)
        self.control.añadir_aruco_como_plano(
            x=self.aruco_origin_x,
            y=self.aruco_origin_y,
            name="aruco_marker",
            size_xy=self.aruco_obstacle_size_xy,
            thickness=self.aruco_obstacle_thickness,
            z_epsilon=self.aruco_obstacle_z_epsilon,
        )

        # Subs/Pubs
        self.attack_result_sub = rospy.Subscriber(
            "battleship/attack_result", String, self.attack_result_cb, queue_size=10
        )
        self.board_layout_sub = rospy.Subscriber(
            "battleship/board_layout", String, self.board_layout_cb, queue_size=10
        )
        #self.board_request_pub = rospy.Publisher(
        #    "battleship/board_request", String, queue_size=10
        #)

        self.ship_boxes: Set[str] = set()
        self.ammo_boxes: Set[str] = set()
        self.cell_xy_base: Dict[Cell, Tuple[float, float]] = {}
        self.ammo_available: List[Tuple[float, float]] = []

        if self.move_to_initial:
            rospy.loginfo("Moviendo a la posición inicial")
            #self._gripper_close()
            self._move_to_initial_position()

        # --- CAMBIO: log coherente con el Z que realmente usas ---
        rospy.loginfo(
            "[robot_attack_executor] SIN TF. ArUco fijo en base_link: (%.3f, %.3f, %.3f), yaw=%.3f rad "
            "(cell=%.3f)",
            self.aruco_origin_x,
            self.aruco_origin_y,
            self.aruco_origin_z,
            self.aruco_yaw,
            self.cell_size,
        )

    # -------------------------
    # Helpers de transformación (tablero -> base_link)
    # -------------------------
    
    def _down_gripper_quat(self) -> Tuple[float, float, float, float]:
        """
        Quaternion para mantener la pinza "mirando hacia abajo" en base_link.
        Convención típica: roll=pi, pitch=0, yaw=0.
        """
        qx, qy, qz, qw = quaternion_from_euler(math.pi, 0.0, 0.0)
        return qx, qy, qz, qw


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

        return self._aruco_to_base_xy(x_local, y_local)

    def _aruco_to_base_xy(self, x_aruco: float, y_aruco: float) -> Tuple[float, float]:
        """Transforma coordenadas en el frame del ArUco al frame base_link."""
        
        x_aruco = -x_aruco
        
        c = math.cos(self.aruco_yaw)
        s = math.sin(self.aruco_yaw)

        # Rotación 2D + traslación
        x_base = self.aruco_origin_x + (c * x_aruco - s * y_aruco)
        y_base = self.aruco_origin_y + (s * x_aruco + c * y_aruco)
        return x_base, y_base

    def _cell_xy_base(self, cell: Cell) -> Tuple[float, float]:
        if cell not in self.cell_xy_base:
            raise RuntimeError(
                f"[robot_attack_executor] No existe cell_xy_base para {cell}. "
                "¿Ha llegado board_layout?"
            )
        return self.cell_xy_base[cell]

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
            quat_aruco_in_robot = [
                float(ori_dict.get("x", 0.0)),
                float(ori_dict.get("y", 0.0)),
                float(ori_dict.get("z", 0.0)),
                float(ori_dict.get("w", 1.0)),
            ]

            _, _, yaw = euler_from_quaternion(quat_aruco_in_robot)

            self.aruco_origin_x = float(pos_dict.get("x", 0.0))
            self.aruco_origin_y = float(pos_dict.get("y", 0.0))
            self.aruco_origin_z = float(pos_dict.get("z", 0.0))
            self.aruco_yaw = yaw
        except Exception as exc:
            rospy.logwarn(
                "[robot_attack_executor] No se pudo cargar Pose_Actual; se mantienen valores previos (error: %s)",
                exc,
            )
            return

        rospy.loginfo(
            "[robot_attack_executor] Pose del ArUco cargada: (x=%.3f, y=%.3f, z=%.3f, yaw=%.3f)",
            self.aruco_origin_x,
            self.aruco_origin_y,
            self.aruco_origin_z,
            self.aruco_yaw,
        )

    def _cell_to_hover_pose(self, cell: Cell) -> Pose:
        x_base, y_base = self._cell_xy_base(cell)

        pose = Pose()
        pose.position.x = x_base
        pose.position.y = y_base
        pose.position.z = self.aruco_origin_z  # o self.hover_z si lo prefieres

        qx, qy, qz, qw = self._down_gripper_quat()
        pose.orientation.x = qx
        pose.orientation.y = qy
        pose.orientation.z = qz
        pose.orientation.w = qw
        return pose

    def _cell_to_box_pose(self, cell: Cell) -> Pose:
        x_base, y_base = self._cell_xy_base(cell)

        pose = Pose()
        pose.position.x = x_base
        pose.position.y = y_base
        pose.position.z = self.board_surface_z + self.ship_box_size / 2.0
        return pose

    def _ammo_xy_to_hover_pose(self, ammo_xy: Tuple[float, float]) -> Pose:
        x_base, y_base = ammo_xy

        pose = Pose()
        pose.position.x = x_base
        pose.position.y = y_base
        pose.position.z = self.aruco_origin_z

        qx, qy, qz, qw = self._down_gripper_quat()
        pose.orientation.x = qx
        pose.orientation.y = qy
        pose.orientation.z = qz
        pose.orientation.w = qw
        return pose

    def _ammo_xy_to_box_pose(self, ammo_xy: Tuple[float, float]) -> Pose:
        x_base, y_base = ammo_xy

        pose = Pose()
        pose.position.x = x_base
        pose.position.y = y_base
        pose.position.z = self.board_surface_z + self.ammo_box_size / 2.0
        return pose

    def _pop_next_ammo_xy(self) -> Optional[Tuple[float, float]]:
        if not self.ammo_available:
            rospy.loginfo("[robot_attack_executor] Sin munición disponible en la cola")
            return None

        return self.ammo_available.pop(0)

    def _move_linear(self, pose: Pose, *, context: str) -> bool:
        success = self.control.mover_en_linea_recta(
            pose,
            wait=True,
            pasos=200,
            z_constante=None,
            eef_step=0.0075,
            intentos=4,
        )

        if success:
            return True

        rospy.logwarn(
            "[robot_attack_executor] No se pudo planificar el movimiento lineal (%s)",
            context,
        )

        # --- CAMBIO RECOMENDADO: fallback a planificación estándar ---
        rospy.logwarn(
            "[robot_attack_executor] Fallback: probando mover_a_pose (%s)",
            context,
        )
        return self.control.mover_a_pose(pose, wait=True)

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

        cell = (col, row)

        # 1) robot -> aruco
        rospy.loginfo(
            "[robot_attack_executor][triangulation] robot->aruco: "
            "(x=%.3f, y=%.3f, yaw=%.3f rad)",
            self.aruco_origin_x,
            self.aruco_origin_y,
            self.aruco_yaw,
        )

        # 2) aruco -> casilla
        if cell not in self.cell_xy_aruco:
            rospy.logerr(
                "[robot_attack_executor][triangulation] No hay xy_aruco para celda %s",
                cell,
            )
            return

        x_aruco, y_aruco = self.cell_xy_aruco[cell]
        rospy.loginfo(
            "[robot_attack_executor][triangulation] aruco->cell: "
            "(x=%.3f, y=%.3f)",
            x_aruco,
            y_aruco,
        )

        # 3) robot -> casilla (TRIANGULACIÓN FINAL)
        try:
            x_base, y_base = self._cell_xy_base(cell)
        except RuntimeError as exc:
            rospy.logerr(str(exc))
            return

        rospy.loginfo(
            "[robot_attack_executor][triangulation] robot->cell: "
            "(x=%.3f, y=%.3f)",
            x_base,
            y_base,
        )

        # -------------------------
        # 1) PICK de munición (si hay)
        # -------------------------
        ammo_xy = self._pop_next_ammo_xy()
        if ammo_xy is not None:
            rospy.loginfo("[robot_attack_executor] Secuencia PICK de munición iniciada.")
            ok = self._pick_ammo_sequence(ammo_xy)  # hover -> bajar casi suelo -> cerrar -> subir
            if not ok:
                rospy.logwarn("[robot_attack_executor] Falló PICK de munición. Abortando ataque.")
                return
        else:
            rospy.logwarn("[robot_attack_executor] No hay munición disponible. Se atacará sin pick/place.")

        # -------------------------
        # 2) PLACE en la casilla (ataque)
        # -------------------------
        rospy.loginfo("[robot_attack_executor] Secuencia PLACE en casilla iniciada.")
        ok = self._place_on_cell_sequence((col, row))  # hover -> bajar (5 cm sobre suelo) -> abrir -> subir
        if not ok:
            rospy.logwarn("[robot_attack_executor] Falló PLACE en casilla. Abortando.")
            return

        # -------------------------
        # 3) Volver a inicial
        # -------------------------
        if self.move_to_initial:
            rospy.loginfo("[robot_attack_executor] Ataque completado, volviendo a Pos_Inicial")
            ok = self._move_to_initial_position()
            if not ok:
                rospy.logwarn("[robot_attack_executor] No se pudo volver a Pos_Inicial tras el ataque.")

                #self.board_request_pub.publish(String("post_robot_attack"))
                #rospy.loginfo("[robot_attack_executor] Petición de captura enviada tras mover el robot")


    def board_layout_cb(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except Exception as exc:
            rospy.logwarn("[robot_attack_executor] Error parseando layout: %s", exc)
            return

        # ==========================
        # DEBUG: imprimir JSON recibido
        # ==========================
        print("\n" + "=" * 25 + " BOARD_LAYOUT RECIBIDO " + "=" * 25)
        print(json.dumps(data, indent=2))
        print("=" * 78 + "\n")

        boards = data.get("boards")
        if not boards:
            return

        layout = boards[0]
        board_corners_aruco = layout.get("board_corners_aruco", [])
        if board_corners_aruco and len(board_corners_aruco) == 4:
            corners_base = []
            for xy in board_corners_aruco:
                try:
                    x_aruco = float(xy[0])
                    y_aruco = float(xy[1])
                except Exception:
                    corners_base = []
                    break
                corners_base.append(self._aruco_to_base_xy(x_aruco, y_aruco))

            if len(corners_base) == 4:
                self.control.añadir_tablero_como_plano(
                    corners_base=corners_base,
                    name=self.board_obstacle_name,
                    thickness=self.board_obstacle_thickness,
                    z_epsilon=self.board_obstacle_z_epsilon,
                )
        else:
            rospy.logwarn("[robot_attack_executor] No llegó board_corners_aruco (o no tiene 4 puntos).")

        self.cell_xy_base = self._build_cell_base_map(
            layout.get("cell_centers_aruco", [])
        )
        self.ammo_available = self._build_ammo_base_list(layout)

        ship_cells = self._extract_cells(layout.get("ship_two_cells", []))
        ship_cells.extend(self._extract_cells(layout.get("ship_one_cells", [])))
        ammo_cells = list(self.ammo_available)

        self._update_obstacles(ship_cells, ammo_cells)

    def _extract_cells(self, cells: Iterable[Sequence[int]]) -> List[Cell]:
        result: List[Cell] = []
        for cell in cells:
            try:
                col, row = cell
                result.append((int(col), int(row)))
            except Exception:
                continue
        return result

    def _build_cell_base_map(
        self, centers_aruco: Iterable[dict]
    ) -> Dict[Cell, Tuple[float, float]]:
        """Convierte los centros enviados por visión (frame ArUco) a base_link."""

        result: Dict[Cell, Tuple[float, float]] = {}
        self.cell_xy_aruco.clear()

        for entry in centers_aruco or []:
            try:
                row = int(entry.get("row"))
                col = int(entry.get("col"))
                xy = entry.get("xy_aruco")
                x_aruco = float(xy[0])
                y_aruco = float(xy[1])
            except Exception:
                continue

            self.cell_xy_aruco[(col, row)] = (x_aruco, y_aruco)

            x_base, y_base = self._aruco_to_base_xy(x_aruco, y_aruco)
            result[(col, row)] = (x_base, y_base)

        return result

    def _build_ammo_base_list(self, layout: dict) -> List[Tuple[float, float]]:
        ammo_points = layout.get("ammo_points_aruco")

        ammo_base: List[Tuple[float, float]] = []
        for entry in ammo_points or []:
            try:
                xy = entry.get("xy_aruco")
                if xy is None:
                    continue
                x_aruco = float(xy[0])
                y_aruco = float(xy[1])
            except Exception:
                continue
            ammo_base.append(self._aruco_to_base_xy(x_aruco, y_aruco))

        return ammo_base


    def _update_obstacles(
        self, ship_cells: List[Cell], ammo_cells: List[Tuple[float, float]]
    ) -> None:
        for name in self.ship_boxes:
            self.control.scene.remove_world_object(name)
        for name in self.ammo_boxes:
            self.control.scene.remove_world_object(name)
        self.ship_boxes.clear()
        self.ammo_boxes.clear()

        for col, row in ship_cells:
            name = f"ship_c{col}_r{row}"
            pose_caja = self._cell_to_box_pose((col, row))
            self.control.añadir_caja_a_escena_de_planificacion(
                pose_caja, name, tamaño=(self.ship_box_size,) * 3
            )
            self.ship_boxes.add(name)

        for idx, ammo_xy in enumerate(ammo_cells):
            name = f"ammo_{idx}"
            pose_caja = self._ammo_xy_to_box_pose(ammo_xy)
            self.control.añadir_caja_a_escena_de_planificacion(
                pose_caja, name, tamaño=(self.ammo_box_size,) * 3
            )
            self.ammo_boxes.add(name)

        rospy.loginfo(
            "[robot_attack_executor] Obstáculos actualizados: %s barcos, %s munición",
            len(self.ship_boxes),
            len(self.ammo_boxes),
        )

    def _move_to_initial_position(self) -> bool:
        """
        Reutiliza exactamente la lógica del initial_position_node:
        lee Pos_Inicial/joints y mueve el robot.
        """
        parametros = rospy.get_param("Pos_Inicial", None)
        joints = None

        if isinstance(parametros, dict):
            joints = parametros.get("joints")

        if not isinstance(joints, list) or len(joints) != 6:
            rospy.logwarn(
                "[robot_attack_executor] No se encontró Pos_Inicial/joints válido, no se puede volver a inicial."
            )
            return False

        rospy.loginfo("[robot_attack_executor] Volviendo a Pos_Inicial: %s", joints)
        ok = self.control.mover_articulaciones(joints, wait=True)

        if not ok:
            rospy.logwarn("[robot_attack_executor] Falló el retorno a Pos_Inicial.")
            return False

        return True

    # -------------------------
    # Helpers de Z y pinza
    # -------------------------

    def _hover_z(self) -> float:
        """Z de seguridad/hover que ya estás usando para moverte por XY."""
        return float(self.aruco_origin_z)

    def _floor_top_z(self) -> float:
        """Z del plano superior del suelo en la escena de MoveIt."""
        return float(self.control.suelo_top_z())

    def _pick_z_near_floor(self) -> float:
        """Z para coger munición: casi suelo (suelo_top + clearance)."""
        return float(self._floor_top_z() + self.pick_clearance_m)

    def _place_z_5cm_over_floor(self) -> float:
        """Z para soltar munición en casilla: 5 cm sobre suelo (o lo que parametrices)."""
        return float(self._floor_top_z() + self.place_clearance_m)

    def _pose_with_z(self, pose: Pose, z: float) -> Pose:
        p = Pose()
        p.position.x = pose.position.x
        p.position.y = pose.position.y
        p.position.z = float(z)
        p.orientation = pose.orientation
        return p

    def _gripper_open(self) -> bool:
        self.control.mover_pinza(self.gripper_open_width, self.gripper_force)
 

    def _gripper_close(self) -> bool:
        self.control.mover_pinza(self.gripper_closed_width, self.gripper_force)

    def _pick_ammo_sequence(self, ammo_xy: Tuple[float, float]) -> bool:
        """
        Secuencia:
          1) ir a munición en hover
          2) abrir la pinza
          3) bajar casi al suelo
          4) cerrar pinza
          5) subir a hover
        """
        hover_pose = self._ammo_xy_to_hover_pose(ammo_xy)
        hover_pose.position.z = self._hover_z()

        rospy.loginfo("[robot_attack_executor] [PICK] Ir a munición (hover)")
        if not self._move_linear(hover_pose, context="ammo_hover"):
            return False
        
        rospy.loginfo("[robot_attack_executor] [PLACE] Abrir pinza")
        self._gripper_open()
        
        rospy.sleep(1)
        
        rospy.loginfo("[robot_attack_executor] Bajando en z")
        pose_actual = self.control.pose_actual()
        pose_actual.position.z -= 0.05
        self.control.mover_trayectoria([pose_actual])

        rospy.loginfo("[robot_attack_executor] [PICK] Cerrar pinza")
        self._gripper_close()
        
        rospy.sleep(1)

        rospy.loginfo("[robot_attack_executor] Subiendo en z")
        pose_actual = self.control.pose_actual()
        pose_actual.position.z += 0.05
        self.control.mover_trayectoria([pose_actual])
        
        rospy.sleep(1)

        return True

    def _place_on_cell_sequence(self, cell: Cell) -> bool:
        """
        Secuencia:
          1) ir a casilla en hover
          2) bajar a z = suelo_top + 5cm
          3) abrir pinza
          4) subir a hover
        """
        hover_pose = self._cell_to_hover_pose(cell)
        hover_pose.position.z = self._hover_z()

        rospy.loginfo("[robot_attack_executor] [PLACE] Ir a casilla (hover)")
        if not self._move_linear(hover_pose, context="cell_hover"):
            return False

        rospy.loginfo("[robot_attack_executor] Bajando en z")
        pose_actual = self.control.pose_actual()
        pose_actual.position.z -= 0.01
        self.control.mover_trayectoria([pose_actual])

        rospy.loginfo("[robot_attack_executor] [PLACE] Abrir pinza")
        self._gripper_open()
        
        rospy.sleep(1)

        rospy.loginfo("[robot_attack_executor] Subiendo en z")
        pose_actual = self.control.pose_actual()
        pose_actual.position.z += 0.01
        self.control.mover_trayectoria([pose_actual])
        
        rospy.sleep(1)

        return True



def main() -> None:
    _executor = RobotAttackExecutor()
    rospy.spin()


if __name__ == "__main__":
    main()
