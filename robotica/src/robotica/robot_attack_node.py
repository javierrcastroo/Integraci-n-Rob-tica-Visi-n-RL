#!/usr/bin/env python3
"""Ejecutor de ataques: mueve el robot a la celda validada por ``game_logic_node``.

Versión SIN TF: el ArUco se asume fijo respecto a base_link y se configura por parámetros.
Incluye yaw opcional (rotación alrededor de Z) para alinear tablero y robot.
"""

import copy
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
        self.aruco_origin_x = 0.0
        self.aruco_origin_y = 0.0
        self.aruco_origin_z = 0.0
        self.aruco_yaw = 0.0

        # Carga inicial del ArUco desde poseAruco.yaml (parámetro Pose_Actual)
        self._load_aruco_pose_from_param()

        # Offsets tablero vs ArUco
        self.board_origin_dx = float(rospy.get_param("~board_origin_dx", 0.0))
        self.board_origin_dy = float(rospy.get_param("~board_origin_dy", 0.0))

        # Tamaños y alturas
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
        self.gripper_open_width = float(rospy.get_param("~gripper_open_width", 75))
        self.gripper_open_width2 = float(rospy.get_param("~gripper_open_width", 30))
        self.gripper_closed_width = float(rospy.get_param("~gripper_closed_width", 2))
        self.gripper_force = float(rospy.get_param("~gripper_force", 20.0))

        self.pick_clearance_m = float(rospy.get_param("~pick_clearance_m", 0.03))   # 3 cm
        self.place_clearance_m = float(rospy.get_param("~place_clearance_m", 0.05)) # 5 cm

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

        self.ship_boxes: Set[str] = set()
        self.ammo_boxes: Set[str] = set()
        self.cell_xy_base: Dict[Cell, Tuple[float, float]] = {}

        # ammo_available ahora guarda (ammo_id, (x_base, y_base))  # <<<
        self.ammo_available: List[Tuple[str, Tuple[float, float]]] = []

        # celdas con barco (y luego también impactos)              # <<<
        self.ship_cells_set: Set[Cell] = set()

        # contador para nombres únicos de impactos                  # <<<
        self.impact_counter: int = 0

        if self.move_to_initial:
            rospy.loginfo("Moviendo a la posición inicial")
            self._move_to_initial_position()

        rospy.loginfo(
            "[robot_attack_executor] SIN TF. ArUco fijo en base_link: (%.3f, %.3f, %.3f), yaw=%.3f rad ",
            self.aruco_origin_x,
            self.aruco_origin_y,
            self.aruco_origin_z,
            self.aruco_yaw,
        )

    # -------------------------
    # Helpers de transformación (tablero -> base_link)
    # -------------------------

    def _down_gripper_quat(self) -> Tuple[float, float, float, float]:
        qx, qy, qz, qw = quaternion_from_euler(math.pi, 0.0, 0.0)
        return qx, qy, qz, qw

    def _rotate_aruco_xy(self, x_aruco: float, y_aruco: float) -> Tuple[float, float]:
        angle_rad = math.radians(135.0)
        c = math.cos(angle_rad)
        s = math.sin(angle_rad)

        x_rot = c * x_aruco - s * y_aruco
        y_rot = s * x_aruco + c * y_aruco
        return x_rot, y_rot

    def _board_to_base_xy(self, x_board: float, y_board: float) -> Tuple[float, float]:
        x_local = x_board + self.board_origin_dx
        y_local = y_board + self.board_origin_dy
        return self._aruco_to_base_xy(x_local, y_local)

    def _aruco_to_base_xy(self, x_aruco: float, y_aruco: float) -> Tuple[float, float]:
        x_aruco = -x_aruco
        c = math.cos(self.aruco_yaw)
        s = math.sin(self.aruco_yaw)

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
        pose.position.z = self.aruco_origin_z

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

    # --- munición: pop + quitar obstáculo asociado -----------------  # <<<

    def _pop_next_ammo_xy(self) -> Optional[Tuple[str, Tuple[float, float]]]:
        """Devuelve (ammo_id, (x_base, y_base)) y elimina el obstáculo de esa munición."""
        if not self.ammo_available:
            rospy.loginfo("[robot_attack_executor] Sin munición disponible en la cola")
            return None

        ammo_id, ammo_xy = self.ammo_available.pop(0)

        box_name = f"ammo_{ammo_id}"
        if box_name in self.ammo_boxes:
            rospy.loginfo(
                "[robot_attack_executor] Eliminando obstáculo de munición usada: %s",
                box_name,
            )
            self.control.scene.remove_world_object(box_name)
            self.ammo_boxes.discard(box_name)

        return ammo_id, ammo_xy

    def _move_linear(self, pose: Pose, *, context: str) -> bool:
        pose_actual = self.control.pose_actual()
        z_segura = pose_actual.position.z

        pose.position.z = z_segura

        rospy.loginfo(
            "[robot_attack_executor] [LINEAR] Contexto=%s, usando Z segura=%.4f",
            context,
            z_segura,
        )

        success = self.control.mover_en_linea_recta(
            pose,
            wait=True,
            pasos=200,
            z_constante=z_segura,
            eef_step=0.0075,
            intentos=4,
        )

        if success:
            return True

        rospy.logwarn(
            "[robot_attack_executor] No se pudo planificar el movimiento lineal (%s)",
            context,
        )

        rospy.logwarn(
            "[robot_attack_executor] Fallback: probando mover_a_pose (%s) con Z segura",
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

        rospy.loginfo(
            "[robot_attack_executor][triangulation] robot->aruco: "
            "(x=%.3f, y=%.3f, yaw=%.3f rad)",
            self.aruco_origin_x,
            self.aruco_origin_y,
            self.aruco_yaw,
        )

        if cell not in self.cell_xy_aruco:
            rospy.logerr(
                "[robot_attack_executor][triangulation] No hay xy_aruco para celda %s",
                cell,
            )
            return

        x_aruco, y_aruco = self.cell_xy_aruco[cell]
        rospy.loginfo(
            "[robot_attack_executor][triangulation] aruco->cell (ya rotado 45º): "
            "(x=%.3f, y=%.3f)",
            x_aruco,
            y_aruco,
        )

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
        ammo_info = self._pop_next_ammo_xy()
        if ammo_info is not None:
            ammo_id, ammo_xy = ammo_info
            rospy.loginfo(
                "[robot_attack_executor] Secuencia PICK de munición iniciada (id=%s).",
                ammo_id,
            )
            ok = self._pick_ammo_sequence(ammo_xy)
            if not ok:
                rospy.logwarn("[robot_attack_executor] Falló PICK de munición. Abortando ataque.")
                return
        else:
            rospy.logwarn("[robot_attack_executor] No hay munición disponible. Se atacará sin pick/place.")

        # -------------------------
        # 2) PLACE en la casilla (ataque)
        # -------------------------
        rospy.loginfo("[robot_attack_executor] Secuencia PLACE en casilla iniciada.")
        ok = self._place_on_cell_sequence(cell)
        if not ok:
            rospy.logwarn("[robot_attack_executor] Falló PLACE en casilla. Abortando.")
            return

        # -------------------------
        # 2.5) Añadir bloque de impacto en la casilla atacada          # <<<
        # -------------------------
        self._add_impact_block(cell)

        # -------------------------
        # 3) Volver a inicial
        # -------------------------
        if self.move_to_initial:
            rospy.loginfo("[robot_attack_executor] Ataque completado, volviendo a Pos_Inicial")
            ok = self._move_to_initial_position()
            if not ok:
                rospy.logwarn("[robot_attack_executor] No se pudo volver a Pos_Inicial tras el ataque.")

    def board_layout_cb(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except Exception as exc:
            rospy.logwarn("[robot_attack_executor] Error parseando layout: %s", exc)
            return

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

                x_rot, y_rot = self._rotate_aruco_xy(x_aruco, y_aruco)
                corners_base.append(self._aruco_to_base_xy(x_rot, y_rot))

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
        self.ammo_available = self._build_ammo_base_list(layout)   # <<<

        ship_cells = self._extract_cells(layout.get("ship_two_cells", []))
        ship_cells.extend(self._extract_cells(layout.get("ship_one_cells", [])))

        # celdas con barco iniciales
        self.ship_cells_set = set(ship_cells)
        rospy.loginfo("[robot_attack_executor] ship_cells_set actualizado con %d celdas", len(self.ship_cells_set))

        # Obstáculos de barcos + munición
        self._update_obstacles(ship_cells, self.ammo_available)    # <<<

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

            x_rot, y_rot = self._rotate_aruco_xy(x_aruco, y_aruco)
            self.cell_xy_aruco[(col, row)] = (x_rot, y_rot)

            x_base, y_base = self._aruco_to_base_xy(x_rot, y_rot)
            result[(col, row)] = (x_base, y_base)

        return result

    # --- munición: construir lista (id, (x_base, y_base)) -----------  # <<<

    def _build_ammo_base_list(self, layout: dict) -> List[Tuple[str, Tuple[float, float]]]:
        """
        Devuelve lista de (ammo_id, (x_base, y_base)) usando los ids del JSON si existen.
        Espera en layout["ammo_points_aruco"] entradas tipo:
          { "id": <algo>, "xy_aruco": [x, y] }
        Si no hay id, usa el índice.
        """
        ammo_points = layout.get("ammo_points_aruco")

        ammo_base: List[Tuple[str, Tuple[float, float]]] = []
        for idx, entry in enumerate(ammo_points or []):
            try:
                if isinstance(entry, dict):
                    ammo_id = str(entry.get("id", idx))
                    xy = entry.get("xy_aruco")
                else:
                    ammo_id = str(idx)
                    xy = entry
                if xy is None:
                    continue

                x_aruco = float(xy[0])
                y_aruco = float(xy[1])
            except Exception:
                continue

            x_rot, y_rot = self._rotate_aruco_xy(x_aruco, y_aruco)
            x_base, y_base = self._aruco_to_base_xy(x_rot, y_rot)
            ammo_base.append((ammo_id, (x_base, y_base)))

        return ammo_base

    def _update_obstacles(
        self,
        ship_cells: List[Cell],
        ammo_cells: List[Tuple[str, Tuple[float, float]]],  # (ammo_id, (x_base, y_base))  # <<<
    ) -> None:
        for name in self.ship_boxes:
            self.control.scene.remove_world_object(name)
        for name in self.ammo_boxes:
            self.control.scene.remove_world_object(name)
        self.ship_boxes.clear()
        self.ammo_boxes.clear()

        # barcos
        for col, row in ship_cells:
            name = f"ship_c{col}_r{row}"
            pose_caja = self._cell_to_box_pose((col, row))
            self.control.añadir_caja_a_escena_de_planificacion(
                pose_caja, name, tamaño=(self.ship_box_size,) * 3
            )
            self.ship_boxes.add(name)

        # munición: usar ammo_id para el nombre del obstáculo
        for ammo_id, ammo_xy in ammo_cells:
            name = f"ammo_{ammo_id}"
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
        return float(self.aruco_origin_z)

    def _floor_top_z(self) -> float:
        return float(self.control.suelo_top_z())

    def _pick_z_near_floor(self) -> float:
        return float(self._floor_top_z() + self.pick_clearance_m)

    def _place_z_5cm_over_floor(self) -> float:
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
        
    def _gripper_open2(self) -> bool:
        self.control.mover_pinza(self.gripper_open_width2, self.gripper_force)

    def _gripper_close(self) -> bool:
        self.control.mover_pinza(self.gripper_closed_width, self.gripper_force)

    def _pick_ammo_sequence(self, ammo_xy: Tuple[float, float]) -> bool:
        hover_pose = self._ammo_xy_to_hover_pose(ammo_xy)
        hover_pose.position.z = self._hover_z()

        rospy.loginfo("[robot_attack_executor] [PICK] Ir a munición (hover)")
        if not self._move_linear(hover_pose, context="ammo_hover"):
            return False

        rospy.loginfo("[robot_attack_executor] [PICK] Abrir pinza")
        self._gripper_open()
        rospy.sleep(1)

        rospy.loginfo("[robot_attack_executor] [PICK] Bajando en z")
        pose_actual = self.control.pose_actual()
        pose_actual.position.z -= 0.06
        self.control.mover_trayectoria([pose_actual])

        rospy.loginfo("[robot_attack_executor] [PICK] Cerrar pinza")
        self._gripper_close()
        rospy.sleep(1)

        rospy.loginfo("[robot_attack_executor] [PICK] Subiendo en z")
        pose_actual = self.control.pose_actual()
        pose_actual.position.z += 0.10
        self.control.mover_trayectoria([pose_actual])
        rospy.sleep(1)

        return True

    def _place_on_cell_sequence(self, cell: Cell) -> bool:
        hover_pose = self._cell_to_hover_pose(cell)
        hover_pose.position.z = self._hover_z()

        rospy.loginfo("[robot_attack_executor] [PLACE] Ir a casilla (hover)")
        if not self._move_linear(hover_pose, context="cell_hover"):
            return False

        rospy.loginfo("[robot_attack_executor] Bajando en z")
        pose_actual = self.control.pose_actual()

        # Decidir profundidad según si la casilla tiene barco o no
        hay_barco = cell in self.ship_cells_set
        pose_actual = self.control.pose_actual()

        if hay_barco:
            delta_z = -0.075  # bajar 7.5 cm
            rospy.loginfo(
                "[robot_attack_executor] [PLACE] Celda %s con BARCO: bajando 7.5 cm",
                cell,
            )
        else:
            delta_z = -0.10   # bajar 10 cm
            rospy.loginfo(
                "[robot_attack_executor] [PLACE] Celda %s de AGUA: bajando 10 cm",
                cell,
            )

            # Mirar celdas adyacentes en columnas (misma fila)
            col, row = cell
            vecinos = [(col - 1, row), (col + 1, row)]
            hay_barco_adyacente = any(v in self.ship_cells_set for v in vecinos)

            rotar_pinza = False
            if hay_barco_adyacente:
                rotar_pinza = True
                rospy.loginfo(
                    "[robot_attack_executor] [PLACE] Agua adyacente a BARCO en %s: "
                    "rotando pinza 90 grados",
                    cell,
                )

            if rotar_pinza:
                q_old = [
                    pose_actual.orientation.x,
                    pose_actual.orientation.y,
                    pose_actual.orientation.z,
                    pose_actual.orientation.w,
                ]
                roll, pitch, yaw = euler_from_quaternion(q_old)
                yaw += math.pi / 2.0  # +90 grados

                q_new = quaternion_from_euler(roll, pitch, yaw)
                pose_actual.orientation.x = q_new[0]
                pose_actual.orientation.y = q_new[1]
                pose_actual.orientation.z = q_new[2]
                pose_actual.orientation.w = q_new[3]

                rospy.loginfo(
                    "[robot_attack_executor] [PLACE] Pinza rotada 90 grados en yaw (celda %s)",
                    cell,
                )

        pose_actual.position.z += delta_z
        ok = self.control.mover_trayectoria([pose_actual])

        rospy.loginfo("[robot_attack_executor] [PLACE] Abrir pinza poco")
        self._gripper_open2()
        rospy.sleep(1)

        rospy.loginfo("[robot_attack_executor] Subiendo en z")
        pose_actual = self.control.pose_actual()
        pose_actual.position.z += 0.07
        self.control.mover_trayectoria([pose_actual])
        rospy.sleep(1)
        
        rospy.loginfo("[robot_attack_executor] [PLACE] Abrir pinza mucho")
        self._gripper_open()
        rospy.sleep(1)


        return True

    # --- bloque de impacto en casilla atacada ----------------------  # <<<

    def _add_impact_block(self, cell: Cell) -> None:
        """
        Añade un bloque nuevo en la casilla atacada:
          - si ya había obstáculo en esa casilla: centro del bloque a 2.6 cm del suelo
          - si no, bloque normal apoyado en el tablero
        Además, añade la celda a ship_cells_set.
        """
        col, row = cell
        ya_habia_obstaculo = cell in self.ship_cells_set

        x_base, y_base = self._cell_xy_base(cell)
        pose_caja = Pose()
        pose_caja.position.x = x_base
        pose_caja.position.y = y_base

        if ya_habia_obstaculo:
            # Bloque a 2.6 cm del suelo (suelo_top_z + 0.026 + mitad de la altura del bloque)
            z_suelo_top = self._floor_top_z()
            pose_caja.position.z = z_suelo_top + 0.026 + self.ammo_box_size / 2.0
            rospy.loginfo(
                "[robot_attack_executor] [IMPACT] Celda %s ya tenía obstáculo; "
                "nuevo bloque centrado a 2.6 cm del suelo",
                cell,
            )
        else:
            # Primer bloque en esa celda: apoyado en el tablero
            pose_caja.position.z = self.board_surface_z + self.ammo_box_size / 2.0
            rospy.loginfo(
                "[robot_attack_executor] [IMPACT] Primer bloque en celda %s, apoyado en el tablero",
                cell,
            )

        # Orientación: misma que mirar hacia abajo
        qx, qy, qz, qw = self._down_gripper_quat()
        pose_caja.orientation.x = qx
        pose_caja.orientation.y = qy
        pose_caja.orientation.z = qz
        pose_caja.orientation.w = qw

        # Nombre único para el bloque de impacto
        self.impact_counter += 1
        name = f"impact_c{col}_r{row}_{self.impact_counter}"

        self.control.añadir_caja_a_escena_de_planificacion(
            pose_caja, name, tamaño=(self.ammo_box_size,) * 3
        )
        self.ship_boxes.add(name)

        # Añadimos la celda a ship_cells_set para que cuente como ocupada
        self.ship_cells_set.add(cell)

        rospy.loginfo(
            "[robot_attack_executor] [IMPACT] Añadido bloque de impacto '%s' en celda %s",
            name,
            cell,
        )


def main() -> None:
    _executor = RobotAttackExecutor()
    rospy.spin()


if __name__ == "__main__":
    main()
