#!/usr/bin/env python3
import os
import sys
import json
import rospy
from std_msgs.msg import String, Empty

# --- AÑADIDO: asegurar que vemos battleship_logic.py en esta carpeta ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from battleship_logic import evaluate_board


from battleship_logic import evaluate_board


def _cells_from_layout(layout):
    """
    Normaliza ship_two_cells y ship_one_cells a listas de tuplas (r, c).
    """
    ship_two_cells = [tuple(c) for c in layout.get("ship_two_cells", [])]
    ship_one_cells = [tuple(c) for c in layout.get("ship_one_cells", [])]
    layout["ship_two_cells"] = ship_two_cells
    layout["ship_one_cells"] = ship_one_cells
    return ship_two_cells, ship_one_cells


def _gesture_to_digit(label):
    """
    Extrae el primer dígito que aparezca en la etiqueta, por ejemplo:
    '0dedos' -> 0, '4dedos' -> 4.
    """
    for ch in label:
        if ch.isdigit():
            return int(ch)
    raise ValueError(f"No se encontró dígito en la etiqueta de gesto: '{label}'")


def _cell_name(row, col):
    # A, B, C,... para filas; 1,2,3... para columnas (1-based)
    return f"{chr(ord('A') + row)}{col + 1}"


class GameLogicNode(object):
    def __init__(self):
        # estado del tablero
        self.current_layout = None
        self.board_valid = False
        self.ship_two_cells = set()
        self.ship_one_cells = set()
        self.all_ship_cells = set()
        self.hits = set()
        self.max_row = None
        self.max_col = None

        # subs & pubs
        self.board_sub = rospy.Subscriber(
            "battleship/board_layout",
            String,
            self.board_cb,
            queue_size=10,
        )
        self.attack_sub = rospy.Subscriber(
            "battleship/attack",
            String,
            self.attack_cb,
            queue_size=10,
        )
        self.result_pub = rospy.Publisher(
            "battleship/attack_result",
            String,
            queue_size=10,
        )
        self.board_request_pub = rospy.Publisher(
            "battleship/board_request",
            String,
            queue_size=10,
        )

        self.rl_turn_pub = rospy.Publisher(
            "/game/your_turn", Empty, queue_size=10
        )
        self.rl_feedback_pub = rospy.Publisher(
            "/game/feedback", String, queue_size=10
        )
        self.rl_state_pub = rospy.Publisher(
            "/game/state", String, queue_size=10
        )

        # Escuchar disparos del agente RL
        self.rl_attack_sub = rospy.Subscriber(
            "/agent/fire_coordinates",
            String,
            self.rl_attack_cb,
            queue_size=10,
        )

        rospy.loginfo("[game_logic_node] Iniciado. Esperando tablero y ataques...")

    def notify_rl_turn(self, player):
        """Asigna el turno al agente RL cuando el jugador 1 ha actuado."""
        if player != "P1":
            return

        if not self.board_valid or self.current_layout is None:
            return

        if self.all_ship_cells.issubset(self.hits):
            return

        rospy.loginfo("[game_logic_node] Turno para el Player 2 (RL)")
        self.rl_turn_pub.publish(Empty())

    # ---------- callback tablero ----------
    def board_cb(self, msg):
        try:
            data = json.loads(msg.data)
        except Exception as e:
            rospy.logwarn(f"[game_logic_node] Error parseando layout de tablero: {e}")
            return

        boards = data.get("boards", [])
        if not boards:
            rospy.logwarn("[game_logic_node] Mensaje de tablero sin 'boards'")
            return

        # de momento usamos solo el primer tablero (T1)
        layout = boards[0]

        # normalizar celdas
        ship_two_cells, ship_one_cells = _cells_from_layout(layout)

        # evaluar con lógica existente
        ok, msg_text = evaluate_board(layout)
        rospy.loginfo(f"[game_logic_node] Evaluación tablero: ok={ok} msg='{msg_text}'")

        self.current_layout = layout
        self.board_valid = ok
        self.ship_two_cells = set(ship_two_cells)
        self.ship_one_cells = set(ship_one_cells)
        self.all_ship_cells = self.ship_two_cells | self.ship_one_cells

        if self.all_ship_cells:
            self.max_row = max(r for (r, c) in self.all_ship_cells)
            self.max_col = max(c for (r, c) in self.all_ship_cells)
        else:
            self.max_row = self.max_col = None

        # reseteamos impactos si ha cambiado el tablero
        self.hits = set()

    # ---------- callback ataque ----------
    def attack_cb(self, msg):
        try:
            data = json.loads(msg.data)
        except Exception as e:
            rospy.logwarn(f"[game_logic_node] Error parseando ataque: {e}")
            return

        

        gestures = data.get("gestures", [])
        player = data.get("player", "P1")

        if len(gestures) != 2:
            self.publish_result(
                status="ERROR",
                result="invalid_attack",
                cell=None,
                message="Se esperaban exactamente 2 gestos (fila, columna)",
            )
            return

        if not self.board_valid or self.current_layout is None:
            self.publish_result(
                status="OK",
                result="board_invalid",
                cell=None,
                message="El tablero no es válido o no está configurado",
            )
            return

        try:
            row_idx = _gesture_to_digit(gestures[0])
            col_idx = _gesture_to_digit(gestures[1])
        except ValueError as e:
            self.publish_result(
                status="ERROR",
                result="invalid_gestures",
                cell=None,
                message=str(e),
            )
            return

        # comprobamos que está dentro del tablero detectado
        if self.max_row is not None and self.max_col is not None:
            if row_idx < 0 or row_idx > self.max_row or col_idx < 0 or col_idx > self.max_col:
                self.publish_result(
                    status="OK",
                    result="out_of_bounds",
                    cell={
                        "row": row_idx,
                        "col": col_idx,
                        "name": _cell_name(row_idx, col_idx),
                    },
                    message="Ataque fuera del tablero detectado",
                )
                self.notify_rl_turn(player)
                return

        cell = (row_idx, col_idx)
        cell_name = _cell_name(row_idx, col_idx)

        # ataque repetido
        if cell in self.hits:
            self.publish_result(
                status="OK",
                result="repeated",
                cell={
                    "row": row_idx,
                    "col": col_idx,
                    "name": cell_name,
                },
                message=f"Ataque repetido en {cell_name}",
            )
            self.notify_rl_turn(player)
            return

        # registramos impacto
        self.hits.add(cell)

        # determinar agua / tocado / hundido
        if cell not in self.all_ship_cells:
            # agua
            self.publish_result(
                status="OK",
                result="miss",
                cell={
                    "row": row_idx,
                    "col": col_idx,
                    "name": cell_name,
                },
                message=f"Agua en {cell_name}",
            )
            self.notify_rl_turn(player)
            return

        # impacto en algún barco
        # barco de 2
        result = "hit"
        message = f"Tocado en {cell_name}"

        # ¿barco de 2 hundido?
        if self.ship_two_cells and cell in self.ship_two_cells:
            if self.ship_two_cells.issubset(self.hits):
                result = "sunk"
                message = f"Hundido barco de 2 en {cell_name}"

        # ¿barco de 1? (cada celda individual)
        if cell in self.ship_one_cells:
            result = "sunk"
            message = f"Hundido barco de 1 en {cell_name}"

        # ¿todos hundidos?
        if self.all_ship_cells.issubset(self.hits):
            result = "sunk_all"
            message = f"¡Todos los barcos hundidos! Último impacto en {cell_name}"

        self.publish_result(
            status="OK",
            result=result,
            cell={
                "row": row_idx,
                "col": col_idx,
                "name": cell_name,
            },
            message=message,
        )

        self.notify_rl_turn(player)

    def rl_attack_cb(self, msg):
        """
        Ataque del agente RL.
        msg.data es una coordenada tipo "A3" (fila-letra, columna-número)
        """
        if not self.board_valid or self.current_layout is None:
            rospy.logwarn("[game_logic_node] RL ha atacado pero el tablero no es válido")
            return
    
        coord = msg.data.strip().upper()
        if len(coord) < 2:
            rospy.logwarn(f"[game_logic_node] Coordenada RL inválida: '{coord}'")
            return
    
        try:
            # 'A3' -> row_idx=0, col_idx=2
            row_idx = ord(coord[0]) - ord('A')
            col_idx = int(coord[1:]) - 1
        except Exception as e:
            rospy.logwarn(f"[game_logic_node] Error parseando coord RL '{coord}': {e}")
            return
    
        # Comprobamos límites de tablero detectado
        if self.max_row is not None and self.max_col is not None:
            if row_idx < 0 or row_idx > self.max_row or col_idx < 0 or col_idx > self.max_col:
                rospy.loginfo(
                    f"[game_logic_node] Ataque RL fuera de tablero: {coord} "
                    f"(row={row_idx}, col={col_idx})"
                )
                # Simplemente ignoramos.
                return
    
        cell = (row_idx, col_idx)
        cell_name = _cell_name(row_idx, col_idx)
    
        # Ataque repetido
        if cell in self.hits:
            rospy.loginfo(f"[game_logic_node] Ataque RL repetido en {cell_name}")
            self.rl_feedback_pub.publish(String("repetido"))
            return
    
        # Registramos impacto
        self.hits.add(cell)
    
        # Agua vs impacto
        if cell not in self.all_ship_cells:
            # Agua
            rospy.loginfo(f"[game_logic_node] RL: Agua en {cell_name}")
            self.rl_feedback_pub.publish(String("agua"))
            return
    
        # Impacto
        result = "hit"
        message = f"Tocado en {cell_name}"
    
        # ¿barco de 2 hundido?
        if self.ship_two_cells and cell in self.ship_two_cells:
            if self.ship_two_cells.issubset(self.hits):
                result = "sunk"
                message = f"Hundido barco de 2 en {cell_name}"
    
        # ¿barco de 1 hundido?
        if cell in self.ship_one_cells:
            result = "sunk"
            message = f"Hundido barco de 1 en {cell_name}"
    
        # ¿todos hundidos?
        if self.all_ship_cells.issubset(self.hits):
            result = "sunk_all"
            message = f"¡Todos los barcos hundidos! Último impacto en {cell_name}"
    
        rospy.loginfo(f"[game_logic_node] RL: {message}")
    
        # Traducir RESULT → feedback RL
        if result == "hit":
            self.rl_feedback_pub.publish(String("tocado"))
    
        elif result == "sunk":
            self.rl_feedback_pub.publish(String("hundido"))
    
        elif result == "sunk_all":
            # RL gana la partida
            self.rl_feedback_pub.publish(String("victoria"))
            self.rl_state_pub.publish(String("win_agent"))


    # ---------- publicación resultado ----------
    def publish_result(self, status, result, cell, message):
        payload = {
            "status": status,
            "result": result,
            "cell": cell,
            "message": message,
            "board_valid": self.board_valid,
        }
        msg = String()
        msg.data = json.dumps(payload)
        rospy.loginfo(f"[game_logic_node] Resultado ataque: {msg.data}")
        self.result_pub.publish(msg)

    def request_board_layout(self, reason):
        msg = String()
        msg.data = reason
        self.board_request_pub.publish(msg)
        rospy.loginfo(f"[game_logic_node] Petición de layout enviada: {reason}")


def main():
    rospy.init_node("game_logic_node", anonymous=True)
    node = GameLogicNode()
    rospy.spin()


if __name__ == "__main__":
    main()
