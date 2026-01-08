#!/usr/bin/env python3
import os
import sys
import json
import random
import rospy
from std_msgs.msg import String, Empty

# --- AÑADIDO: asegurar que vemos battleship_logic.py en esta carpeta ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from battleship_logic import evaluate_board

RL_BOARDS = [
    {
        "id": "RL_1",
        "ship_two_cells": [[0, 0], [1, 0]],
        "ship_one_cells": [[3, 0], [0, 3], [4, 4]],
    },
    {
        "id": "RL_2",
        "ship_two_cells": [[2, 1], [2, 2]],
        "ship_one_cells": [[0, 0], [4, 1], [1, 4]],
    },
    {
        "id": "RL_3",
        "ship_two_cells": [[3, 3], [4, 3]],
        "ship_one_cells": [[0, 1], [2, 0], [1, 4]],
    },
    {
        "id": "RL_4",
        "ship_two_cells": [[1, 2], [1, 3]],
        "ship_one_cells": [[3, 0], [4, 2], [0, 4]],
    },
    {
        "id": "RL_5",
        "ship_two_cells": [[0, 4], [1, 4]],
        "ship_one_cells": [[3, 1], [4, 3], [2, 0]],
    },
]


def _cells_from_layout(layout):
    """
    Normaliza ship_two_cells y ship_one_cells a listas de tuplas (col, row).

    Se asume que en el JSON original vienen como [col, row],
    y aquí los convertimos a (col, row) SOLO para uso interno.
    NO modificamos el layout para no romper evaluate_board.
    """
    raw_two = layout.get("ship_two_cells", [])
    raw_one = layout.get("ship_one_cells", [])

    # JSON: [row, col] -> interno: (col, row)
    ship_two_cells = [(c[0], c[1]) for c in raw_two]
    ship_one_cells = [(c[0], c[1]) for c in raw_one]

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


def _cell_name(col, row):
    """
    Nombre de celda a partir de (col, row):

      - col -> letra (A, B, C, ...)
      - row -> número (1, 2, 3, ...)

    Por ejemplo:
      (0, 0) -> A1
      (4, 4) -> E5
    """
    return f"{chr(ord('A') + col)}{row + 1}"


class GameLogicNode(object):
    def __init__(self):
        # estado del tablero
        self.current_layout = None
        self.board_valid = False
        self.ship_two_cells = set()   # contiene (col, row)
        self.ship_one_cells = set()   # contiene (col, row)
        self.all_ship_cells = set()   # contiene (col, row)
        self.hits = set()             # contiene (col, row)

        self.rl_layout = random.choice(RL_BOARDS)
        rospy.loginfo(f"[game_logic_node] RL usa tablero {self.rl_layout['id']}")
        
        rl_two, rl_one = _cells_from_layout(self.rl_layout)
        
        self.rl_ship_two_cells = set(rl_two)
        self.rl_ship_one_cells = set(rl_one)
        self.rl_all_ship_cells = self.rl_ship_two_cells | self.rl_ship_one_cells
        self.rl_hits = set()

        # --- HARDCODE: tablero 5x5, índices 0..4 ---
        self.max_col = 4
        self.max_row = 4

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

        # Publicaciones para RL
        self.rl_turn_pub = rospy.Publisher(
            "/game/your_turn", Empty, queue_size=10
        )
        self.rl_feedback_pub = rospy.Publisher(
            "/game/feedback", String, queue_size=10
        )
        self.rl_state_pub = rospy.Publisher(
            "/game/state", String, queue_size=10
        )

        # Escuchar disparos del agente RL (coordenadas tipo "A3")
        self.rl_attack_sub = rospy.Subscriber(
            "/agent/fire_coordinates",
            String,
            self.rl_attack_cb,
            queue_size=10,
        )

        rospy.loginfo("[game_logic_node] Iniciado. Esperando tablero y ataques...")

    def notify_rl_turn(self):
        rospy.loginfo("[game_logic_node] Turno para RL")
        rospy.Timer(
            rospy.Duration(0.5),
            lambda _: self.rl_turn_pub.publish(Empty()),
            oneshot=True
        )

    def reset_rl_board(self):
        self.rl_layout = random.choice(RL_BOARDS)
        rospy.loginfo(f"[game_logic_node] Nuevo tablero RL {self.rl_layout['id']}")
    
        rl_two, rl_one = _cells_from_layout(self.rl_layout)
        self.rl_ship_two_cells = set(rl_two)
        self.rl_ship_one_cells = set(rl_one)
        self.rl_all_ship_cells = self.rl_ship_two_cells | self.rl_ship_one_cells
        self.rl_hits = set()

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

        # normalizar celdas a (col, row) para uso interno
        ship_two_cells, ship_one_cells = _cells_from_layout(layout)

        # evaluar con lógica existente (usa layout original, sin invertir coordenadas)
        ok, msg_text = evaluate_board(layout)
        rospy.loginfo(f"[game_logic_node] Evaluación tablero: ok={ok} msg='{msg_text}'")

        self.current_layout = layout
        self.board_valid = ok
        self.ship_two_cells = set(ship_two_cells)
        self.ship_one_cells = set(ship_one_cells)
        self.all_ship_cells = self.ship_two_cells | self.ship_one_cells

        # HARDCODE: mantenemos límites 0..4 independientemente de barcos
        self.max_col = 4
        self.max_row = 4

        # reseteamos impactos si ha cambiado el tablero
        self.hits = set()

    # ---------- callback ataque (gestos humano) ----------
    def attack_cb(self, msg):
        try:
            data = json.loads(msg.data)
        except Exception as e:
            rospy.logwarn(f"[game_logic_node] Error parseando ataque: {e}")
            return

        gestures = data.get("gestures", [])
        player = data.get("player", "P1")  # por si lo necesitas más adelante

        if len(gestures) != 2:
            self.publish_result(
                status="ERROR",
                result="invalid_attack",
                cell=None,
                message="Se esperaban exactamente 2 gestos (columna, fila)",
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
            # Convención: gestures[0] -> columna, gestures[1] -> fila
            col_idx = _gesture_to_digit(gestures[0])
            row_idx = _gesture_to_digit(gestures[1])
        except ValueError as e:
            self.publish_result(
                status="ERROR",
                result="invalid_gestures",
                cell=None,
                message=str(e),
            )
            return

        # comprobamos que está dentro del tablero 0..4
        if not (0 <= col_idx <= self.max_col and 0 <= row_idx <= self.max_row):
            self.publish_result(
                status="OK",
                result="out_of_bounds",
                cell={
                    "row": row_idx,
                    "col": col_idx,
                    "name": _cell_name(col_idx, row_idx),
                },
                message="Ataque fuera del tablero detectado",
            )
            return

        cell = (col_idx, row_idx)
        cell_name = _cell_name(col_idx, row_idx)

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
                    "col": col_idx,
                    "row": row_idx,
                    "name": cell_name,
                },
                message=f"Agua en {cell_name}",
            )
            self.notify_rl_turn()
            return

        # impacto en algún barco
        result = "hit"
        message = f"Tocado en {cell_name}"

        # ¿barco de 2 hundido?
        if self.ship_two_cells and cell in self.ship_two_cells:
            if self.ship_two_cells.issubset(self.hits):
                result = "sunk"
                message = f"Hundido barco de 2 (último impacto en {cell_name})"

        # ¿barco de 1 hundido?
        if cell in self.ship_one_cells:
            result = "sunk"
            message = f"Hundido barco de 1 en {cell_name}"

        # ¿todos hundidos?
        if self.all_ship_cells and self.all_ship_cells.issubset(self.hits):
            result = "sunk_all"
            message = f"¡Todos los barcos hundidos! Último impacto en {cell_name}"

        self.publish_result(
            status="OK",
            result=result,
            cell={
                "col": col_idx,
                "row": row_idx,
                "name": cell_name,
            },
            message=message,
        )

    def rl_attack_cb(self, msg):
        """
        Ataque del agente RL.
        msg.data es una coordenada tipo "A3":

          - Letra -> columna (A=0, B=1, ...)
          - Número -> fila (1->0, 2->1, ...)
        """
        if not self.rl_all_ship_cells:
            rospy.logwarn("[game_logic_node] Tablero RL no inicializado")
            return

        coord = msg.data.strip().upper()
        if len(coord) < 2:
            rospy.logwarn(f"[game_logic_node] Coordenada RL inválida: '{coord}'")
            return

        try:
            # 'A3' -> col_idx=0, row_idx=2
            col_char = coord[0]
            row_str = coord[1:]

            col_idx = ord(col_char) - ord('A')
            row_idx = int(row_str) - 1
        except Exception as e:
            rospy.logwarn(f"[game_logic_node] Error parseando coord RL '{coord}': {e}")
            return

        # Comprobamos límites de tablero 0..4
        if not (0 <= col_idx <= self.max_col and 0 <= row_idx <= self.max_row):
            rospy.loginfo(
                f"[game_logic_node] Ataque RL fuera de tablero: {coord} "
                f"(col={col_idx}, row={row_idx})"
            )
            # Simplemente ignoramos.
            return

        cell = (col_idx, row_idx)
        cell_name = _cell_name(col_idx, row_idx)

        # Ataque repetido
        if cell in self.rl_hits:
            rospy.loginfo(f"[game_logic_node] Ataque RL repetido en {cell_name}")
            rospy.Timer(
                rospy.Duration(0.2),
                lambda _: self.rl_feedback_pub.publish(String("repetido")),
                oneshot=True
            )
            return

        # Registramos impacto
        self.hits.add(cell)

        # Agua vs impacto
        if cell not in self.rl_all_ship_cells:
            # Agua
            rospy.loginfo(f"[game_logic_node] RL: Agua en {cell_name}")
            rospy.Timer(
                rospy.Duration(0.2),
                lambda _: self.rl_feedback_pub.publish(String("agua")),
                oneshot=True
            )
            return

        # Impacto
        result = "hit"
        message = f"Tocado en {cell_name}"

        # ¿barco de 2 hundido?
        if self.rl_ship_two_cells and cell in self.rl_ship_two_cells:
            if self.rl_ship_two_cells.issubset(self.rl_hits):
                result = "sunk"
                message = f"Hundido barco de 2 (último impacto en {cell_name})"

        # ¿barco de 1 hundido?
        if cell in self.rl_ship_one_cells:
            result = "sunk"
            message = f"Hundido barco de 1 en {cell_name}"

        # ¿todos hundidos?
        if self.rl_all_ship_cells and self.rl_all_ship_cells.issubset(self.rl_hits):
            result = "sunk_all"
            message = f"¡Todos los barcos hundidos! Último impacto en {cell_name}"

        rospy.loginfo(f"[game_logic_node] RL: {message}")

        # Traducir RESULT → feedback RL
        if result == "hit":
            rospy.Timer(
                rospy.Duration(0.3),
                lambda _: self.rl_feedback_pub.publish(String("tocado")),
                oneshot=True
            )
        elif result == "sunk":
            rospy.Timer(
                rospy.Duration(0.3),
                lambda _: self.rl_feedback_pub.publish(String("hundido")),
                oneshot=True
            )
        elif result == "sunk_all":
            # RL gana la partida
            rospy.Timer(
                rospy.Duration(0.3),
                lambda _: self.rl_feedback_pub.publish(String("victoria")),
                oneshot=True
            )
            rospy.Timer(
                rospy.Duration(0.3),
                lambda _: self.rl_state_pub.publish(String("win_agent")),
                oneshot=True
            )
            self.reset_rl_board()

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
