#!/usr/bin/env python3
import rospy
import json
from std_msgs.msg import String, Empty


def cell_name(col, row):
    return f"{chr(ord('A') + col)}{row + 1}"

def _cells_from_layout(layout):
    raw_two = layout.get("ship_two_cells", [])
    raw_one = layout.get("ship_one_cells", [])

    ship_two_cells = [(c[0], c[1]) for c in raw_two]
    ship_one_cells = [(c[0], c[1]) for c in raw_one]

    return ship_two_cells, ship_one_cells


class GameLogicFigurasNode:
    def __init__(self):
        # Tablero de RL (ataca el usuario)
        self.rl_ships = set()
        self.rl_hits = set()
        self.board_valid = False

        # Tablero de usuario (ataca RL)
        self.figuras_layout = {
            "id": "FIGURAS_1",
            "board_size": 5,
            "ship_two_cells": [
                [0, 0], [0, 1]
            ],
            "ship_one_cells": [
                [2, 0], [2, 2], [2, 4]
            ],
        }
        two, one = _cells_from_layout(self.figuras_layout)
        self.figuras_ships = set(two) | set(one)
        self.figuras_hits = set()

        self.max_row = 4
        self.max_col = 4

        self.figuras_layout_pub = rospy.Publisher(
            "/game/layout",
            String,
            queue_size=1,
            latch=True
        )
        
        self.result_pub = rospy.Publisher(
            "battleship/attack_result", String, queue_size=10
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
        
        # Layout del tablero
        self.board_sub = rospy.Subscriber(
            "battleship/board_layout",
            String,
            self.board_cb,
            queue_size=10,
        )

        # Ataques de P1_Figuras
        self.figuras_attack_sub = rospy.Subscriber(
            "battleship/attack",
            String,
            self.figuras_attack_cb,
            queue_size=10,
        )

        # Ataques del agente RL
        self.rl_attack_sub = rospy.Subscriber(
            "/agent/fire_coordinates",
            String,
            self.rl_attack_cb,
            queue_size=10,
        )
        
        self.publish_figuras_layout()
        rospy.loginfo("[game_logic_figuras] Nodo iniciado (P1_Figuras vs RL)")

    def publish_figuras_layout(self):
        payload = {
            "id": self.figuras_layout["id"],
            "board_size": self.figuras_layout["board_size"],
            "ship_two_cells": self.figuras_layout["ship_two_cells"],
            "ship_one_cells": self.figuras_layout["ship_one_cells"],
        }
    
        msg = String()
        msg.data = json.dumps(payload)
        self.figuras_layout_pub.publish(msg)
    
        rospy.loginfo("[game_logic_figuras] Layout de Figuras publicado")


    # CALLBACK LAYOUT
    def board_cb(self, msg):
        try:
            data = json.loads(msg.data)
        except Exception as e:
            rospy.logwarn(f"[game_logic_figuras] Error parseando layout: {e}")
            return

        boards = data.get("boards", [])
        if not boards:
            rospy.logwarn("[game_logic_figuras] Layout sin boards")
            return

        layout = boards[0]

        ship_two, ship_one = _cells_from_layout(layout)

        self.rl_ships = set(ship_two) | set(ship_one)
        self.rl_hits = set()
        self.board_valid = True

        rospy.loginfo(
            f"[game_logic_figuras] Layout RL cargado con "
            f"{len(self.rl_ships)} celdas"
        )

    # TURNO RL
    def notify_rl_turn(self):
        rospy.loginfo("[game_logic_figuras] Turno para RL")
        rospy.Timer(
            rospy.Duration(0.5),
            lambda _: self.rl_turn_pub.publish(Empty()),
            oneshot=True
        )


    # ATAQUE P1_FIGURAS → TABLERO RL
    def figuras_attack_cb(self, msg):
        try:
            data = json.loads(msg.data)
            rospy.logwarn(f"[game_logic_figuras] Ataque P1_Figuras: {data}")
        except Exception:
            return
            
        if not self.board_valid:
            rospy.logwarn("[game_logic_figuras] Tablero RL no cargado")
            return

        gestures = data.get("gestures", [])
        if len(gestures) != 2:
            return

        col, row = gestures
        cell = (col, row)
        name = cell_name(col, row)

        if not (0 <= col <= self.max_col and 0 <= row <= self.max_row):
            self.publish_result("out_of_bounds", (col, row), "Ataque fuera del tablero detectado")
            return


        # Repetido
        if cell in self.rl_hits:
            self.publish_result("repeated", cell, f"Repetido {name}")
            return

        self.rl_hits.add(cell)

        # Agua
        if cell not in self.rl_ships:
            self.publish_result("miss", cell, f"Agua en {name}")
            self.notify_rl_turn()
            return

        # Impacto
        self.rl_ships.remove(cell)
        result = "sunk"

        if not self.rl_ships:
            result = "sunk_all"

        self.publish_result(result, cell, f"Impacto en {name}")


    # ATAQUE RL → TABLERO FIGURAS
    def rl_attack_cb(self, msg):
        coord = msg.data.strip().upper()
        if len(coord) < 2:
            return

        try:
            col = ord(coord[0]) - ord("A")
            row = int(coord[1:]) - 1
        except Exception:
            return

        cell = (col, row)
        name = cell_name(col, row)

        # Repetido
        if cell in self.figuras_hits:
            rospy.Timer(rospy.Duration(0.2), lambda _: self.rl_feedback_pub.publish(String("repetido")), oneshot=True)
            return

        self.figuras_hits.add(cell)

        # Agua
        if cell not in self.figuras_ships:
            rospy.Timer(rospy.Duration(0.2), lambda _: self.rl_feedback_pub.publish(String("agua")), oneshot=True)
            rospy.loginfo(f"[game_logic_figuras] RL agua {name}")
            return

        # Impacto
        self.figuras_ships.remove(cell)

        if not self.figuras_ships:
            rospy.Timer(rospy.Duration(0.2), lambda _: self.rl_feedback_pub.publish(String("victoria")), oneshot=True)
            rospy.Timer(rospy.Duration(0.2), lambda _: self.rl_state_pub.publish(String("win_agent")), oneshot=True)
            rospy.loginfo("[game_logic_figuras] RL gana la partida")
            return

        rospy.Timer(rospy.Duration(0.2), lambda _: self.rl_feedback_pub.publish(String("tocado")), oneshot=True)
        rospy.loginfo(f"[game_logic_figuras] RL tocado {name}")
        


    # RESULTADOS VISUALES
    def publish_result(self, result, cell, message):
        payload = {
            "status": "OK",
            "result": result,
            "cell": {
                "row": cell[1],
                "col": cell[0],
                "name": cell_name(cell[0], cell[1]),    
            },
            "message": message,
        }
        msg = String()
        msg.data = json.dumps(payload)
        self.result_pub.publish(msg)


def main():
    rospy.init_node("game_logic_figuras_node")
    GameLogicFigurasNode()
    rospy.spin()

if __name__ == "__main__":
    main()
