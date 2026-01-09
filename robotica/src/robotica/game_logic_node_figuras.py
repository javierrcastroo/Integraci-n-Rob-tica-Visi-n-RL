#!/usr/bin/env python3
import rospy
import json
from std_msgs.msg import String, Empty


def cell_name(r, c):
    return f"{chr(ord('A') + r)}{c + 1}"


class GameLogicFigurasNode:
    def __init__(self):
        # Tablero DEFENDIDO por RL (ataca P1_Figuras)
        self.rl_ships = {
            (1, 1), (1, 2),
            (3, 3)
        }
        self.rl_hits = set()

        # Tablero DEFENDIDO por P1_Figuras (ataca RL)
        self.figuras_ships = {
            (0, 0), (0, 1),
            (2, 0), (2, 2), (2, 4)
        }
        self.figuras_hits = set()

        self.max_row = 4
        self.max_col = 4


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

        rospy.loginfo("[game_logic_figuras] Nodo iniciado (P1_Figuras vs RL)")


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

        gestures = data.get("gestures", [])
        if len(gestures) != 2:
            return

        r, c = gestures
        cell = (c, r)
        name = cell_name(c, r)

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
            r = ord(coord[0]) - ord("A")
            c = int(coord[1:]) - 1
        except Exception:
            return

        cell = (r, c)
        name = cell_name(r, c)

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
                "row": cell[0],
                "col": cell[1],
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