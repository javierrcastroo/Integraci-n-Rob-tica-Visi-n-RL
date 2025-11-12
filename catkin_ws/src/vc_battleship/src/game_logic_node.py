#!/usr/bin/env python
import ast
import json

import rospy
from std_msgs.msg import String

from game_logic import GameLogic

class GameLogicNode(object):
    def __init__(self):
        rospy.init_node('game_logic_node')

        # lógica de hundir la flota
        self.game = GameLogic()

        # publicadores
        self.pub_result = rospy.Publisher('/game/shot_result', String, queue_size=10)
        self.pub_target = rospy.Publisher('/game/target_xy', String, queue_size=10)
        self.pub_board_status = rospy.Publisher('/game/board_status', String, queue_size=10)
        self.pub_ships = rospy.Publisher('/game/ships_xy', String, queue_size=10)

        # suscriptor a los gestos
        rospy.Subscriber('/gesture/attack_list', String, self.cb_attack_list)

        # estado del tablero detectado
        rospy.Subscriber('/board/object_states', String, self.cb_board_objects)

        self.last_board_status = None

    def cb_attack_list(self, msg):
        # msg.data es algo como "[(2, 4)]" o "[(2,4), (1,5)]"
        try:
            attacks = ast.literal_eval(msg.data)
        except Exception as e:
            rospy.logerr(f"[GAME] No pude parsear la lista: {msg.data} ({e})")
            return

        if not isinstance(attacks, list):
            rospy.logwarn("[GAME] El mensaje no es una lista.")
            return

        for (col1, row1) in attacks:
            # gesture manda (col,fila) empezando en 1
            col = col1 - 1
            row = row1 - 1

            result = self.game.shoot(row, col)

            if result.get("result") == "sin_tablero":
                rospy.logwarn("[GAME] Ignorando disparo porque el tablero aún no es válido")
                continue

            # publicar resultado de juego
            out = String()
            base = f"{result['result']}|extra_turn={result['extra_turn']}|cell=({col1},{row1})"
            if result.get("ship_type"):
                base += f"|ship={result['ship_type']}"
            out.data = base
            self.pub_result.publish(out)
            rospy.loginfo(f"[GAME] Disparo a ({col1},{row1}) -> {result}")

            coords = self.game.get_cell_coordinates(row, col)
            if coords is not None:
                msg_xy = String()
                msg_xy.data = json.dumps(
                    {
                        "cell": [col1, row1],
                        "dx_cm": coords[0],
                        "dy_cm": coords[1],
                    }
                )
                self.pub_target.publish(msg_xy)

    def cb_board_objects(self, msg):
        try:
            objects = json.loads(msg.data)
        except Exception as exc:
            rospy.logwarn(f"[GAME] No pude interpretar el estado del tablero: {exc}")
            return

        if not isinstance(objects, list):
            rospy.logwarn("[GAME] El estado del tablero debería ser una lista")
            return

        status = self.game.update_board_from_detections(objects)

        if status != self.last_board_status:
            status_msg = String()
            status_msg.data = json.dumps(status, ensure_ascii=False)
            self.pub_board_status.publish(status_msg)
            self.last_board_status = status

        if status.get("valid"):
            if status.get("changed"):
                ships_msg = String()
                ships_msg.data = json.dumps(
                    self.game.get_ship_coordinates(), ensure_ascii=False
                )
                self.pub_ships.publish(ships_msg)
                rospy.loginfo("[GAME] Tablero válido actualizado")
        else:
            rospy.logwarn_throttle(2.0, f"[GAME] Tablero inválido: {status.get('message')}")

def main():
    node = GameLogicNode()
    rospy.spin()

if __name__ == '__main__':
    main()
