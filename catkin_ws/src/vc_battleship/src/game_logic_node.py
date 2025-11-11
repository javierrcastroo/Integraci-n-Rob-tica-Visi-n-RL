#!/usr/bin/env python
import rospy
from std_msgs.msg import String
import ast

from vc_battleship.game_logic import GameLogic

# si ya creaste el service que hicimos antes:
# from your_pkg.srv import ResolveCell, ResolveCellRequest

class GameLogicNode(object):
    def __init__(self):
        rospy.init_node('game_logic_node')

        # lógica de hundir la flota
        self.game = GameLogic()

        # publicadores
        self.pub_result = rospy.Publisher('/game/shot_result', String, queue_size=10)
        self.pub_target = rospy.Publisher('/game/target_xy', String, queue_size=10)

        # suscriptor a los gestos
        rospy.Subscriber('/gesture/attack_list', String, self.cb_attack_list)

        # service del tablero (puede que no exista aún)
        self.board_srv = None
        try:
            rospy.wait_for_service('/board/resolve_cell', timeout=2.0)
            self.board_srv = rospy.ServiceProxy('/board/resolve_cell', None)  # ← lo ajustamos abajo
        except rospy.ROSException:
            rospy.logwarn("[GAME] Service /board/resolve_cell no disponible todavía. Solo haremos lógica.")

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

            # publicar resultado de juego
            out = String()
            out.data = f"{result['result']}|extra_turn={result['extra_turn']}|cell=({col1},{row1})"
            self.pub_result.publish(out)
            rospy.loginfo(f"[GAME] Disparo a ({col1},{row1}) -> {result}")

            # intentar pedir al tablero la posición real
            # OJO: aquí necesitamos el tipo real del service.
            # Como antes te puse un srv custom (ResolveCell), te dejo el esquema abajo.
            # Si aún no lo tenéis, comenta este bloque.
            """
            if self.board_srv is not None:
                try:
                    req = ResolveCellRequest()
                    req.col = col1
                    req.row = row1
                    resp = self.board_srv(req)
                    if resp.success:
                        msg_xy = String()
                        msg_xy.data = f"({resp.x}, {resp.y}) in {resp.frame_id}"
                        self.pub_target.publish(msg_xy)
                    else:
                        rospy.logwarn("[GAME] El board no pudo resolver la casilla.")
                except rospy.ServiceException as e:
                    rospy.logwarn(f"[GAME] Error llamando a /board/resolve_cell: {e}")
            """

def main():
    node = GameLogicNode()
    rospy.spin()

if __name__ == '__main__':
    main()
