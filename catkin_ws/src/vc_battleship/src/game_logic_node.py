#!/usr/bin/env python
import ast
import json

import rospy
from std_msgs.msg import String
from std_srvs.srv import Trigger

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
        self.pub_ammo = rospy.Publisher('/game/ammo_pick', String, queue_size=10)

        # suscriptor a los gestos
        rospy.Subscriber('/gesture/attack_list', String, self.cb_attack_list)

        self.board_service_name = rospy.get_param('~board_service', '/board/capture_state')
        self.board_service = None
        self.last_board_status = None
        self.last_ships_payload = None
        self.last_ammo_payload = None

        self._connect_board_service()

        # obtención inicial del tablero (si está disponible)
        self.refresh_board_state(reason="inicio")

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

        if not attacks:
            return

        if not self.refresh_board_state(reason="pre_ataque"):
            rospy.logwarn("[GAME] No hay tablero válido todavía")
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

        # actualizar estado del tablero tras procesar los ataques
        self.refresh_board_state(reason="post_ataque")
        self._publish_ships(force=True)
        self._publish_ammo(force=True)

    def refresh_board_state(self, reason="manual"):
        if self.board_service is None:
            self._connect_board_service()
        if self.board_service is None:
            return False

        try:
            resp = self.board_service()
        except rospy.ServiceException as exc:
            rospy.logwarn(f"[GAME] Error solicitando tablero ({reason}): {exc}")
            return False

        if not resp.success:
            rospy.logwarn_throttle(2.0, f"[GAME] Tablero no disponible ({reason})")
            return False

        try:
            payload = json.loads(resp.message)
        except Exception as exc:
            rospy.logwarn(f"[GAME] Respuesta inválida del tablero: {exc}")
            return False

        status = self.game.update_board_from_detections(payload)

        if status != self.last_board_status:
            status_msg = String()
            status_msg.data = json.dumps(status, ensure_ascii=False)
            self.pub_board_status.publish(status_msg)
            self.last_board_status = status

        if status.get("valid"):
            self._publish_ships()
            self._publish_ammo()
            if status.get("changed"):
                rospy.loginfo(f"[GAME] Tablero actualizado tras {reason}")
        else:
            rospy.logwarn_throttle(2.0, f"[GAME] Tablero inválido: {status.get('message')}")
            self._publish_ammo()

        return status.get("valid", False)

    def _connect_board_service(self):
        if self.board_service is not None:
            return
        try:
            rospy.wait_for_service(self.board_service_name, timeout=5.0)
            self.board_service = rospy.ServiceProxy(self.board_service_name, Trigger)
            rospy.loginfo(f"[GAME] Conectado al servicio {self.board_service_name}")
        except (rospy.ServiceException, rospy.ROSException):
            rospy.logwarn(
                f"[GAME] No se pudo conectar al servicio {self.board_service_name}"
            )

    def _publish_ships(self, force=False):
        if not self.game.board_ready:
            return
        payload = json.dumps(self.game.get_ship_coordinates(), ensure_ascii=False)
        if not force and payload == self.last_ships_payload:
            return
        ships_msg = String()
        ships_msg.data = payload
        self.pub_ships.publish(ships_msg)
        self.last_ships_payload = payload

    def _publish_ammo(self, force=False):
        ammo_state = self.game.get_ammo_state()
        payload = json.dumps(ammo_state, ensure_ascii=False)
        if not force and payload == self.last_ammo_payload:
            return
        msg = String()
        msg.data = payload
        self.pub_ammo.publish(msg)
        self.last_ammo_payload = payload

def main():
    node = GameLogicNode()
    rospy.spin()

if __name__ == '__main__':
    main()
