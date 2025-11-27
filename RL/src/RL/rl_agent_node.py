#!/usr/bin/env python3
import rospy
import rospkg
import numpy as np
import gymnasium as gym

from std_msgs.msg import String, Empty
from sb3_contrib import MaskablePPO

from env_register import register_env

register_env()

BOARD_SIZE = 5

rospack = rospkg.RosPack()
MODEL_PATH = rospack.get_path("RL") + "/src/RL/models/saved_models/best_model"

last_action = None
fire_pub = None
env = None
model = None


def coord_to_index(coord):
    coord = coord.strip().upper()
    row = ord(coord[0]) - ord("A")
    col = int(coord[1:]) - 1
    return row, col


def index_to_coord(row, col):
    return f"{chr(ord('A') + row)}{col + 1}"


def agent_fire():
    """Genera acción RL y la publica vía ROS."""
    global last_action, env, model

    # Obs y máscara del entorno mock
    obs = env._obs()
    mask = env._valid_action_mask()

    action, _ = model.predict(obs, action_masks=mask, deterministic=True)

    row = action // BOARD_SIZE
    col = action % BOARD_SIZE
    last_action = (row, col)

    coord_msg = index_to_coord(row, col)

    rospy.loginfo(f"[RL] Disparo → {coord_msg}")
    fire_pub.publish(coord_msg)


def your_turn_callback(msg):
    rospy.loginfo("[RL] Turno recibido")
    agent_fire()


def feedback_callback(msg):
    global last_action, env

    if last_action is None:
        return

    fb = msg.data.strip().lower()

    # Actualizar entorno mock con el resultado del disparo
    env.update_from_feedback(last_action, fb)

    if fb == "agua":
        rospy.loginfo("[RL] Agua")

    elif fb == "tocado":
        rospy.loginfo("[RL] Tocado → sigo tirando")
        agent_fire()

    elif fb == "hundido":
        rospy.loginfo("[RL] Hundido → sigo tirando")
        agent_fire()

    elif fb == "victoria":
        rospy.loginfo("[RL] ¡Victoria del agente!")
        reset_internal_state()

    else:
        rospy.logwarn(f"[RL-WARN] Feedback desconocido: {fb}")


def state_callback(msg):
    state = msg.data.strip().lower()

    if state == "win_agent":
        rospy.loginfo("[RL] GameLogic dice: victoria del agente")
        reset_internal_state()

    elif state == "win_human":
        rospy.loginfo("[RL] GameLogic dice: victoria humana")
        reset_internal_state()


def reset_internal_state():
    global last_action, env
    env.reset()
    last_action = None
    rospy.loginfo("[RL] Estado interno reseteado para nueva partida")


if __name__ == "__main__":
    rospy.init_node("rl_agent_node")

    rospy.loginfo("[RL] Creando entorno BattleshipROS-v0...")
    env = gym.make("BattleshipROS-v0")

    rospy.loginfo("[RL] Cargando modelo PPO...")
    model = MaskablePPO.load(MODEL_PATH, env=env)
    rospy.loginfo("[RL] Modelo cargado.")

    fire_pub = rospy.Publisher("/agent/fire_coordinates", String, queue_size=10)

    rospy.Subscriber("/game/your_turn", Empty, your_turn_callback)
    rospy.Subscriber("/game/feedback", String, feedback_callback)
    rospy.Subscriber("/game/state", String, state_callback)

    rospy.loginfo("[RL] Nodo del agente RL inicializado. Esperando turnos...")
    rospy.spin()
