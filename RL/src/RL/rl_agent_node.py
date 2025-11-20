#!/usr/bin/env python3
import rospy
import numpy as np

from std_msgs.msg import String, Empty
from sb3_contrib import MaskablePPO

# --------------- CONFIGURACIÓN DEL AGENTE -----------------

BOARD_SIZE = 5

# Valores iguales que en tu entorno Gym
MISS = 1
HIT = 2

# Se carga el mejor modelo entrenado
MODEL_PATH = rospy.get_package_path("RL") + "/src/RL/models/best_model.zip"

# Estado interno del agente (igual que en el entorno Gym)
guess_board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
own_board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)  # NO USADO
ships_remaining = 0  # no usado en ROS pero mantenido para obs
phase = 2  # siempre fase de disparo
last_action = None

# Publicador global
fire_pub = None


# -----------------------------------------------------------
# -------------------- UTILIDADES RL ------------------------
# -----------------------------------------------------------

def coord_to_index(coord):
    """Convierte 'A3' → (row, col)."""
    coord = coord.strip().upper()
    row = ord(coord[0]) - ord('A')
    col = int(coord[1:]) - 1
    return row, col


def index_to_coord(row, col):
    return f"{chr(ord('A') + row)}{col + 1}"


def build_mask():
    """True = permitido, False = ya disparado."""
    flat = (guess_board == 0).flatten()
    return flat


def build_obs():
    """Construye la observación como en tu entorno Gym."""
    guess_flat = guess_board.flatten().astype(np.float32)
    own_flat = own_board.flatten().astype(np.float32)
    phase_id = np.array([2.0], dtype=np.float32)
    turn_bit = np.array([1.0], dtype=np.float32)
    me_remaining = np.array([0.0], dtype=np.float32)
    op_remaining = np.array([0.0], dtype=np.float32)
    return np.concatenate([guess_flat, own_flat, phase_id, turn_bit, me_remaining, op_remaining])


# -----------------------------------------------------------
# ------------------ PUBLICAR DISPARO -----------------------
# -----------------------------------------------------------

def agent_fire():
    """Genera acción RL y la publica vía ROS."""
    global last_action

    obs = build_obs()
    mask = build_mask()

    action, _ = model.predict(obs, action_masks=mask, deterministic=True)

    row = action // BOARD_SIZE
    col = action % BOARD_SIZE

    last_action = (row, col)
    coord_msg = index_to_coord(row, col)

    rospy.loginfo(f"[RL] Disparo → {coord_msg}")
    fire_pub.publish(coord_msg)


# -----------------------------------------------------------
# ------------------ CALLBACKS ROS --------------------------
# -----------------------------------------------------------

def your_turn_callback(msg):
    """game_logic avisa que es nuestro turno."""
    rospy.loginfo("[RL] Turno recibido")
    agent_fire()


def feedback_callback(msg):
    """
    Recibe feedback del disparo:
    Se espera mensajes String con valores:
        'agua'
        'tocado'
        'hundido'
        'victoria'
    """
    global guess_board, last_action

    if last_action is None:
        return

    fb = msg.data.strip().lower()
    row, col = last_action

    if fb == "agua":
        guess_board[row, col] = MISS
        rospy.loginfo("[RL] Agua")

    elif fb == "tocado":
        guess_board[row, col] = HIT
        rospy.loginfo("[RL] Tocado → sigo tirando")
        agent_fire()  # turno extra

    elif fb == "hundido":
        guess_board[row, col] = HIT
        rospy.loginfo("[RL] Hundido")
        agent_fire()  # turno extra por norma del juego

    elif fb == "victoria":
        guess_board[row, col] = HIT
        rospy.loginfo("[RL] ¡Victoria del agente!")
        reset_internal_state()

    else:
        rospy.logwarn(f"[RL-WARN] Feedback desconocido: {fb}")


def state_callback(msg):
    """Recibe estado final de partida, si existe."""
    state = msg.data.strip().lower()

    if state == "win_agent":
        rospy.loginfo("[RL] GameLogic dice: victoria del agente")
        reset_internal_state()

    elif state == "win_human":
        rospy.loginfo("[RL] GameLogic dice: victoria humana")
        reset_internal_state()


# -----------------------------------------------------------
# ------------------ RESET DEL ESTADO RL --------------------
# -----------------------------------------------------------

def reset_internal_state():
    """Resetea el tablero interno del agente."""
    global guess_board, last_action
    guess_board[:] = 0
    last_action = None
    rospy.loginfo("[RL] Estado interno reseteado para nueva partida")


# -----------------------------------------------------------
# ------------------------- MAIN ----------------------------
# -----------------------------------------------------------

if __name__ == "__main__":
    rospy.init_node("rl_agent_node")

    rospy.loginfo("[RL] Cargando modelo PPO...")
    model = MaskablePPO.load(MODEL_PATH)
    rospy.loginfo("[RL] Modelo cargado.")

    # Publisher
    fire_pub = rospy.Publisher("/agent/fire_coordinates", String, queue_size=10)

    # Subscribers
    rospy.Subscriber("/game/your_turn", Empty, your_turn_callback)
    rospy.Subscriber("/game/feedback", String, feedback_callback)
    rospy.Subscriber("/game/state", String, state_callback)

    rospy.loginfo("[RL] Nodo del agente RL inicializado. Esperando turnos...")
    rospy.spin()
