#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(__file__))

import numpy as np
import cv2
import rospy
import rospkg
import subprocess
import json

from std_msgs.msg import String, Empty
from board_visualizer import draw_guess_board


BOARD_SIZE = 5

model_proc = None
model_stdin = None
model_stdout = None

last_action = None
fire_pub = None
rl_truth_board = None
guess_window_name = "Guess Board"
truth_window_name = "Tablero Agente RL"

# Matriz interna que refleja cómo el agente "ve" el tablero
guess_board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)


def update_guis():
    """Redibuja el tablero de guess en la ventana (solo desde el hilo principal)."""
    global last_action, guess_board, rl_truth_board

    img = draw_guess_board(guess_board, last_shot=last_action)
    cv2.imshow(guess_window_name, img)
    # Debug opcional:
    # rospy.loginfo(f"[refuerzo] img shape={img.shape}, dtype={img.dtype}, "
    #               f"min={img.min()}, max={img.max()}")
    if rl_truth_board is not None:
        img_truth = draw_guess_board(rl_truth_board, last_shot=None)
        cv2.imshow(truth_window_name, img_truth)
    # Importante: waitKey en el mismo hilo siempre
    cv2.waitKey(1)


def mark_diagonals_as_miss(y, x):
    """Marca como MISS (1) las diagonales alrededor de un hit."""
    global guess_board
    diag_offsets = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for dy, dx in diag_offsets:
        ny, nx = y + dy, x + dx
        if 0 <= ny < len(guess_board) and 0 <= nx < len(guess_board[0]):
            if guess_board[ny, nx] == 0:
                guess_board[ny, nx] = 1

def rl_layout_cb(msg):
    global rl_truth_board
    
    layout = json.loads(msg.data)
    rl_truth_board = np.ones((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)  # todo agua

    for c, r in layout.get("ship_two_cells", []):
        rl_truth_board[r, c] = 2

    for c, r in layout.get("ship_one_cells", []):
        rl_truth_board[r, c] = 2

    rospy.loginfo(f"[refuerzo] Layout RL recibido: {layout}")

def index_to_coord(row, col):
    return f"{chr(ord('A') + col)}{row + 1}"


def stream_server_stderr(proc):
    """Captura stderr del servidor RL en un hilo (solo logs, sin GUI)."""
    import threading

    def _reader():
        for line in proc.stderr:
            rospy.loginfo("[refuerzo-SERVER] " + line.strip())

    th = threading.Thread(target=_reader, daemon=True)
    th.start()


def start_rl_server():
    """Lanza rl_model_server.py dentro del venv_rl."""
    global model_proc, model_stdin, model_stdout

    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path("refuerzo")

    server = f"{pkg_path}/src/refuerzo/rl_model_server.py"
    model = f"{pkg_path}/src/refuerzo/models/saved_models/best_model"

    # Usar Python de venv_rl
    python = f"{pkg_path}/venv_rl/bin/python"

    model_proc = subprocess.Popen(
        [python, server, model],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    model_stdin = model_proc.stdin
    model_stdout = model_proc.stdout

    rospy.loginfo("[refuerzo] Servidor RL lanzado.")

    stream_server_stderr(model_proc)


def rl_predict():
    """Envía {cmd: predict} al servidor RL y recibe row, col."""
    global model_stdin, model_stdout

    # Pedimos predicción
    model_stdin.write(json.dumps({"cmd": "predict"}) + "\n")
    model_stdin.flush()

    while True:
        resp = model_stdout.readline().strip()
        if not resp:
            continue

        # 1. Debe empezar por '{'
        if not resp.startswith("{"):
            rospy.logwarn(f"[refuerzo] Ignorando línea no JSON: '{resp}'")
            continue

        # 2. Intentar parsear como JSON
        try:
            data = json.loads(resp)
        except json.JSONDecodeError:
            rospy.logwarn(f"[refuerzo] Línea JSON inválida: '{resp}'")
            continue

        # 3. Validar campos obligatorios
        if "row" not in data or "col" not in data:
            rospy.logwarn(f"[refuerzo] JSON sin row/col: {data}")
            continue

        try:
            row = int(data["row"])
            col = int(data["col"])
        except Exception:
            rospy.logwarn(f"[refuerzo] row/col no son enteros: {data}")
            continue

        return row, col


def agent_fire():
    """Lógica de disparo del agente (sin tocar GUI)."""
    global last_action

    row, col = rl_predict()
    last_action = (row, col)

    coord = index_to_coord(row, col)
    rospy.loginfo(f"[refuerzo] Disparo → {coord}")

    fire_pub.publish(coord)
    # NO llamamos a update_gui() aquí → solo desde el hilo principal


def your_turn_callback(_):
    rospy.loginfo("[refuerzo] Turno recibido")
    agent_fire()


def feedback_callback(msg):
    """Recibe feedback del GameLogic y actualiza guess_board + RL (sin tocar GUI)."""
    global guess_board, last_action, model_stdin

    if last_action is None:
        return

    fb = msg.data.strip().lower()
    row, col = last_action

    # Notificar al servidor RL
    model_stdin.write(json.dumps({
        "cmd": "feedback",
        "row": row,
        "col": col,
        "feedback": fb
    }) + "\n")
    model_stdin.flush()

    # Actualización del tablero local
    if fb == "agua":
        guess_board[row, col] = 1
        rospy.loginfo("[refuerzo] Agua")

    elif fb in ["tocado", "hundido"]:
        guess_board[row, col] = 2
        mark_diagonals_as_miss(row, col)
        rospy.loginfo(f"[refuerzo] {fb.capitalize()} → turno extra")
        agent_fire()

    elif fb == "repetido":
        rospy.loginfo("[refuerzo] Disparo repetido → turno extra")
        agent_fire()

    elif fb == "victoria":
        guess_board[row, col] = 2
        rospy.loginfo("[refuerzo] ¡Victoria del agente!")
        reset_internal_state()

    # NO llamamos a update_gui() aquí; el main loop pinta periódicamente


def state_callback(msg):
    """Detecta victoria de humano o agente."""
    state = msg.data.strip().lower()

    if "win" in state:
        reset_internal_state()


def reset_internal_state():
    """Reinicio completo del estado interno del agente RL."""
    global last_action, guess_board, model_stdin

    guess_board[:] = 0
    last_action = None

    model_stdin.write(json.dumps({"cmd": "reset"}) + "\n")
    model_stdin.flush()

    rospy.loginfo("[refuerzo] Estado interno reseteado")
    # Tampoco tocamos GUI aquí


if __name__ == "__main__":
    rospy.init_node("rl_agent_node")

    # Crear ventana de OpenCV en el hilo principal (igual que en hand_main_viewer)
    cv2.namedWindow(guess_window_name)
    cv2.namedWindow(truth_window_name)

    # Pintar tablero vacío al inicio
    update_guis()

    # Lanzar modelo RL en venv_rl
    start_rl_server()

    fire_pub = rospy.Publisher("/agent/fire_coordinates", String, queue_size=10)

    rospy.Subscriber("/game/your_turn", Empty, your_turn_callback)
    rospy.Subscriber("/game/feedback", String, feedback_callback)
    rospy.Subscriber("/game/state", String, state_callback)
    rospy.Subscriber("/game/layout", String, rl_layout_cb,queue_size=1)

    rospy.loginfo("[refuerzo] Nodo iniciado. Esperando turnos...")

    rate = rospy.Rate(15)  # 15 FPS de refresco
    while not rospy.is_shutdown():
        update_guis()
        rate.sleep()

    cv2.destroyAllWindows()
