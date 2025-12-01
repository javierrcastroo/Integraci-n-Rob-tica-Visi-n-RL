#!/usr/bin/env python3

import rospy
import rospkg
import subprocess
import json

from std_msgs.msg import String, Empty

BOARD_SIZE = 5

model_proc = None
model_stdin = None
model_stdout = None

last_action = None
fire_pub = None


# ------------------------- COORDS ----------------------

def index_to_coord(row, col):
    return f"{chr(ord('A') + row)}{col + 1}"


# ------------------- INFERENCIA REMOTA -----------------

def start_rl_server():
    global model_proc, model_stdin, model_stdout

    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path("RL")

    server = f"{pkg_path}/src/RL/rl_model_server.py"
    model  = f"{pkg_path}/src/RL/models/saved_models/best_model"

    python = f"{pkg_path}/venv_rl/bin/python"

    model_proc = subprocess.Popen(
        [python, server, model],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, bufsize=1
    )

    model_stdin = model_proc.stdin
    model_stdout = model_proc.stdout

    rospy.loginfo("[RL] Servidor RL lanzado.")


def rl_predict():
    global model_stdin, model_stdout

    model_stdin.write(json.dumps({"cmd": "predict"}) + "\n")
    model_stdin.flush()

    resp = model_stdout.readline()
    data = json.loads(resp)

    return data["row"], data["col"]


# ------------------------ LÓGICA -----------------------

def agent_fire():
    global last_action

    row, col = rl_predict()
    last_action = (row, col)

    coord = index_to_coord(row, col)
    rospy.loginfo(f"[RL] Disparo → {coord}")
    fire_pub.publish(coord)


# --------------------- CALLBACKS ----------------------

def your_turn_callback(_):
    rospy.loginfo("[RL] Turno recibido")
    agent_fire()


def feedback_callback(msg):
    global last_action

    if last_action is None:
        return

    fb = msg.data.strip().lower()

    model_stdin.write(json.dumps({
        "cmd": "feedback",
        "row": last_action[0],
        "col": last_action[1],
        "feedback": fb
    }) + "\n")
    model_stdin.flush()

    if fb in ["tocado", "hundido"]:
        agent_fire()
    elif fb == "victoria":
        reset_internal_state()


def state_callback(msg):
    state = msg.data.strip().lower()
    if "win" in state:
        reset_internal_state()


def reset_internal_state():
    global last_action
    last_action = None
    model_stdin.write(json.dumps({"cmd": "reset"}) + "\n")
    model_stdin.flush()
    rospy.loginfo("[RL] Estado reiniciado")


# ----------------------- MAIN -------------------------

if __name__ == "__main__":
    rospy.init_node("rl_agent_node")

    start_rl_server()

    fire_pub = rospy.Publisher("/agent/fire_coordinates", String, queue_size=10)

    rospy.Subscriber("/game/your_turn", Empty, your_turn_callback)
    rospy.Subscriber("/game/feedback", String, feedback_callback)
    rospy.Subscriber("/game/state", String, state_callback)

    rospy.loginfo("[RL] Nodo iniciado. Esperando turnos...")
    rospy.spin()

