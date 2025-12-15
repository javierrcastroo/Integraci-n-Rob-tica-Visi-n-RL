#!/home/laboratorio/ros_workspace/refuerzo/venv_rl/bin/python
# -*- coding: utf-8 -*-

import sys
import json
import numpy as np

import gymnasium as gym
from env_register import register_env
register_env()

from battleship_mock__env import BattleshipMockROSEnv
from sb3_contrib import MaskablePPO


if len(sys.argv) < 2:
    print("Uso: rl_model_server.py MODEL_PATH", file=sys.stderr)
    sys.exit(1)

MODEL_PATH = sys.argv[1]

print(f"Cargando modelo PPO: {MODEL_PATH}", file=sys.stderr)

dummy_env = gym.make("BattleshipROS-v0")
model = MaskablePPO.load(MODEL_PATH, env=dummy_env)

print("Modelo cargado correctamente.", file=sys.stderr)


# Crear entorno mock REAL con estado de actualización
env = BattleshipMockROSEnv(board_size=5)
env.reset()

print("Entorno BattleshipMockROSEnv inicializado.", file=sys.stderr)


# Bucle principal: stdin → acción / feedback
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue

    data = json.loads(line)
    cmd = data["cmd"]

    # PETICIÓN DE ACCIÓN
    if cmd == "predict":
        obs = env._obs()
        mask = env._valid_action_mask()

        action, _ = model.predict(obs, action_masks=mask, deterministic=True)

        row = action // env.board_size
        col = action % env.board_size

        sys.stdout.write(json.dumps({
            "action": int(action),
            "row": int(row),
            "col": int(col)
        }) + "\n")
        sys.stdout.flush()

    # FEEDBACK DESDE ROS
    elif cmd == "feedback":
        row = data["row"]
        col = data["col"]
        fb  = data["feedback"]
            
        env._update_from_feedback((row, col), fb)

    # RESET DESDE ROS
    elif cmd == "reset":
        env.reset()
