from gymnasium.envs.registration import register
import sys
import os

def register_env():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    if base_dir not in sys.path:
        sys.path.append(base_dir)
    
    try:
        register(
            id="BattleshipROS-v0",
            entry_point="battleship_mock__env:BattleshipMockROSEnv",

            max_episode_steps=25,  # 5x5 → 25 disparos como máximo
        )
    except Exception:
        pass
