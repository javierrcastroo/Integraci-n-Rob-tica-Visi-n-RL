from gymnasium.envs.registration import register
import rospkg
import sys
import os

def register_env():
    rospack = rospkg.RosPack()
    RL_PATH = rospack.get_path("RL") + "src/RL"
    
    if RL_PATH not in sys.path:
        sys.path.append(RL_PATH)
    
    try:
        register(
            id="BattleshipROS-v0",
            entry_point="battleship_mock_env:BattleshipMockROSEnv",

            max_episode_steps=25,  # 5x5 → 25 disparos como máximo
        )
    except Exception:
        pass
