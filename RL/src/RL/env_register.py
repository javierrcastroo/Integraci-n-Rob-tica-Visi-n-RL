from gym.envs.registration import register

def register_env():
    try:
        register(
            id="BattleshipROS-v0",
            entry_point="RL.battleship_mock__env:BattleshipMockROSEnv",
            max_episode_steps=25,  # 5x5 → 25 disparos como máximo
        )
    except Exception:
        pass