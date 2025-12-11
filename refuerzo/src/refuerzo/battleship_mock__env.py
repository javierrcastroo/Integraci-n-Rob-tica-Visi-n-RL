import numpy as np
from gymnasium import Env, spaces


class BattleshipMockROSEnv(Env):
    """
    Entorno MOCK para ROS:
    - NO tiene tablero enemigo
    - NO simula disparos ni recompensas reales
    - SOLO proporciona observation_space y action_space coherentes
      con el entrenamiento, más obs/máscara en función de guess_board.
    """

    metadata = {"render_modes": []}

    def __init__(self, board_size=5):
        super().__init__()
        self.board_size = board_size

        # Acción: 25 celdas (5x5)
        self.action_space = spaces.Discrete(board_size * board_size)

        # Observación:
        # guess(25) + own(25) + phase(1) + turn(1) + me(1) + op(1) = 54
        obs_len = 25 + 25 + 1 + 1 + 1 + 1
        self.observation_space = spaces.Box(
            low=0.0,
            high=2.0,
            shape=(obs_len,),
            dtype=np.float32,
        )

        # Estado interno mínimo
        self.guess_board = np.zeros((board_size, board_size), np.int8)
        self.own_board = np.zeros((board_size, board_size), np.int8)

    def _mark_diagonals_as_miss(self, y, x):
        """Marca como MISS (1) las diagonales alrededor de un hit."""
        diag_offsets = [(-1,-1), (-1,1), (1,-1), (1,1)]
        for dy, dx in diag_offsets:
            ny, nx = y + dy, x + dx
            if 0 <= ny < self.board_size and 0 <= nx < self.board_size:
                if self.guess_board[ny, nx] == 0:
                    self.guess_board[ny, nx] = 1

    def _obs(self):
        guess_flat = self.guess_board.flatten().astype(np.float32)
        own_flat = self.own_board.flatten().astype(np.float32)

        phase = np.array([2.0], dtype=np.float32)
        turn = np.array([1.0], dtype=np.float32)
        me_rem = np.array([0.0], dtype=np.float32)
        op_rem = np.array([0.0], dtype=np.float32)

        return np.concatenate(
            [guess_flat, own_flat, phase, turn, me_rem, op_rem]
        ).astype(np.float32)

    def _valid_action_mask(self):
        """
        True donde NO hemos disparado todavía (guess==0).
        """
        return (self.guess_board == 0).flatten()


    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.guess_board[:] = 0
        self.own_board[:] = 0

        obs = self._obs()
        info = {"action_mask": self._valid_action_mask()}
        return obs, info

    def step(self, action):
        # No usamos esta transición para nada en ROS, pero la dejamos coherente
        obs = self._obs()
        reward = 0.0
        terminated = False
        truncated = False
        info = {"action_mask": self._valid_action_mask()}
        return obs, reward, terminated, truncated, info

    # Para sincronizar con ROS
    def update_from_feedback(self, last_action, fb):
        """
        Actualiza guess_board en función del feedback ROS.
        last_action = (row, col)
        fb = 'agua' | 'tocado' | 'hundido' | 'victoria'
        """
        row, col = last_action

        if fb == "agua":
            self.guess_board[row, col] = 1   # MISS
        else:
            self.guess_board[row, col] = 2   # HIT (tocado/hundido/victoria)
            self._mark_diagonals_as_miss(row, col)
