import random

BOARD_SIZE = 5

class GameLogic:
    def __init__(self):
        # 0 = agua, 1 = barco
        self.board = [[0]*BOARD_SIZE for _ in range(BOARD_SIZE)]
        # para saber si ya se disparó
        self.shots = [[False]*BOARD_SIZE for _ in range(BOARD_SIZE)]
        # lista de barcos: cada barco es lista de (r,c)
        self.ships = []

        # configuración que pediste
        self.ship_lengths = [3, 1, 1, 1]

        self.place_all_ships()

    # ------------------------------------------------------------------
    # COLOCACIÓN
    # ------------------------------------------------------------------
    def place_all_ships(self):
        for length in self.ship_lengths:
            placed = False
            while not placed:
                placed = self.try_place_ship(length)

    def try_place_ship(self, length):
        # orientación: 0 = horizontal, 1 = vertical
        orient = random.choice([0, 1])
        if orient == 0:
            # horizontal
            row = random.randint(0, BOARD_SIZE - 1)
            col = random.randint(0, BOARD_SIZE - length)
            coords = [(row, col + i) for i in range(length)]
        else:
            # vertical
            row = random.randint(0, BOARD_SIZE - length)
            col = random.randint(0, BOARD_SIZE - 1)
            coords = [(row + i, col) for i in range(length)]

        # comprobar que no hay solape y no hay barcos pegados
        if self.can_place(coords):
            # colocar
            for r, c in coords:
                self.board[r][c] = 1
            self.ships.append(coords)
            return True
        return False

    def can_place(self, coords):
        for r, c in coords:
            # ocupada?
            if self.board[r][c] == 1:
                return False
            # vecinos? no puede haber ni en diagonal
            if not self.no_neighbors(r, c):
                return False
        return True

    def no_neighbors(self, r, c):
        # mira en las 8 direcciones alrededor
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                    if self.board[nr][nc] == 1:
                        # si el vecino es parte del mismo barco que estamos intentando poner,
                        # todavía no está puesto en board, así que esto funciona
                        return False
        return True

    # ------------------------------------------------------------------
    # DISPAROS
    # ------------------------------------------------------------------
    def shoot(self, row, col):
        """
        row, col en 0..4
        Devuelve un dict con:
        {
            "result": "agua" | "tocado" | "hundido" | "repetido" | "victoria",
            "extra_turn": bool
        }
        reglas:
        - si fallas -> extra_turn = False
        - si tocas -> extra_turn = True
        - si hundes -> extra_turn = True
        """
        # disparo fuera
        if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
            return {"result": "fuera", "extra_turn": False}

        # ya disparaste aquí
        if self.shots[row][col]:
            return {"result": "repetido", "extra_turn": False}

        self.shots[row][col] = True

        # agua
        if self.board[row][col] == 0:
            return {"result": "agua", "extra_turn": False}

        # si había barco, marcamos y vemos si se ha hundido
        # encontrar a qué barco pertenece
        for ship in self.ships:
            if (row, col) in ship:
                # ¿todas las casillas de ese barco están disparadas?
                sunk = all(self.shots[r][c] for (r, c) in ship)
                if sunk:
                    # ¿hemos hundido todos?
                    if self.all_sunk():
                        return {"result": "victoria", "extra_turn": True}
                    else:
                        return {"result": "hundido", "extra_turn": True}
                else:
                    return {"result": "tocado", "extra_turn": True}

        # no debería llegar aquí
        return {"result": "error", "extra_turn": False}

    def all_sunk(self):
        for ship in self.ships:
            for (r, c) in ship:
                if not self.shots[r][c]:
                    return False
        return True

    # ------------------------------------------------------------------
    # DEBUG
    # ------------------------------------------------------------------
    def print_board(self, reveal=False):
        """
        Si reveal=True muestra dónde están los barcos
        Si False, muestra sólo disparos
        """
        for r in range(BOARD_SIZE):
            row_str = []
            for c in range(BOARD_SIZE):
                if reveal:
                    if self.board[r][c] == 1:
                        ch = "B"
                    else:
                        ch = "."
                else:
                    if self.shots[r][c]:
                        ch = "x"
                    else:
                        ch = "."
                row_str.append(ch)
            print(" ".join(row_str))
        print()
