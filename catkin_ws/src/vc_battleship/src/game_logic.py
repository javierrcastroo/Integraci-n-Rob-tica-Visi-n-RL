from collections import defaultdict

BOARD_SIZE = 5


class GameLogic:
    """Valida un tablero detectado por visión y gestiona los disparos."""

    REQUIRED_SHIPS = {"ship3": 1, "ship1": 3}
    SHIP_LENGTH = {"ship3": 3, "ship1": 1}

    def __init__(self):
        self.reset_board()
        self.current_signature = None

    # ------------------------------------------------------------------
    # CONFIGURACIÓN DEL TABLERO
    # ------------------------------------------------------------------
    def reset_board(self):
        self.board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        self.shots = [[False] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        self.ship_defs = []  # lista de dicts {id, type, cells}
        self.ship_map = {}
        self.cell_to_coords = {}
        self.board_ready = False
        self.validation_msg = "Tablero no configurado"

    def update_board_from_detections(self, objects_info):
        """Recibe la lista JSON que publica board_node."""

        cells_by_type = defaultdict(set)
        coords_by_cell = {}

        for info in objects_info:
            ship_type = info.get("ship_type")
            if ship_type not in self.REQUIRED_SHIPS:
                continue

            col = info.get("col")
            row = info.get("row")
            if col is None or row is None:
                continue

            r = int(row) - 1
            c = int(col) - 1
            if not (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE):
                continue

            cells_by_type[ship_type].add((r, c))

            dx = info.get("dx_cm")
            dy = info.get("dy_cm")
            if dx is not None and dy is not None:
                coords_by_cell[(r, c)] = (float(dx), float(dy))

        signature = frozenset((ship_type, r, c) for ship_type, cells in cells_by_type.items() for (r, c) in cells)

        status = {
            "valid": False,
            "message": "No se detectaron barcos",
            "changed": False,
        }

        if not signature:
            self.reset_board()
            self.current_signature = None
            self.validation_msg = status["message"]
            return status

        if signature == self.current_signature and self.board_ready:
            # mismo tablero, solo refrescamos coordenadas físicas
            self._update_coords(coords_by_cell)
            status.update({
                "valid": True,
                "message": self.validation_msg,
                "changed": False,
            })
            return status

        valid, message, ship_defs = self._validate_layout(cells_by_type)

        status["message"] = message

        if not valid:
            self.reset_board()
            self.current_signature = None
            self.validation_msg = message
            return status

        self._build_board(ship_defs, coords_by_cell)
        self.current_signature = signature
        self.validation_msg = message
        status.update({"valid": True, "message": message, "changed": True})
        return status

    def _build_board(self, ship_defs, coords_by_cell):
        self.reset_board()

        ship_id = 1
        for ship in ship_defs:
            ship_copy = {
                "id": ship_id,
                "type": ship["type"],
                "cells": list(ship["cells"]),
            }
            self.ship_defs.append(ship_copy)
            self.ship_map[ship_id] = ship_copy
            for (r, c) in ship_copy["cells"]:
                self.board[r][c] = ship_id
                self.cell_to_coords[(r, c)] = coords_by_cell.get((r, c))
            ship_id += 1

        self.board_ready = True
        self.validation_msg = "Tablero válido"

    def _update_coords(self, coords_by_cell):
        for key, value in coords_by_cell.items():
            self.cell_to_coords[key] = value

    def _validate_layout(self, cells_by_type):
        errors = []

        ship_defs = []

        for ship_type, required_count in self.REQUIRED_SHIPS.items():
            cells = cells_by_type.get(ship_type, set())
            expected_cells = required_count * self.SHIP_LENGTH[ship_type]
            if len(cells) != expected_cells:
                errors.append(
                    f"Se esperaban {expected_cells} casillas para {ship_type} y se detectaron {len(cells)}"
                )

        if errors:
            return False, "; ".join(errors), []

        ship3_cells = sorted(cells_by_type.get("ship3", set()))
        if not self._validate_ship3(ship3_cells):
            return False, "El barco de 3 casillas debe ser recto y contiguo", []
        ship_defs.append({"type": "ship3", "cells": ship3_cells})

        ship1_cells = sorted(cells_by_type.get("ship1", set()))
        for cell in ship1_cells:
            ship_defs.append({"type": "ship1", "cells": [cell]})

        if not self._validate_separation(ship_defs):
            return False, "Los barcos no pueden tocarse ni en diagonal", []

        return True, "Tablero válido", ship_defs

    def _validate_ship3(self, cells):
        if len(cells) != 3:
            return False

        rows = {r for r, _ in cells}
        cols = {c for _, c in cells}
        if len(rows) == 1:
            seq = sorted(c for _, c in cells)
        elif len(cols) == 1:
            seq = sorted(r for r, _ in cells)
        else:
            return False

        return seq[1] - seq[0] == 1 and seq[2] - seq[1] == 1

    def _validate_separation(self, ship_defs):
        cell_to_ship = {}
        for idx, ship in enumerate(ship_defs):
            for cell in ship["cells"]:
                cell_to_ship[cell] = idx

        for idx, ship in enumerate(ship_defs):
            for (r, c) in ship["cells"]:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if not (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE):
                            continue
                        other = cell_to_ship.get((nr, nc))
                        if other is None:
                            continue
                        if other != idx:
                            return False
        return True

    # ------------------------------------------------------------------
    # DISPAROS
    # ------------------------------------------------------------------
    def shoot(self, row, col):
        if not self.board_ready:
            return {"result": "sin_tablero", "extra_turn": False}

        if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
            return {"result": "fuera", "extra_turn": False}

        if self.shots[row][col]:
            return {"result": "repetido", "extra_turn": False}

        self.shots[row][col] = True

        ship_id = self.board[row][col]
        if ship_id == 0:
            return {"result": "agua", "extra_turn": False}

        ship = self.ship_map.get(ship_id)
        if ship is None:
            return {"result": "error", "extra_turn": False}

        sunk = all(self.shots[r][c] for (r, c) in ship["cells"])
        result = {
            "result": "tocado",
            "extra_turn": True,
            "ship_type": ship["type"],
        }

        if sunk:
            if self.all_sunk():
                result["result"] = "victoria"
            else:
                result["result"] = "hundido"

        return result

    def all_sunk(self):
        for ship in self.ship_defs:
            if not all(self.shots[r][c] for (r, c) in ship["cells"]):
                return False
        return True

    # ------------------------------------------------------------------
    # UTILIDADES
    # ------------------------------------------------------------------
    def get_ship_coordinates(self):
        ships_info = []
        for ship in self.ship_defs:
            cells = []
            for (r, c) in ship["cells"]:
                coords = self.cell_to_coords.get((r, c))
                cells.append(
                    {
                        "row": r + 1,
                        "col": c + 1,
                        "dx_cm": coords[0] if coords is not None else None,
                        "dy_cm": coords[1] if coords is not None else None,
                    }
                )
            ships_info.append({"type": ship["type"], "cells": cells})
        return ships_info

    def get_cell_coordinates(self, row, col):
        return self.cell_to_coords.get((row, col))
