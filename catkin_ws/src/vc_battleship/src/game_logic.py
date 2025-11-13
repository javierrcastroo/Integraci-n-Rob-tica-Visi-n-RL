from collections import defaultdict

from game_board_validation import validate_layout

BOARD_SIZE = 5


class GameLogic:
    """Valida un tablero detectado por visión y gestiona los disparos."""

    REQUIRED_SHIPS = {"ship2": 1, "ship1": 3}
    SHIP_LENGTH = {"ship2": 2, "ship1": 1}

    def __init__(self):
        self.reset_board()
        self.current_signature = None
        self.ammo_list = []
        self.selected_ammo = None

    # ------------------------------------------------------------------
    # CONFIGURACIÓN DEL TABLERO
    # ------------------------------------------------------------------
    def reset_board(self):
        self.board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        self.shots = [[False] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        self.ship_defs = []  # lista de dicts {id, type, cells}
        self.ship_map = {}
        self.cell_to_coords = {}
        self.initial_cells = {}
        self.remaining_cells = set()
        self.missing_cells = set()
        self.board_ready = False
        self.validation_msg = "Tablero no configurado"

    def update_board_from_detections(self, payload):
        """Recibe la estructura JSON que publica board_node."""

        ammo_data = None
        if isinstance(payload, dict):
            objects_info = payload.get("objects", [])
            ammo_data = payload.get("ammo")
        else:
            objects_info = payload

        self._update_ammo_state(ammo_data)

        if objects_info is None:
            objects_info = []

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

        signature = frozenset(
            (ship_type, r, c)
            for ship_type, cells in cells_by_type.items()
            for (r, c) in cells
        )

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

        if self.board_ready:
            status.update(self._update_running_board(cells_by_type, coords_by_cell))
            return status

        valid, message, ship_defs = validate_layout(
            cells_by_type,
            self.REQUIRED_SHIPS,
            self.SHIP_LENGTH,
            BOARD_SIZE,
        )

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

    def _update_ammo_state(self, ammo_data):
        if not ammo_data:
            self.ammo_list = []
            self.selected_ammo = None
            return

        cleaned = []
        for entry in ammo_data.get("list") or []:
            try:
                ammo_id = int(entry.get("id"))
            except (TypeError, ValueError):
                continue
            px = entry.get("px")
            py = entry.get("py")
            dx_cm = entry.get("dx_cm")
            dy_cm = entry.get("dy_cm")
            cleaned.append(
                {
                    "id": ammo_id,
                    "px": int(px) if px is not None else None,
                    "py": int(py) if py is not None else None,
                    "dx_cm": float(dx_cm) if dx_cm is not None else None,
                    "dy_cm": float(dy_cm) if dy_cm is not None else None,
                }
            )

        self.ammo_list = cleaned

        selected = None
        raw_selected = ammo_data.get("selected")
        if raw_selected:
            try:
                sel_id = int(raw_selected.get("id"))
            except (TypeError, ValueError):
                sel_id = None
            if sel_id is not None:
                for item in self.ammo_list:
                    if item["id"] == sel_id:
                        selected = dict(item)
                        break

        self.selected_ammo = selected

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
                self.initial_cells[(r, c)] = ship_copy["type"]
                self.remaining_cells.add((r, c))
            ship_id += 1

        self.board_ready = True
        self.validation_msg = "Tablero válido"

    def _update_coords(self, coords_by_cell):
        for key, value in coords_by_cell.items():
            self.cell_to_coords[key] = value

    def _update_running_board(self, cells_by_type, coords_by_cell):
        detected_cells = set()
        for ship_type, cells in cells_by_type.items():
            for cell in cells:
                detected_cells.add(cell)
                if cell not in self.initial_cells:
                    return {
                        "valid": False,
                        "message": "Se detectaron fichas fuera del tablero inicial",
                        "changed": False,
                    }
                if self.initial_cells[cell] != ship_type:
                    return {
                        "valid": False,
                        "message": "El tipo de barco detectado no coincide con el inicial",
                        "changed": False,
                    }

        initial_cells_set = set(self.initial_cells.keys())

        if detected_cells - initial_cells_set:
            return {
                "valid": False,
                "message": "Se detectaron celdas inexistentes",
                "changed": False,
            }

        new_missing = initial_cells_set - detected_cells
        newly_missing = new_missing - self.missing_cells

        for (r, c) in newly_missing:
            if not self.shots[r][c]:
                return {
                    "valid": False,
                    "message": "Faltan fichas que no han sido impactadas",
                    "changed": False,
                }

        self.missing_cells = new_missing
        self.remaining_cells = detected_cells
        self._update_coords(coords_by_cell)

        message = "Tablero válido"
        if self.missing_cells:
            message += f" (impactos: {len(self.missing_cells)})"

        changed = bool(newly_missing)
        self.validation_msg = message

        return {
            "valid": True,
            "message": message,
            "changed": changed,
        }

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
                        "hit": self.shots[r][c],
                    }
                )
            ships_info.append({"type": ship["type"], "cells": cells})
        return ships_info

    def get_cell_coordinates(self, row, col):
        return self.cell_to_coords.get((row, col))

    def get_ammo_state(self):
        return {"list": list(self.ammo_list), "selected": self.selected_ammo}

    def get_selected_ammo(self):
        return self.selected_ammo
