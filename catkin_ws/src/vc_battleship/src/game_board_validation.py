"""Validación de la disposición de barcos para el juego."""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Set, Tuple

BOARD_SIZE = 5


def validate_layout(
    cells_by_type: Dict[str, Set[Tuple[int, int]]],
    required_ships: Dict[str, int],
    ship_length: Dict[str, int],
    board_size: int = BOARD_SIZE,
) -> Tuple[bool, str, List[Dict]]:
    errors: List[str] = []
    ship_defs: List[Dict] = []

    for ship_type, required_count in required_ships.items():
        cells = cells_by_type.get(ship_type, set())
        expected_cells = required_count * ship_length[ship_type]
        if len(cells) != expected_cells:
            errors.append(
                f"Se esperaban {expected_cells} casillas para {ship_type} y se detectaron {len(cells)}"
            )

    if errors:
        return False, "; ".join(errors), []

    ship2_cells = sorted(cells_by_type.get("ship2", set()))
    if not validate_ship2(ship2_cells):
        return False, "El barco de 2 casillas debe ser recto y contiguo", []
    ship_defs.append({"type": "ship2", "cells": ship2_cells})

    ship1_cells = sorted(cells_by_type.get("ship1", set()))
    for cell in ship1_cells:
        ship_defs.append({"type": "ship1", "cells": [cell]})

    if not validate_separation(ship_defs, board_size):
        return False, "Los barcos no pueden tocarse ni en diagonal", []

    return True, "Tablero válido", ship_defs


def validate_ship2(cells: Sequence[Tuple[int, int]]):
    if len(cells) != 2:
        return False

    rows = {r for r, _ in cells}
    cols = {c for _, c in cells}
    if len(rows) == 1:
        seq = sorted(c for _, c in cells)
    elif len(cols) == 1:
        seq = sorted(r for r, _ in cells)
    else:
        return False

    return seq[1] - seq[0] == 1


def validate_separation(ship_defs: Iterable[Dict], board_size: int = BOARD_SIZE) -> bool:
    cell_to_ship: Dict[Tuple[int, int], int] = {}
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
                    if not (0 <= nr < board_size and 0 <= nc < board_size):
                        continue
                    other = cell_to_ship.get((nr, nc))
                    if other is None:
                        continue
                    if other != idx:
                        return False
    return True
