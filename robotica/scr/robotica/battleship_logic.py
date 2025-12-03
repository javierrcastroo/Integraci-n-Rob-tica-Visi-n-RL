"""Validación sencilla de tableros de Hundir la Flota."""

def _cells_adjacent(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1


def evaluate_board(layout):
    """Recibe un layout con listas de celdas ocupadas y devuelve (ok, mensaje)."""
    ship_two_cells = layout.get("ship_two_cells", [])
    ship_one_cells = layout.get("ship_one_cells", [])

    errors = []

    # comprobar cantidades
    if len(ship_two_cells) != 2:
        errors.append("El barco de 2 casillas no está completo")
    if len(ship_one_cells) != 3:
        errors.append("Tiene que haber exactamente 3 barcos de 1 casilla")

    # comprobar duplicados
    all_cells = ship_two_cells + ship_one_cells
    if len(all_cells) != len(set(all_cells)):
        errors.append("Hay celdas solapadas entre barcos")

    # comprobar barco de tamaño 2
    if len(ship_two_cells) == 2:
        (r0, c0), (r1, c1) = ship_two_cells
        dr = abs(r0 - r1)
        dc = abs(c0 - c1)
        if not ((dr == 0 and dc == 1) or (dc == 0 and dr == 1)):
            errors.append("El barco de 2 debe estar recto y ocupar 2 casillas contiguas")

    # comprobar separaciones
    singles = ship_one_cells
    if ship_two_cells:
        for cell in ship_two_cells:
            for other in singles:
                if _cells_adjacent(cell, other):
                    errors.append("Los barcos no pueden tocarse")
                    break
    for idx, cell in enumerate(singles):
        for other in singles[idx + 1:]:
            if _cells_adjacent(cell, other):
                errors.append("Los barcos no pueden tocarse")
                break
        if errors:
            break

    if not errors:
        return True, "Distribución correcta"
    # devolver solo errores únicos para no repetir el mismo texto
    uniq_errors = []
    for err in errors:
        if err not in uniq_errors:
            uniq_errors.append(err)
    return False, " | ".join(uniq_errors)