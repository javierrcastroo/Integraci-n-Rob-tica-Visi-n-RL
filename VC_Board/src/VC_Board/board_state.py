# board_state.py
"""
Gestión del estado de cada tablero + origen global compartido
"""

def init_board_state(name):
    """
    Crea un diccionario con el estado de un tablero concreto (T1, T2, ...).
    """
    return {
        "name": name,
        "last_quad": None,
        "miss": 0,
        "ship_two_cells": [],
        "ship_one_cells": [],
    }


# origen global = el cubo verde que el usuario calibra con 'r'
# lo vamos actualizando cada frame
GLOBAL_ORIGIN = None       # (x, y) en píxeles
GLOBAL_ORIGIN_MISS = 0     # para aguantar unos frames si desaparece
GLOBAL_ORIGIN_MAX_MISS = 10