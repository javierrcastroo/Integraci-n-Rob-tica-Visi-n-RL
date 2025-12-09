import numpy as np
import cv2

# Colores BGR típicos en CV2
COLOR_BACKGROUND = (40, 40, 40)
COLOR_GRID = (100, 100, 100)

COLOR_UNKNOWN = (70, 70, 70)      # gris oscuro
COLOR_MISS = (255, 150, 0)        # azul claro brillante
COLOR_HIT = (0, 0, 255)           # rojo

CELL_SIZE = 80     # píxeles
MARGIN = 60        # espacio para las letras A B C...

def draw_guess_board(guess_board, last_shot=None, title="Guess Board"):
    """
    guess_board : matriz 5x5 con valores {0,1,2}
    last_shot  : (row, col) o None
    """
    board_size = guess_board.shape[0]
    img_size = MARGIN + board_size * CELL_SIZE + 10

    # Crea imagen fondo
    img = np.full((img_size, img_size, 3), COLOR_BACKGROUND, dtype=np.uint8)

    # Texto del título
    cv2.putText(img, title, (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    # Coordenadas verticales (A,B,C...)
    for r in range(board_size):
        text = chr(ord('A') + r)
        y = MARGIN + r * CELL_SIZE + CELL_SIZE//2 + 10
        cv2.putText(img, text, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    # Coordenadas horizontales (1,2,3...)
    for c in range(board_size):
        text = str(c+1)
        x = MARGIN + c * CELL_SIZE + CELL_SIZE//2 - 10
        cv2.putText(img, text, (x, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    # Dibuja celdas
    for r in range(board_size):
        for c in range(board_size):
            y1 = MARGIN + r * CELL_SIZE
            y2 = y1 + CELL_SIZE
            x1 = MARGIN + c * CELL_SIZE
            x2 = x1 + CELL_SIZE

            cell_value = guess_board[r, c]

            if cell_value == 0:      # desconocido
                color = COLOR_UNKNOWN
            elif cell_value == 1:    # miss
                color = COLOR_MISS
            elif cell_value == 2:    # hit
                color = COLOR_HIT
            else:
                color = (255, 0, 255)  # error (magenta)

            # Se dibuja el rectángulo
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)

            # Cuadrícula
            cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_GRID, 2)

    # Resalta último disparo
    if last_shot is not None:
        r, c = last_shot
        y1 = MARGIN + r * CELL_SIZE
        y2 = y1 + CELL_SIZE
        x1 = MARGIN + c * CELL_SIZE
        x2 = x1 + CELL_SIZE
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,255), 4)  # amarillo

    return img
