import cv2
import numpy as np

########################################################################
# --- Configuración ---
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
GRID_SIZE = 5         
QUEUE_SIZE = 3     # Para la Zona de Mando 4x1
MIN_AREA_THRESHOLD = 2000 

TRACKBAR_WINDOW = "Ajustes (Trackbars)"

# Función vacía necesaria para que funcionen los trackbars
def nothing(x):
    pass

cv2.namedWindow(TRACKBAR_WINDOW)
cv2.createTrackbar('Sat-Thresh', TRACKBAR_WINDOW, 40, 255, nothing)
cv2.createTrackbar('Val-Thresh', TRACKBAR_WINDOW, 100, 255, nothing)
cv2.createTrackbar('Epsilon x1000', TRACKBAR_WINDOW, 40, 100, nothing)
cv2.createTrackbar('Circularity x100', TRACKBAR_WINDOW, 70, 100, nothing)


# --- CLASIFICADOR DE COLOR (Corregido) ---
def get_color_name(h, s, v):
    
    # 1. Chequeo de NEGRO
    if v < 50:
        return 'Negro'
    
    # 2. Chequeo de BLANCO/GRIS (Aumenté un poco la exigencia de saturación)
    if s < 60: 
        return '???' 

    # 3. Rangos de Tono (H)

    # --- ROJO (Incluye Naranja y Amarillo fuerte) ---
    # Bajamos el límite a 30. El amarillo puro suele ser 30.
    # Si es mayor a 30, ya empieza a ser verdoso.
    if (h <= 40) or (h >= 160):
        return 'Rojo'
    
    # --- VERDE (Alejado del amarillo) ---
    # Antes empezaba en 36. Lo subimos a 45.
    # Esto ignora el color "limón" o "verde amarillento" que confunde a la cámara.
    elif 45 <= h <= 90:
        return 'Verde'
    
    # --- AZUL ---
    elif 95 <= h <= 145:
        return 'Azul'
    
    else:
        return '???' # Aquí caerán los colores ambiguos (H: 33-44)

########################################################################

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def find_all_grids(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    main_grid_warp = None
    queue_grid_warp = None
    found_grids = [] 

    if not contours:
        return frame, None, None 

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for c in contours[:5]:
        area = cv2.contourArea(c)
        
        if area < MIN_AREA_THRESHOLD:
            continue 
        
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)
            corners = approx.reshape(4, 2)
            ordered_corners = order_points(corners.astype("float32"))
            
            (tl, tr, br, bl) = ordered_corners
            
            widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            maxWidth = max(int(widthA), int(widthB))

            heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            maxHeight = max(int(heightA), int(heightB))

            if maxHeight == 0 or maxWidth == 0:
                continue 

            aspect_ratio = maxWidth / float(maxHeight)
            
            if 0.8 < aspect_ratio < 1.2: # Tablero
                target_size = 500
                target_corners = np.array([
                    [0, 0], [target_size - 1, 0],
                    [target_size - 1, target_size - 1], [0, target_size - 1]
                ], dtype="float32")
                matrix = cv2.getPerspectiveTransform(ordered_corners, target_corners)
                warp = cv2.warpPerspective(frame, matrix, (target_size, target_size))
                found_grids.append({'type': 'main', 'warp': warp, 'area': area})

            elif aspect_ratio < 0.7: # Zona de Mando
                std_height = 500
                target_w = int(std_height * aspect_ratio)
                target_h = std_height
                if target_w < 100: target_w = 100 

                target_corners = np.array([
                    [0, 0], [target_w - 1, 0],
                    [target_w - 1, target_h - 1], [0, target_h - 1]
                ], dtype="float32")
                matrix = cv2.getPerspectiveTransform(ordered_corners, target_corners)
                warp = cv2.warpPerspective(frame, matrix, (target_w, target_h))
                found_grids.append({'type': 'queue', 'warp': warp, 'area': area})
            
    main_candidates = [g for g in found_grids if g['type'] == 'main']
    if main_candidates:
        main_grid_warp = max(main_candidates, key=lambda x: x['area'])['warp']

    queue_candidates = [g for g in found_grids if g['type'] == 'queue']
    if queue_candidates:
        queue_grid_warp = max(queue_candidates, key=lambda x: x['area'])['warp']

    return frame, main_grid_warp, queue_grid_warp

def detect_shape_in_cell(cell_hsv, sat_thresh, val_thresh, epsilon_factor, circ_thresh):
    
    empty_mask = np.array([], dtype="uint8")
    
    if cell_hsv.size == 0: 
        return '0', '???', "v:0 c:0.0", "h:0 s:0 v:0", empty_mask
        
    cell_s = cell_hsv[:, :, 1]
    cell_v = cell_hsv[:, :, 2]
    avg_saturation = np.mean(cell_s)
    avg_value = np.mean(cell_v)

    combined_mask = np.zeros(cell_s.shape, dtype="uint8")
    shape = '0'
    debug_shape = "v:0 c:0.0"
    color_name = '???'
    debug_color = "h:0 s:0 v:0"
    # if color_name == '???':
                # Si el color no es Rojo, Verde o Azul claro,
                # asumimos que es ruido (sombra o linea negra) y lo borramos.
                # return '0', '???', "IGNORADO", debug_color, empty_mask
    if avg_saturation > sat_thresh or avg_value < val_thresh:
        
        _, s_mask = cv2.threshold(cell_s, sat_thresh, 255, cv2.THRESH_BINARY)
        _, v_mask = cv2.threshold(cell_v, val_thresh, 255, cv2.THRESH_BINARY_INV)
        combined_mask = cv2.bitwise_or(s_mask, v_mask)

        shape = '???'
        num_vertices = 0
        circularity = 0.0

        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            piece_contour = max(contours, key=cv2.contourArea)
            
            if cv2.contourArea(piece_contour) > (combined_mask.size * 0.1):
                
                mask_color = np.zeros(cell_hsv.shape[:2], dtype="uint8")
                cv2.drawContours(mask_color, [piece_contour], -1, 255, -1)
                
                mean_hsv_tuple = cv2.mean(cell_hsv, mask=mask_color)
                
                h_mean = mean_hsv_tuple[0]
                s_mean = mean_hsv_tuple[1]
                v_mean = mean_hsv_tuple[2]
                
                color_name = get_color_name(h_mean, s_mean, v_mean)
                debug_color = f"h:{int(h_mean)} s:{int(s_mean)} v:{int(v_mean)}"
                
                peri = cv2.arcLength(piece_contour, True)
                epsilon = epsilon_factor * peri 
                approx = cv2.approxPolyDP(piece_contour, epsilon, True)
                
                num_vertices = len(approx)
                
                if num_vertices == 3:
                    shape = 'T' # Triángulo
                elif num_vertices == 4:
                    shape = 'S' # Cuadrado
                elif num_vertices > 4:
                    area = cv2.contourArea(piece_contour)
                    if peri > 0:
                        circularity = (4 * np.pi * area) / (peri**2)
                        if circularity > circ_thresh:
                            shape = 'C' # Círculo
                        else:
                            shape = 'St' # Estrella
                
                debug_shape = f"v:{num_vertices} c:{circularity:.2f}"

    return shape, color_name, debug_shape, debug_color, combined_mask

def analyze_grid_state(warped_img, sat_thresh, val_thresh, epsilon_factor, circ_thresh):
    
    hsv_warped = cv2.cvtColor(warped_img, cv2.COLOR_BGR2HSV)
    img_size = hsv_warped.shape[0]
    cell_size = img_size // GRID_SIZE

    debug_mask_grid = np.zeros((img_size, img_size), dtype="uint8")
    
    grid_shape_matrix = np.full((GRID_SIZE, GRID_SIZE), '0', dtype=object)
    grid_color_matrix = np.full((GRID_SIZE, GRID_SIZE), '???', dtype=object)
    debug_shape_matrix = np.full((GRID_SIZE, GRID_SIZE), '', dtype=object)
    debug_color_matrix = np.full((GRID_SIZE, GRID_SIZE), '', dtype=object)

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            
            padding = int(cell_size * 0.05)
            x1 = col * cell_size + padding
            y1 = row * cell_size + padding
            x2 = (col + 1) * cell_size - padding
            y2 = (row + 1) * cell_size - padding
            
            cell_hsv = hsv_warped[y1:y2, x1:x2]
            
            shape, color, dbg_shape, dbg_color, cell_mask = detect_shape_in_cell(
                cell_hsv, sat_thresh, val_thresh, epsilon_factor, circ_thresh
            )

            grid_shape_matrix[row][col] = shape
            grid_color_matrix[row][col] = color
            debug_shape_matrix[row][col] = dbg_shape
            debug_color_matrix[row][col] = dbg_color
            
            if cell_mask.size > 0:
                x1_full, y1_full = col * cell_size, row * cell_size
                x2_full, y2_full = (col + 1) * cell_size, (row + 1) * cell_size
                mask_resized = cv2.resize(cell_mask, (cell_size, cell_size))
                debug_mask_grid[y1_full:y2_full, x1_full:x2_full] = mask_resized

    return grid_shape_matrix, grid_color_matrix, debug_shape_matrix, debug_color_matrix, debug_mask_grid

def analyze_queue_state(warped_img, sat_thresh, val_thresh, epsilon_factor, circ_thresh):
    
    hsv_warped = cv2.cvtColor(warped_img, cv2.COLOR_BGR2HSV)
    img_height = hsv_warped.shape[0]
    img_width = hsv_warped.shape[1]
    cell_height = img_height // QUEUE_SIZE

    debug_mask_grid = np.zeros((img_height, img_width), dtype="uint8")
    
    queue_shape_list = np.full(QUEUE_SIZE, '0', dtype=object)
    queue_color_list = np.full(QUEUE_SIZE, '???', dtype=object)
    debug_shape_list = np.full(QUEUE_SIZE, '', dtype=object)
    debug_color_list = np.full(QUEUE_SIZE, '', dtype=object)

    for row in range(QUEUE_SIZE):
        
        padding_y = int(cell_height * 0.05)
        padding_x = int(img_width * 0.15)
        
        x1 = padding_x
        y1 = row * cell_height + padding_y
        x2 = img_width - padding_x
        y2 = (row + 1) * cell_height - padding_y
        
        cell_hsv = hsv_warped[y1:y2, x1:x2]
        
        shape, color, dbg_shape, dbg_color, cell_mask = detect_shape_in_cell(
            cell_hsv, sat_thresh, val_thresh, epsilon_factor, circ_thresh
        )

        queue_shape_list[row] = shape
        queue_color_list[row] = color
        debug_shape_list[row] = dbg_shape
        debug_color_list[row] = dbg_color
        
        if cell_mask.size > 0:
            x1_full, y1_full = 0, row * cell_height
            x2_full, y2_full = img_width, (row + 1) * cell_height
            mask_resized = cv2.resize(cell_mask, (x2_full-x1_full, y2_full-y1_full))
            debug_mask_grid[y1_full:y2_full, x1_full:x2_full] = mask_resized

    return queue_shape_list, queue_color_list, debug_shape_list, debug_color_list, debug_mask_grid

def draw_queue_state(display_img, shape_list, color_list, dbg_shape_list, dbg_color_list):
    img_height = display_img.shape[0]
    img_width = display_img.shape[1]
    cell_height = img_height // QUEUE_SIZE
    
    for i in range(1, QUEUE_SIZE):
        cv2.line(display_img, (0, i * cell_height), (img_width, i * cell_height), (0, 0, 255), 2)

    for row in range(QUEUE_SIZE):
        shape_text = shape_list[row]
        
        if shape_text != '0':
            color_text = color_list[row]
            dbg_shape_text = dbg_shape_list[row]
            dbg_color_text = dbg_color_list[row]
            
            x_shape = int(img_width * 0.3)
            y_shape = row * cell_height + int(cell_height * 0.4) 
            y_color = row * cell_height + int(cell_height * 0.6) 
            
            x_data = int(img_width * 0.1)
            y_data_shape = row * cell_height + int(cell_height * 0.8) 
            y_data_color = row * cell_height + int(cell_height * 0.9) 

            text_color = (0, 0, 255) # Rojo
            debug_color = (0, 255, 0) # Verde

            cv2.putText(display_img, shape_text, (x_shape, y_shape), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 3)
            cv2.putText(display_img, color_text, (x_shape, y_color), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
            
            cv2.putText(display_img, dbg_shape_text, (x_data, y_data_shape), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, debug_color, 1)
            cv2.putText(display_img, dbg_color_text, (x_data, y_data_color), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, debug_color, 1)
    return display_img

def draw_grid_on_image(image, grid_size):
    img_size = image.shape[0]
    cell_size = img_size // grid_size
    for i in range(1, grid_size):
        cv2.line(image, (i * cell_size, 0), (i * cell_size, img_size), (0, 0, 255), 2)
        cv2.line(image, (0, i * cell_size), (img_size, i * cell_size), (0, 0, 255), 2)
    return image

def draw_grid_state(display_img, shape_matrix, color_matrix, dbg_shape_matrix, dbg_color_matrix):
    img_size = display_img.shape[0]
    cell_size = img_size // GRID_SIZE
    
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            shape_text = shape_matrix[row][col]
            
            if shape_text != '0':
                color_text = color_matrix[row][col]
                dbg_shape_text = dbg_shape_matrix[row][col]
                dbg_color_text = dbg_color_matrix[row][col]

                x_shape = col * cell_size + int(cell_size * 0.3)
                y_shape = row * cell_size + int(cell_size * 0.4)
                y_color = row * cell_size + int(cell_size * 0.6)
                
                x_data = col * cell_size + int(cell_size * 0.1)
                y_data_shape = row * cell_size + int(cell_size * 0.8)
                y_data_color = row * cell_size + int(cell_size * 0.9)

                text_color = (0, 0, 255) # Rojo
                debug_color = (0, 255, 0) # Verde

                cv2.putText(display_img, shape_text, (x_shape, y_shape), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 3)
                cv2.putText(display_img, color_text, (x_shape, y_color), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
                
                cv2.putText(display_img, dbg_shape_text, (x_data, y_data_shape), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, debug_color, 1)
                cv2.putText(display_img, dbg_color_text, (x_data, y_data_color), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, debug_color, 1)
    return display_img

# --- NUEVA LÓGICA DE ATAQUE ---

def map_coords(row, col):
    """
    Mapea índices de matriz (0..4) a coordenadas del papel.
    
    TU PAPEL FÍSICO:
    - Filas (Eje Y, de arriba a abajo): A, B, C, D, E
    - Columnas (Eje X, de izq a der): 1, 2, 3, 4, 5
    """
    
    # Nombres de las FILAS (índice 'row')
    row_names = ['A', 'B', 'C', 'D', 'E']
    
    # Nombres de las COLUMNAS (índice 'col')
    col_names = ['1', '2', '3', '4', '5']

    if 0 <= row < 5 and 0 <= col < 5:
        # Devolvemos primero la Letra (Fila) y luego el Número (Columna)
        return f"{row_names[row]}{col_names[col]}"
        
    return "???"

def find_attack_sequence(command_list, grid_matrix):
    """
    Compara la lista de mando con la matriz del tablero y devuelve
    la secuencia de ataque, consumiendo piezas para no repetirlas.
    """
    attack_log = []
    
    # Crear una copia del tablero para "consumir" piezas
    board_copy = np.copy(grid_matrix)
    
    for item_to_find in command_list:
        # Ignorar ranuras vacías en la zona de mando
        if item_to_find == '0' or item_to_find == '???':
            continue
        
        found_in_grid = False
        
        # Buscar en el tablero (de arriba a abajo, izq a der)
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                grid_item = board_copy[r][c]
                
                if grid_item == item_to_find:
                    coords = map_coords(r, c)
                    attack_log.append(f"Orden: '{item_to_find}', Blanco: {coords}")
                    
                    # Marcar pieza como "usada" para que no se repita
                    board_copy[r][c] = 'USED' 
                    
                    found_in_grid = True
                    break # Salir del bucle de columnas
            if found_in_grid:
                break # Salir del bucle de filas
        
        if not found_in_grid:
            attack_log.append(f"Orden: '{item_to_find}', Blanco: ¡NO ENCONTRADO!")
    
    return attack_log

# # --- Bucle Principal (MODIFICADO) ---
# print("Iniciando cámara...")
# print("Controles:")
# print("  q: Salir")
# print("  espacio: Imprimir estado de debug")
# print("  c: Iniciar ataque por COLOR")
# print("  f: Iniciar ataque por FORMA")

# cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW) 

# if not cap.isOpened():
#     print("Error: No se pudo abrir la cámara.")
#     exit()

# cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# cv2.namedWindow("Tablero Corregido", cv2.WINDOW_NORMAL)
# cv2.namedWindow("Camara en Vivo", cv2.WINDOW_NORMAL)
# cv2.namedWindow("Debug de Mascaras", cv2.WINDOW_NORMAL)
# cv2.namedWindow("Zona de Mando", cv2.WINDOW_NORMAL) 

# # Variables para guardar el estado
# current_grid_shape = None
# current_grid_color = None
# current_grid_dbg_shape = None
# current_grid_dbg_color = None

# current_queue_shape = None
# current_queue_color = None
# current_queue_dbg_shape = None
# current_queue_dbg_color = None

# # ... (TODO EL CÓDIGO ANTERIOR DE DETECTOR_2.PY SE MANTIENE) ...
# ... (HASTA LA FUNCIÓN find_attack_sequence) ...

# --- AÑADE O REEMPLAZA ESTO AL FINAL ---

def inicializar_camara():
    cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    return cap

def procesar_frame_actual(cap, sat, val, eps, circ):
    ret, frame = cap.read()
    if not ret: return None, None, None, None, None
    
    # Hacemos una copia limpia para dibujar encima
    display_frame = frame.copy()
    h_frame, w_frame, _ = display_frame.shape # Normalmente 480, 640
    
    processed_frame, main_grid_warp, queue_warp = find_all_grids(frame.copy())
    
    curr_grid_shape = None
    curr_grid_color = None
    curr_queue_shape = None
    curr_queue_color = None
    
    # --- 1. Procesar TABLERO (Derecha) ---
    if main_grid_warp is not None:
        curr_grid_shape, curr_grid_color, dbg_s, dbg_c, _ = analyze_grid_state(
            main_grid_warp, sat, val, eps, circ
        )
        
        tablero_vis = main_grid_warp.copy()
        tablero_vis = draw_grid_on_image(tablero_vis, GRID_SIZE)
        tablero_vis = draw_grid_state(tablero_vis, curr_grid_shape, curr_grid_color, dbg_s, dbg_c)
        
        # Reducimos tamaño a 200x200
        mini_tablero = cv2.resize(tablero_vis, (200, 200))
        
        # Pegar con seguridad (evitando salirnos de la imagen)
        y_end = min(200, h_frame)
        x_start = max(0, w_frame-200)
        
        display_frame[0:y_end, x_start:w_frame] = mini_tablero[0:y_end, 0:(w_frame-x_start)]
        
        cv2.rectangle(display_frame, (x_start, 0), (w_frame, y_end), (0, 255, 0), 2)
        cv2.putText(display_frame, "TABLERO", (w_frame-195, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    
    # --- 2. Procesar ZONA MANDO (Izquierda) ---
    if queue_warp is not None:
        curr_queue_shape, curr_queue_color, dbg_s, dbg_c, _ = analyze_queue_state(
            queue_warp, sat, val, eps, circ
        )
        
        mando_vis = queue_warp.copy()
        mando_vis = draw_queue_state(mando_vis, curr_queue_shape, curr_queue_color, dbg_s, dbg_c)
        
        # Reducimos tamaño (ancho fijo 100px)
        h_m, w_m, _ = mando_vis.shape
        if w_m > 0:
            ratio = 100 / w_m
            new_h = int(h_m * ratio)
            
            # --- FIX: EVITAR QUE SEA MÁS ALTO QUE LA PANTALLA ---
            if new_h > h_frame:
                new_h = h_frame # Lo limitamos a 480 (o la altura que tenga)
            
            mini_mando = cv2.resize(mando_vis, (100, new_h))
            
            # Pegar
            display_frame[0:new_h, 0:100] = mini_mando
            
            cv2.rectangle(display_frame, (0, 0), (100, new_h), (0, 255, 0), 2)
            cv2.putText(display_frame, "MANDO", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    cv2.imshow("SISTEMA DE VISION (Todo en Uno)", display_frame)
    
    return curr_grid_shape, curr_grid_color, curr_queue_shape, curr_queue_color

def obtener_indices_ataque(command_list, grid_matrix):
    """Versión limpia que devuelve solo coordenadas numéricas (fila, col)"""
    indices = []
    if command_list is None or grid_matrix is None: return []
    
    board_copy = np.copy(grid_matrix)
    for item in command_list:
        if item in ['0', '???']: continue
        found = False
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if board_copy[r][c] == item:
                    indices.append((r, c))
                    board_copy[r][c] = 'USED'
                    found = True
                    break
            if found: break
    return indices
# while True:
    
#     sat_thresh = cv2.getTrackbarPos('Sat-Thresh', TRACKBAR_WINDOW)
#     val_thresh = cv2.getTrackbarPos('Val-Thresh', TRACKBAR_WINDOW)
#     epsilon_val = cv2.getTrackbarPos('Epsilon x1000', TRACKBAR_WINDOW)
#     epsilon_factor = epsilon_val / 1000.0
#     circularity_val = cv2.getTrackbarPos('Circularity x100', TRACKBAR_WINDOW)
#     circ_thresh = circularity_val / 100.0

#     ret, frame = cap.read()
#     if not ret: 
#         print("Error: No se pudo leer el fotograma. Saliendo.")
#         break
        
#     processed_frame, main_grid_warp, queue_warp = find_all_grids(frame.copy())
    
#     current_grid_shape = None
#     current_queue_shape = None
    
#     # --- Procesar el Tablero Principal ---
#     if main_grid_warp is not None:
#         current_grid_shape, current_grid_color, current_grid_dbg_shape, \
#         current_grid_dbg_color, debug_mask_image = analyze_grid_state(
#             main_grid_warp, sat_thresh, val_thresh, epsilon_factor, circ_thresh
#         )
        
#         warped_display = main_grid_warp.copy()
#         warped_display = draw_grid_on_image(warped_display, GRID_SIZE)
#         warped_display = draw_grid_state(warped_display, current_grid_shape, 
#                                          current_grid_color, current_grid_dbg_shape, 
#                                          current_grid_dbg_color) 
        
#         cv2.imshow("Tablero Corregido", warped_display)
#         cv2.imshow("Debug de Mascaras", debug_mask_image)
#     else:
#         cv2.imshow("Tablero Corregido", np.zeros((300, 300, 3), dtype="uint8"))
#         cv2.imshow("Debug de Mascaras", np.zeros((300, 300), dtype="uint8"))

#     # --- Procesar la Zona de Mando ---
#     if queue_warp is not None:
#         current_queue_shape, current_queue_color, current_queue_dbg_shape, \
#         current_queue_dbg_color, queue_mask_image = analyze_queue_state(
#             queue_warp, sat_thresh, val_thresh, epsilon_factor, circ_thresh
#         )
        
#         queue_display = queue_warp.copy()
#         queue_display = draw_queue_state(queue_display, current_queue_shape,
#                                          current_queue_color, current_queue_dbg_shape,
#                                          current_queue_dbg_color)
        
#         cv2.imshow("Zona de Mando", queue_display)
#     else:
#         cv2.imshow("Zona de Mando", np.zeros((300, 200, 3), dtype="uint8"))
        

#     cv2.imshow("Camara en Vivo", processed_frame)
    
#     # --- MANEJO DE TECLAS (MODIFICADO) ---
#     key = cv2.waitKey(1) & 0xFF

#     if key == ord('q'): break
    
#     # Presiona 'espacio' para volcar el estado actual (Debug)
#     if key == ord(' '):
#         print("\n=====================================")
#         print(f"(Valores: S:{sat_thresh}, V:{val_thresh}, E:{epsilon_factor:.3f}, C:{circ_thresh:.2f})")
        
#         if current_grid_shape is not None:
#             print(f"\n--- [TABLERO: FORMAS] ---")
#             for row in current_grid_shape: print(row)
            
#             print(f"\n--- [TABLERO: COLOR] ---")
#             for row in current_grid_color: print(row)
            
#             print(f"\n--- [TABLERO: DEBUG FORMA] ---")
#             for row in current_grid_dbg_shape: print(row)
            
#             print(f"\n--- [TABLERO: DEBUG COLOR] ---")
#             for row in current_grid_dbg_color: print(row)
#         else:
#             print("\n[!] Tablero principal no detectado.")

#         if current_queue_shape is not None:
#             print(f"\n--- [ZONA DE MANDO: FORMAS] ---")
#             print(current_queue_shape)
            
#             print(f"\n--- [ZONA DE MANDO: COLOR] ---")
#             print(current_queue_color)
            
#             print(f"\n--- [ZONA DE MANDO: DEBUG FORMA] ---")
#             print(current_queue_dbg_shape)

#             print(f"\n--- [ZONA DE MANDO: DEBUG COLOR] ---")
#             print(current_queue_dbg_color)
#         else:
#             print("\n[!] Zona de Mando no detectada.")
            
#         print("=====================================\n")

#     # Presiona 'c' para ataque por COLOR
#     if key == ord('c'):
#         print("\n=====================================")
#         print("INICIANDO ATAQUE POR COLOR...")
#         if current_queue_color is not None and current_grid_color is not None:
#             ataques = find_attack_sequence(current_queue_color, current_grid_color)
#             for i, ataque in enumerate(ataques):
#                 print(f"Paso {i+1}: {ataque}")
#         else:
#             print("[!] Tablero o Zona de Mando no detectados para el ataque.")
#         print("=====================================\n")

#     # Presiona 'f' para ataque por FORMA (Shape)
#     if key == ord('f'):
#         print("\n=====================================")
#         print("INICIANDO ATAQUE POR FORMA...")
#         if current_queue_shape is not None and current_grid_shape is not None:
#             ataques = find_attack_sequence(current_queue_shape, current_grid_shape)
#             for i, ataque in enumerate(ataques):
#                 print(f"Paso {i+1}: {ataque}")
#         else:
#             print("[!] Tablero o Zona de Mando no detectados para el ataque.")
#         print("=====================================\n")

# cap.release()
# cv2.destroyAllWindows()
# print("Script finalizado.")