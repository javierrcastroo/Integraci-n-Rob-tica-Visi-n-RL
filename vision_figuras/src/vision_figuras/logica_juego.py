import pygame
import random
import sys

# --- CONSTANTES ---
AGUA = "~"
BARCO = "O"
TOCADO = "X"
FALLO = "M"
HUNDIDO = "H"
TAMANO_TABLERO = 5

# Colores
COLOR_FONDO = (30, 30, 50)
COLOR_AGUA = (0, 105, 148)
COLOR_BARCO = (160, 160, 160)
COLOR_FALLO = (0, 150, 200)
COLOR_TOCADO = (255, 100, 0)
COLOR_HUNDIDO = (200, 0, 0)
COLOR_ESTRATEGIA = (0, 255, 0)
COLOR_TEXTO = (200, 200, 200)

class Barco:
    def __init__(self, nombre, tamano):
        self.nombre = nombre
        self.tamano = tamano
        self.coordenadas = []
        self.impactos = 0
    def esta_hundido(self): return self.impactos == self.tamano
    def registrar_impacto(self): self.impactos += 1

class Tablero:
    def __init__(self, tamano):
        self.tamano = tamano
        self.grid = [[AGUA for _ in range(tamano)] for _ in range(tamano)]
        self.barcos = []

    def colocar_barco(self, barco, fila_ini, col_ini, orientacion):
        coords = []
        if orientacion == "H":
            if col_ini + barco.tamano > self.tamano: return False
            for i in range(barco.tamano):
                if self.grid[fila_ini][col_ini + i] == BARCO: return False
                coords.append((fila_ini, col_ini + i))
        else:
            if fila_ini + barco.tamano > self.tamano: return False
            for i in range(barco.tamano):
                if self.grid[fila_ini + i][col_ini] == BARCO: return False
                coords.append((fila_ini + i, col_ini))
        
        for f, c in coords: self.grid[f][c] = BARCO
        barco.coordenadas = coords
        self.barcos.append(barco)
        return True

    def colocar_flota_random(self):
        flota = [("Submarino", 3), ("Destructor", 2), ("Lancha", 1)]
        for nombre, tam in flota:
            barco = Barco(nombre, tam)
            while True:
                if self.colocar_barco(barco, random.randint(0,4), random.randint(0,4), random.choice(["H","V"])):
                    break

    def recibir_disparo(self, f, c):
        if self.grid[f][c] == AGUA:
            self.grid[f][c] = FALLO
            return FALLO
        elif self.grid[f][c] == BARCO:
            self.grid[f][c] = TOCADO
            for b in self.barcos:
                if (f,c) in b.coordenadas:
                    b.registrar_impacto()
                    if b.esta_hundido():
                        for bf, bc in b.coordenadas: self.grid[bf][bc] = HUNDIDO
                        return HUNDIDO
            return TOCADO
        return FALLO

    def todos_hundidos(self):
        return all(b.esta_hundido() for b in self.barcos)

# --- CLASE PRINCIPAL (MODO PING-PONG) ---
class BatallaNavalGame:
    def __init__(self):
        pygame.init()
        self.font = pygame.font.SysFont('Arial', 16)
        self.header_font = pygame.font.SysFont('Consolas', 20, bold=True)
        
        # Configuración Ventana
        self.cell_size = 40
        self.margin = 5
        self.grid_px = (TAMANO_TABLERO * (self.cell_size + self.margin)) + self.margin
        self.width = (self.grid_px * 2) + 100
        self.height = self.grid_px + 100
        
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Batalla Naval - Duelo Intercalado")
        
        self.tablero_jugador = Tablero(TAMANO_TABLERO)
        self.tablero_ia = Tablero(TAMANO_TABLERO)
        self.tablero_jugador.colocar_flota_random()
        self.tablero_ia.colocar_flota_random()
        
        self.mensaje = "Cámara Lista. Usa 'C' o 'F'."
        self.estrategia_visual = [] 
        
        # --- VARIABLES DE ESTADO ---
        self.cola_jugador = []
        self.cola_ia = []
        
        self.ultimo_tiempo = 0
        self.delay_animacion = 1000  # 1 segundo entre turno y turno
        self.estado_actual = "ESPERA" # ESPERA, TURNO_JUGADOR, TURNO_IA, FIN

    def manejar_eventos(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return False
        return True

    def encolar_ataque(self, lista_coordenadas):
        """Prepara los cargadores de ambos bandos"""
        if not lista_coordenadas:
            self.mensaje = "¡No detecto fichas válidas!"
            return

        if self.estado_actual == "ESPERA":
            # 1. Cargar disparos del JUGADOR
            self.cola_jugador = list(lista_coordenadas)
            self.estrategia_visual = list(lista_coordenadas)
            
            # 2. Cargar disparos de la IA (mismo número de tiros)
            self.cola_ia = []
            for _ in range(len(lista_coordenadas)):
                while True:
                    f, c = random.randint(0, 4), random.randint(0, 4)
                    # La IA no dispara donde ya disparó
                    if self.tablero_jugador.grid[f][c] not in [FALLO, TOCADO, HUNDIDO] and (f,c) not in self.cola_ia:
                        self.cola_ia.append((f, c))
                        break
            
            # 3. Iniciar el duelo
            self.estado_actual = "TURNO_JUGADOR"
            self.mensaje = "¡INICIANDO DUELO DE ARTILLERÍA!"
            self.ultimo_tiempo = pygame.time.get_ticks()

    def actualizar(self):
        """Gestiona el ritmo del juego (Ping-Pong)"""
        if self.estado_actual in ["ESPERA", "FIN"]:
            return

        tiempo_actual = pygame.time.get_ticks()
        
        # Solo avanzamos si ha pasado el tiempo de espera
        if tiempo_actual - self.ultimo_tiempo > self.delay_animacion:
            
            # --- TURNO JUGADOR ---
            if self.estado_actual == "TURNO_JUGADOR":
                if self.cola_jugador:
                    f, c = self.cola_jugador.pop(0)
                    resultado = self.tablero_ia.recibir_disparo(f, c)
                    
                    if resultado == HUNDIDO: txt = "¡HUNDIDO!"
                    elif resultado == TOCADO: txt = "¡IMPACTO!"
                    else: txt = "Agua..."
                    self.mensaje = f"TÚ disparas en {chr(65+f)}{c+1}: {txt}"
                    
                    # Chequeo Victoria Inmediata
                    if self.tablero_ia.todos_hundidos():
                        self.mensaje = "¡VICTORIA! HAS DESTRUIDO AL ENEMIGO."
                        self.estado_actual = "FIN"
                    else:
                        # Cambio de turno
                        self.estado_actual = "TURNO_IA"
                else:
                    # Si no quedan balas, volvemos a espera
                    self.mensaje = "Ronda finalizada. Prepara nueva estrategia."
                    self.estado_actual = "ESPERA"
                    self.estrategia_visual = []

            # --- TURNO IA ---
            elif self.estado_actual == "TURNO_IA":
                if self.cola_ia:
                    f, c = self.cola_ia.pop(0)
                    resultado = self.tablero_jugador.recibir_disparo(f, c)
                    
                    self.mensaje = f"LA IA dispara en {chr(65+f)}{c+1}..."
                    
                    # Chequeo Derrota Inmediata
                    if self.tablero_jugador.todos_hundidos():
                        self.mensaje = "DERROTA... TE HAN HUNDIDO."
                        self.estado_actual = "FIN"
                    else:
                        # Cambio de turno
                        self.estado_actual = "TURNO_JUGADOR"
                else:
                    # Seguridad (aunque no debería llegar aquí si las listas son iguales)
                    self.estado_actual = "ESPERA"

            self.ultimo_tiempo = tiempo_actual

    def dibujar(self):
        self.screen.fill(COLOR_FONDO)
        
        # Títulos
        self.screen.blit(self.header_font.render("TU FLOTA", True, COLOR_TEXTO), (50, 20))
        self.screen.blit(self.header_font.render("RADAR ENEMIGO", True, COLOR_TEXTO), (self.grid_px + 80, 20))
        
        self.dibujar_grid(self.tablero_jugador, 50, 60, True)
        # Pasamos la estrategia visual para que se vea
        self.dibujar_grid(self.tablero_ia, self.grid_px + 80, 60, False, self.estrategia_visual)
        
        # Mensaje centrado
        msg_color = (255, 255, 0)
        if "VICTORIA" in self.mensaje: msg_color = (0, 255, 0)
        if "DERROTA" in self.mensaje: msg_color = (255, 0, 0)
        
        msg_surf = self.font.render(self.mensaje, True, msg_color)
        self.screen.blit(msg_surf, (self.width//2 - msg_surf.get_width()//2, self.height - 30))
        
        pygame.display.flip()

    def dibujar_grid(self, tablero, ox, oy, mostrar_barcos, highlights=[]):
        for f in range(TAMANO_TABLERO):
            for c in range(TAMANO_TABLERO):
                rect = pygame.Rect(ox + c*45, oy + f*45, 40, 40)
                estado = tablero.grid[f][c]
                color = COLOR_AGUA
                
                if estado == BARCO and mostrar_barcos: color = COLOR_BARCO
                elif estado == FALLO: color = COLOR_FALLO
                elif estado == TOCADO: color = COLOR_TOCADO
                elif estado == HUNDIDO: color = COLOR_HUNDIDO
                
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (0,0,0), rect, 1)
                
                if f == 0: 
                    lbl = self.font.render(str(c+1), True, COLOR_TEXTO)
                    self.screen.blit(lbl, (rect.centerx-5, oy-20))
                if c == 0:
                    lbl = self.font.render(chr(65+f), True, COLOR_TEXTO)
                    self.screen.blit(lbl, (ox-20, rect.centery-10))

        # Dibujamos los puntos verdes de tu estrategia
        # PERO solo dibujamos los que están pendientes de disparar
        if self.estado_actual != "FIN":
            for f, c in highlights:
                # Si esa casilla ya ha sido disparada, no pintamos el punto verde
                if tablero.grid[f][c] == AGUA or tablero.grid[f][c] == BARCO: 
                     cx = ox + c*45 + 20
                     cy = oy + f*45 + 20
                     pygame.draw.circle(self.screen, COLOR_ESTRATEGIA, (cx, cy), 5)

    def cerrar(self):
        pygame.quit()