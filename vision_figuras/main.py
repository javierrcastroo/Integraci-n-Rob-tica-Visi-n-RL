import cv2
import sys
import proyecto.detector as detector   # Tu archivo de visión
import logica_juego # Tu archivo de juego (clase)

def main():
    # 1. INICIALIZAR JUEGO
    juego = logica_juego.BatallaNavalGame()
    
    # 2. INICIALIZAR CÁMARA
    cap = detector.inicializar_camara()
    
    if not cap.isOpened():
        print("Error al abrir cámara")
        return

    print("--- SISTEMA LISTO ---")
    print("Presiona 'C' para atacar por COLOR")
    print("Presiona 'F' para atacar por FORMA")
    print("Presiona 'Q' para salir")

    running = True
    while running:
        # --- PARTE A: PROCESAR VISIÓN ---
        # Leemos los valores de los trackbars (si la ventana trackbars existe, si no usa valores fijos)
        try:
            sat = cv2.getTrackbarPos('Sat-Thresh', detector.TRACKBAR_WINDOW)
            val = cv2.getTrackbarPos('Val-Thresh', detector.TRACKBAR_WINDOW)
            eps = cv2.getTrackbarPos('Epsilon x1000', detector.TRACKBAR_WINDOW) / 1000.0
            circ = cv2.getTrackbarPos('Circularity x100', detector.TRACKBAR_WINDOW) / 100.0
        except:
            sat, val, eps, circ = 40, 100, 0.040, 0.70 # Valores por defecto

        # Llamamos a la función de un solo frame
        g_shape, g_color, q_shape, q_color = detector.procesar_frame_actual(cap, sat, val, eps, circ)

        # --- PARTE B: CONTROL DE TECLAS (EL GATILLO) ---
        key = cv2.waitKey(1) & 0xFF
        ataque_a_realizar = []
        
        if key == ord('q'):
            running = False
            
        elif key == ord('c'): # ATAQUE POR COLOR
            print("¡Gatillo COLOR apretado!")
            ataque_a_realizar = detector.obtener_indices_ataque(q_color, g_color)
            juego.encolar_ataque(ataque_a_realizar)
            
        elif key == ord('f'): # ATAQUE POR FORMA
            print("¡Gatillo FORMA apretado!")
            ataque_a_realizar = detector.obtener_indices_ataque(q_shape, g_shape)
            juego.encolar_ataque(ataque_a_realizar)

        # --- PARTE C: ACTUALIZAR JUEGO (PYGAME) ---
        # Esto dibuja la ventana azul y mantiene el juego vivo
        juego.actualizar()

        if not juego.manejar_eventos():
            running = False
            
        juego.dibujar()

    # Limpieza al salir
    cap.release()
    cv2.destroyAllWindows()
    juego.cerrar()
    print("Fin del programa.")

if __name__ == "__main__":
    main()