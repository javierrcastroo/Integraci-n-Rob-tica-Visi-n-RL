import rospy
import cv2
import json
import numpy as np
import sys  
import os   

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
# --------------------------------------------

from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# Importamos tu librería de visión existente
import detector

# --- CLASE AUXILIAR: ADAPTADOR DE CÁMARA ---
class RosCameraAdapter:

    def __init__(self):
        self.current_frame = None

    def set_frame(self, frame):
        self.current_frame = frame

    def read(self):
        if self.current_frame is None:
            return False, None
        return True, self.current_frame.copy()

    def isOpened(self):
        return True

    def release(self):
        pass
    
    def set(self, prop, val): pass
    def get(self, prop): return 0
    
    @staticmethod
    def json_default(obj):
        if isinstance(obj, tuple):
            return list(obj)
        raise TypeError


# --- NODO PRINCIPAL ---
def main():
    # 1. INICIALIZACIÓN DEL NODO
    rospy.init_node("figuras_node", anonymous=False)
    
    bridge = CvBridge()
    
    image_topic = "/camara_gestos/usb_cam/image_raw"
    attack_topic = "battleship/attack"
    result_topic = "battleship/attack_result"
    
    attack_pub = rospy.Publisher(attack_topic, String, queue_size=10)
    
    fake_camera = RosCameraAdapter()
    
    # 3. CALLBACKS
    def _cb_image(msg):
        try:
            frame = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            fake_camera.set_frame(frame)
        except CvBridgeError as exc:
            rospy.logwarn(f"[figuras_node] Error CvBridge: {exc}")

    def _cb_attack_result(msg):
        try:
            data = json.loads(msg.data)
        except Exception as exc:
            rospy.logwarn(f"[figuras_node] Error parseando resultado: {exc}")
            return

        cell = data.get("cell", {})
        cell_name = cell.get("gestures", "?")
        result = data.get("result", "unknown")
        message = data.get("message", "")

        rospy.loginfo(
            f"[JUEGO] Resultado en {cell_name}: {result.upper()} - {message}"
        )

    # 4. SUSCRIPCIONES
    rospy.Subscriber(image_topic, Image, _cb_image, queue_size=1)
    rospy.Subscriber(result_topic, String, _cb_attack_result, queue_size=10)
    
    rospy.loginfo(f"[figuras_node] Escuchando cámara en: {image_topic}")
    rospy.loginfo(f"[figuras_node] Publicando ataques en: {attack_topic}")

    # 5. BUCLE PRINCIPAL
    rate = rospy.Rate(10)

    # --- NUEVA LISTA (COLA) DE ATAQUES ---
    cola_ataques = [] 

    print("\n--- SISTEMA DE VISIÓN DE FIGURAS LISTO ---")
    print("Asegúrate de tener el foco en la ventana de imagen.")
    print("  [C] -> Detectar por COLOR y añadir a la cola")
    print("  [F] -> Detectar por FORMA y añadir a la cola")
    print("  [P] -> PROCESAR COLA (Enviar siguiente ataque y borrarlo)")
    print("  [Q] -> Salir\n")

    while not rospy.is_shutdown():
        if fake_camera.current_frame is None:
            rate.sleep()
            continue

        # --- A. LEER TRACKBARS ---
        try:
            sat = cv2.getTrackbarPos('Sat-Thresh', detector.TRACKBAR_WINDOW)
            val = cv2.getTrackbarPos('Val-Thresh', detector.TRACKBAR_WINDOW)
            eps = cv2.getTrackbarPos('Epsilon x1000', detector.TRACKBAR_WINDOW) / 1000.0
            circ = cv2.getTrackbarPos('Circularity x100', detector.TRACKBAR_WINDOW) / 100.0
        except:
            sat, val, eps, circ = 40, 100, 0.040, 0.70

        # --- B. PROCESAMIENTO DE VISIÓN ---
        g_shape, g_color, q_shape, q_color = detector.procesar_frame_actual(
            fake_camera, sat, val, eps, circ
        )
        
        # Dibujar info de la cola en pantalla para feedback visual (Opcional)
        cv2.putText(fake_camera.current_frame, f"Cola: {len(cola_ataques)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Main View", fake_camera.current_frame) # Aseguramos ver el frame

        # --- C. CONTROL DE TECLAS ---
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            rospy.loginfo("Cerrando nodo...")
            break

        elif key == ord('c'):
            # --- CARGAR COLA POR COLOR ---
            coords = detector.obtener_indices_ataque(q_color, g_color)
            if coords:
                # Extend añade todos los elementos de la lista coords a la cola
                cola_ataques.extend(coords)
                rospy.loginfo(f"Añadidos {len(coords)} objetivos de COLOR a la cola. Total pendiente: {len(cola_ataques)}")
            else:
                rospy.logwarn("No se detectaron coordenadas por COLOR.")

        elif key == ord('f'):
            # --- CARGAR COLA POR FORMA ---
            coords = detector.obtener_indices_ataque(q_shape, g_shape)
            if coords:
                cola_ataques.extend(coords)
                rospy.loginfo(f"Añadidos {len(coords)} objetivos de FORMA a la cola. Total pendiente: {len(cola_ataques)}")
            else:
                rospy.logwarn("No se detectaron coordenadas por FORMA.")
        
        elif key == ord('p'):
            # --- DISPARAR SIGUIENTE (POP) ---
            if len(cola_ataques) > 0:
                # Sacamos el primer elemento (índice 0) y lo borramos de la lista
                siguiente_ataque = cola_ataques.pop(0)
                
                ataque_payload = {
                    "gestures": siguiente_ataque
                }
                
                # Publicar
                msg = String()
                msg.data = json.dumps(ataque_payload, default=RosCameraAdapter.json_default)
                attack_pub.publish(msg)
                
                rospy.loginfo(f"DISPARANDO A: {siguiente_ataque} | Restantes en cola: {len(cola_ataques)}")
            else:
                rospy.logwarn("¡La cola está vacía! Usa C o F para buscar objetivos primero.")
                
        elif key == ord('r'):
            # --- VACIAR COLA ---
            cola_ataques = [] 
            rospy.loginfo(f"Cola vaciada")
           

        rate.sleep()
    

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()