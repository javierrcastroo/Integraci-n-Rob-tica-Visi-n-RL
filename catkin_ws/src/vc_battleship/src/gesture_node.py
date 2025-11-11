#!/usr/bin/env python
import rospy
import cv2
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from collections import deque

# IMPORTA TUS MÓDULOS
# Ajusta el prefijo según tu paquete. La idea es esta:
from vc_battleship.segmentation import segment_hand_mask
from vc_battleship.features import compute_feature_vector
from vc_battleship.classifier import knn_predict
from vc_battleship.storage import load_gesture_gallery
from vc_battleship.hand_config import PREVIEW_W, PREVIEW_H, RECOGNIZE_MODE, CONFIDENCE_THRESHOLD


def majority_vote(labels):
    if not labels:
        return None
    return max(set(labels), key=labels.count)


def label_to_number(label):
    """
    Convierte el label del KNN a número.
    Cambia aquí si tus labels son distintos.
    """
    mapping = {
        "0dedos": 0,
        "1dedo": 1,
        "2dedos": 2,
        "3dedos": 3,
        "4dedos": 4,
        "5dedos": 5,
    }
    return mapping.get(label, None)


def is_ok_gesture(label):
    if not label:
        return False
    label = label.lower()
    return label in ("ok", "p", "confirm")


class GestureNode(object):
    def __init__(self):
        rospy.init_node('gesture_node', anonymous=True)

        self.bridge = CvBridge()
        self.last_img = None

        # publicador para la lista de ataques
        self.pub_attacks = rospy.Publisher('/gesture/attack_list', String, queue_size=10)

        # nos suscribimos a la webcam
        rospy.Subscriber('/webcam/image_raw', Image, self.cb_image)

        # cargamos galería
        self.gallery = load_gesture_gallery() if RECOGNIZE_MODE else []

        # buffer de predicciones para voto mayoritario
        self.recent_preds = deque(maxlen=7)

        # lógica de estabilidad
        self.STABLE_FRAMES = 60
        self.last_stable_label = None
        self.stable_counter = 0

        # buffers de ataques
        self.current_attack = []   # p.ej. [2, 4]
        self.attack_list = []      # lista de ataques [(2,4), (1,5), ...]

    def cb_image(self, msg):
        self.last_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def process_frame(self, frame_bgr):
        """
        Aquí va tu pipeline de mano resumido: segmentar -> features -> knn
        Devuelve el label o None.
        """
        # espejo + resize como en tu código original
        frame = cv2.flip(frame_bgr, 1)
        frame = cv2.resize(frame, (PREVIEW_W, PREVIEW_H))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # de momento sin ROI: segmentamos con lo que tengas en segment_hand_mask
        mask = segment_hand_mask(hsv, lower=None, upper=None)

        feat = compute_feature_vector(mask)
        if feat is None or not self.gallery:
            return None

        label, dist = knn_predict(feat, self.gallery, k=5)
        if dist is not None and dist > CONFIDENCE_THRESHOLD:
            return "????"
        return label

    def run(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            if self.last_img is not None:
                label = self.process_frame(self.last_img)

                # voto mayoritario
                if label is not None:
                    self.recent_preds.append(label)
                stable_label = majority_vote(list(self.recent_preds))

                # lógica de estabilidad sobre el gesto estable
                if stable_label is not None and stable_label != "????":
                    if stable_label == self.last_stable_label:
                        self.stable_counter += 1
                    else:
                        self.last_stable_label = stable_label
                        self.stable_counter = 1

                    if self.stable_counter >= self.STABLE_FRAMES:
                        # 1) ¿es OK?
                        if is_ok_gesture(stable_label):
                            if len(self.current_attack) == 2:
                                self.attack_list.append(tuple(self.current_attack))
                                rospy.loginfo(f"[GESTURE] Ataque guardado: {tuple(self.current_attack)}")

                                # publicar TODA la lista para el game_logic
                                msg = String()
                                msg.data = str(self.attack_list)
                                self.pub_attacks.publish(msg)
                                rospy.loginfo(f"[GESTURE] Lista enviada: {msg.data}")

                                # vaciamos solo el ataque actual
                                self.current_attack = []
                                # si quieres vaciar también la lista total, descomenta:
                                # self.attack_list = []
                            else:
                                rospy.logwarn("[GESTURE] OK detectado pero no había 2 números.")
                        else:
                            # 2) si no es OK, esperamos que sea un número de dedos
                            num = label_to_number(stable_label)
                            if num is not None:
                                if len(self.current_attack) < 2:
                                    self.current_attack.append(num)
                                    rospy.loginfo(f"[GESTURE] Añadido número {num} → {self.current_attack}")
                                else:
                                    rospy.loginfo(f"[GESTURE] Ya había 2 números {self.current_attack}, ignorando {num}")
                            else:
                                rospy.loginfo(f"[GESTURE] Gesto estable '{stable_label}' pero no es número ni OK.")

                        # reseteamos el contador para no repetir
                        self.stable_counter = 0
                else:
                    self.last_stable_label = None
                    self.stable_counter = 0

            rate.sleep()


if __name__ == '__main__':
    node = GestureNode()
    node.run()
