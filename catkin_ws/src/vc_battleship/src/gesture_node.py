#!/usr/bin/env python
import rospy
import cv2
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from collections import deque

# tus módulos
from segmentation import segment_hand_mask, calibrate_from_roi
from features import compute_feature_vector
from classifier import knn_predict
from storage import load_gesture_gallery
from hand_config import PREVIEW_W, PREVIEW_H, RECOGNIZE_MODE, CONFIDENCE_THRESHOLD
import ui  # <-- para el ROI

def majority_vote(labels):
    if not labels:
        return None
    return max(set(labels), key=labels.count)

def label_to_number(label):
    mapping = {
        "0dedos": 0,
        "1dedo": 1,
        "2dedos": 2,
        "3dedos": 3,
        "4dedos": 4,
        "5dedos": 5,
        # tu gesto extra:
        "cool": 6,
    }
    return mapping.get(label, None)



def is_ok_gesture(label):
    if not label:
        return False
    label = label.lower()
    # añadimos 'cool' como gesto de confirmación
    return label in ("ok", "p", "confirm", "cool")

class GestureNode(object):
    def __init__(self):
        rospy.init_node('gesture_node', anonymous=True)

        self.bridge = CvBridge()
        self.last_img = None

        self.pub_attacks = rospy.Publisher('/gesture/attack_list', String, queue_size=10)
        rospy.Subscriber('/webcam/image_raw', Image, self.cb_image)

        self.gallery = load_gesture_gallery() if RECOGNIZE_MODE else []
        rospy.loginfo(f"Cargadas {len(self.gallery)} muestras en galería.")

        self.recent_preds = deque(maxlen=7)
        self.STABLE_FRAMES = 60
        self.last_stable_label = None
        self.stable_counter = 0

        self.current_attack = []
        self.attack_list = []

        # HSV por defecto (luego lo recalibras con ROI)
        self.lower_skin = (0, 30, 60)
        self.upper_skin = (20, 150, 255)

        # ventana + callback de ratón como en tu hand_main
        cv2.namedWindow("Gesture")
        cv2.setMouseCallback("Gesture", ui.mouse_callback)

    def cb_image(self, msg):
        self.last_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def process_frame(self, frame_bgr):
        # espejo + resize
        frame = cv2.flip(frame_bgr, 1)
        frame = cv2.resize(frame, (PREVIEW_W, PREVIEW_H))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # dibujar ROI en la vista
        vis = frame.copy()
        ui.draw_roi_rectangle(vis)

        # segmentar con el HSV actual
        mask = segment_hand_mask(hsv, self.lower_skin, self.upper_skin)

        # mostrar para que puedas ver si está bien segmentado
        cv2.imshow("Gesture", vis)
        cv2.imshow("Gesture_mask", mask)

        # si no hay galería, no clasificamos (solo mostramos)
        if not self.gallery:
            return None

        feat = compute_feature_vector(mask)
        if feat is None:
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

                # leer teclas (como en tu hand_main)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):
                    break
                if key == ord('c'):
                    # calibrar HSV desde el ROI actual
                    if ui.roi_defined:
                        # ojo: hay que volver a construir el frame igual que en process_frame
                        frame = cv2.flip(self.last_img, 1)
                        frame = cv2.resize(frame, (PREVIEW_W, PREVIEW_H))
                        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                        x0, x1 = sorted([ui.x_start, ui.x_end])
                        y0, y1 = sorted([ui.y_start, ui.y_end])
                        if (x1 - x0) > 5 and (y1 - y0) > 5:
                            roi_hsv = hsv[y0:y1, x0:x1]
                            self.lower_skin, self.upper_skin = calibrate_from_roi(roi_hsv)
                            rospy.loginfo(f"[GESTURE] HSV calibrado: {self.lower_skin} {self.upper_skin}")
                        else:
                            rospy.logwarn("[GESTURE] ROI muy pequeño para calibrar")
                    else:
                        rospy.logwarn("[GESTURE] Dibuja un ROI primero")

                # lógica de estabilidad (igual que antes)
                if label is not None:
                    self.recent_preds.append(label)

                stable_label = majority_vote(list(self.recent_preds))

                if stable_label is not None and stable_label != "????":
                    if stable_label == self.last_stable_label:
                        self.stable_counter += 1
                    else:
                        self.last_stable_label = stable_label
                        self.stable_counter = 1

                    if self.stable_counter >= self.STABLE_FRAMES:
                        if is_ok_gesture(stable_label):
                            if len(self.current_attack) == 2:
                                self.attack_list.append(tuple(self.current_attack))
                                rospy.loginfo(f"[GESTURE] Ataque guardado: {tuple(self.current_attack)}")
                                msg = String()
                                msg.data = str(self.attack_list)
                                self.pub_attacks.publish(msg)
                                rospy.loginfo(f"[GESTURE] Lista enviada: {msg.data}")
                                self.current_attack = []
                            else:
                                rospy.logwarn("[GESTURE] OK detectado pero no había 2 números.")
                        else:
                            num = label_to_number(stable_label)
                            if num is not None:
                                if len(self.current_attack) < 2:
                                    self.current_attack.append(num)
                                    rospy.loginfo(f"[GESTURE] Añadido número {num} → {self.current_attack}")
                                else:
                                    rospy.loginfo(f"[GESTURE] Ya había 2 números {self.current_attack}, ignoro {num}")
                            else:
                                rospy.loginfo(f"[GESTURE] Gesto estable '{stable_label}' pero no es número ni OK.")
                        self.stable_counter = 0
                else:
                    self.last_stable_label = None
                    self.stable_counter = 0

            rate.sleep()

        cv2.destroyAllWindows()

if __name__ == '__main__':
    node = GestureNode()
    node.run()
