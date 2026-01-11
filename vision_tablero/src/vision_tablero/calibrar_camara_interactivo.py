# calibrar_camara_interactivo.py
import cv2
import numpy as np
import glob
import os

def capturar(cam_index=0, outdir="calib_images"):
    os.makedirs(outdir, exist_ok=True)
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("[ERROR] no se pudo abrir la cámara")
        return
    print("[INFO] espacio = guardar, q = salir")
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("captura", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord(' '):
            fp = os.path.join(outdir, f"calib_{i:02d}.jpg")
            cv2.imwrite(fp, frame)
            print("[INFO] guardada", fp)
            i += 1
        elif k == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def calibrar(img_dir="calib_images", out_file="camera_params.npz", rows=9, cols=6):
    imgs = []
    for ext in ("*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG"):
        imgs.extend(glob.glob(os.path.join(img_dir, ext)))
    if len(imgs) == 0:
        print("[ERROR] no hay imágenes en", img_dir)
        return
    cb_size = (rows, cols)
    objp = np.zeros((rows*cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)
    objpoints = []
    imgpoints = []
    gray_shape = None
    for fname in imgs:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_shape = gray.shape[::-1]
        ret, corners = cv2.findChessboardCorners(gray, cb_size, None)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,30,0.001))
            imgpoints.append(corners2)
    if len(objpoints) < 3:
        print("[ERROR] pocas imágenes válidas")
        return
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray_shape, None, None
    )
    np.savez(out_file, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
    print("[INFO] guardado", out_file)

if __name__ == "__main__":
    print("1) capturar  2) calibrar")
    op = input("elige [1/2]: ").strip()
    if op == "1":
        cam = input("cámara (0 por defecto): ").strip() or "0"
        outdir = input("carpeta (calib_images): ").strip() or "calib_images"
        capturar(int(cam), outdir)
    else:
        img_dir = input("carpeta de imágenes (calib_images): ").strip() or "calib_images"
        out = input("salida (.npz) (camera_params.npz): ").strip() or "camera_params.npz"
        rows = int(input("esquinas horizontales (9): ").strip() or "9")
        cols = int(input("esquinas verticales (6): ").strip() or "6")
        calibrar(img_dir, out, rows, cols)