import face_recognition
import cv2
import numpy as np
import matplotlib.pyplot as plt


def comparar_facial(camara, imagen_dni):
    img1 = face_recognition.load_image_file(camara)
    img2 = face_recognition.load_image_file(imagen_dni)

    img1_encoding = face_recognition.face_encodings(img1)[0]
    img2_encoding = face_recognition.face_encodings(img2)[0]

    resultado = face_recognition.compare_faces([img1_encoding], img2_encoding)[0]
    distancia = face_recognition.face_distance([img1_encoding], img2_encoding)[0]

    return resultado, distancia

def tomar_foto_instantanea(cam_id=0, output_path='captura.jpg'):
    """Toma una foto inmediata desde la cámara y la guarda en output_path."""
    cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)  # usa DirectShow en Windows
    if not cap.isOpened():
        raise IOError("No se puede abrir la cámara")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise IOError("No se pudo leer un frame de la cámara")
    cv2.imwrite(output_path, frame)
    return output_path


if __name__ == "__main__":
    try:
        camara_path = tomar_foto_instantanea()
        resultado, distancia = comparar_facial(camara_path, 'dni_front_especimen.jpg')
        print(f"¿Coinciden las caras? {'Sí' if resultado else 'No'}")
        print(f"Distancia facial: {distancia:.4f}")
    except Exception as e:
        print(f"Error: {e}")