import time
import cv2
import numpy as np
import face_recognition



def eye_aspect_ratio(eye_points):
    p2_p6 = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
    p3_p5 = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
    p1_p4 = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))
    if p1_p4 == 0:
        return 0.0
    return (p2_p6 + p3_p5) / (2.0 * p1_p4)


def cargar_encoding_dni(dni_path):
    img = face_recognition.load_image_file(dni_path)
    h, w = img.shape[:2]

    # En un DNI español, la foto está en la mitad izquierda.
    # Recortamos esa zona para facilitar la detección.
    img_crop = img[:, :w // 2]

    # Lista de intentos: (imagen, upsample, descripción)
    intentos = [
        (img, 1, "original"),
        (img_crop, 1, "recorte izquierdo"),
        (cv2.resize(img_crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC), 1, "recorte x2"),
        (cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC), 1, "original x2"),
        (cv2.resize(img_crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC), 1, "recorte x3"),
        (cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC), 2, "original x2 upsample=2"),
    ]

    for attempt_img, upsample, desc in intentos:
        face_locs = face_recognition.face_locations(attempt_img, number_of_times_to_upsample=upsample, model="hog")
        if face_locs:
            encodings = face_recognition.face_encodings(attempt_img, known_face_locations=face_locs, num_jitters=1)
            if encodings:
                print(f"Cara detectada con: {desc}")
                return encodings[0]

    raise ValueError("No se detectó ningún rostro en la imagen del DNI.")


def indice_rostro_principal(face_locations):
    areas = []
    for i, (top, right, bottom, left) in enumerate(face_locations):
        areas.append((i, max(0, right - left) * max(0, bottom - top)))
    return max(areas, key=lambda item: item[1])[0]


def ratio_nariz_cara(landmarks):
    chin = landmarks["chin"]
    nose_tip = landmarks["nose_tip"]
    left_x = chin[0][0]
    right_x = chin[-1][0]
    nose_x = int(np.mean([p[0] for p in nose_tip]))
    width = max(1, right_x - left_x)
    return (nose_x - left_x) / width


def escanear_dni_camara(cam_id=0):
    # Abrir cámara
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise IOError("No se puede abrir la cámara para escanear el DNI.")

    print("INSTRUCCIONES: Pon la foto de tu DNI frente a la cámara y pulsa la tecla 'C'.")
    face_encoding = None

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Disminuir tamaño de la ventana
        frame_mostrar = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        cv2.putText(frame_mostrar, "Pon tu DNI en la camara y pulsa 'C' para escanear", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow("Escaneando DNI", frame_mostrar)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            # Intentar detectar la cara en el frame capturado
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locs = face_recognition.face_locations(rgb, model="hog")
            if face_locs:
                encodings = face_recognition.face_encodings(rgb, known_face_locations=face_locs, num_jitters=1)
                if encodings:
                    face_encoding = encodings[0]
                    print("¡Cara del DNI escaneada con éxito!")
                    break
            print("No se detectó ninguna cara en la foto. Asegúrate de enfocar bien y vuelve a pulsar 'C'.")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    if face_encoding is None:
        raise ValueError("Escaneo del DNI cancelado o no se encontró ninguna cara.")
        
    return face_encoding


def verificar_en_vivo(dni_encoding, cam_id=0, tolerance=0.5, timeout_seg=20):

    # Abrir cámara (primero sin backend, luego DSHOW como fallback)
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise IOError("No se puede abrir la cámara.")

    blink_ok = False
    giro_izq_ok = False
    giro_der_ok = False
    vio_ojos_abiertos = False
    frames_ojos_cerrados = 0
    distancias_validas = []

    inicio = time.time()

    try:
        for _ in range(3):
            cap.read()

        while (time.time() - inicio) < timeout_seg:
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Disminuir tamaño de la ventana (mitad) para mayor velocidad y menor tamaño en pantalla
            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb, model="hog")

            if not face_locations:
                cv2.putText(
                    frame,
                    "No se detecta rostro",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
                cv2.imshow("Verificacion en vivo", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            idx = indice_rostro_principal(face_locations)
            loc = [face_locations[idx]]

            encs = face_recognition.face_encodings(rgb, known_face_locations=loc)
            lms = face_recognition.face_landmarks(rgb, face_locations=loc)

            if not encs or not lms:
                continue

            cam_encoding = encs[0]
            landmarks = lms[0]

            if "left_eye" in landmarks and "right_eye" in landmarks:
                ear_left = eye_aspect_ratio(landmarks["left_eye"])
                ear_right = eye_aspect_ratio(landmarks["right_eye"])
                ear = (ear_left + ear_right) / 2.0

                if ear > 0.23:
                    if vio_ojos_abiertos and frames_ojos_cerrados >= 2:
                        blink_ok = True
                    vio_ojos_abiertos = True
                    frames_ojos_cerrados = 0
                elif ear < 0.19:
                    frames_ojos_cerrados += 1

            if "chin" in landmarks and "nose_tip" in landmarks:
                ratio = ratio_nariz_cara(landmarks)
                if ratio < 0.42:
                    giro_izq_ok = True
                if ratio > 0.58:
                    giro_der_ok = True

            liveness_ok = giro_izq_ok and giro_der_ok

            if liveness_ok:
                distancia = face_recognition.face_distance([dni_encoding], cam_encoding)[0]
                distancias_validas.append(float(distancia))

            cv2.putText(
                frame,
                f"Liveness:{'OK' if liveness_ok else 'PENDIENTE'}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"GiroIzq:{giro_izq_ok} GiroDer:{giro_der_ok}",
                (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                "Gira la cabeza a ambos lados (Q para salir)",
                (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Verificacion en vivo", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if len(distancias_validas) >= 6:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    if not (giro_izq_ok and giro_der_ok):
        raise ValueError("Error: prueba de vida (girar cabeza) no superada.")

    if not distancias_validas:
        raise ValueError("Error: no se pudo obtener comparación válida con el DNI.")

    distancia_final = float(np.median(distancias_validas))
    coincide = distancia_final <= tolerance
    return coincide, distancia_final


if __name__ == "__main__":
    try:
        # PANTALLA 1: Escanear DNI
        print("--- PASO 1: ESCANEAR DNI ---")
        cam_utilizar = 1  # 0 para cámara del portátil, 1 para DroidCam (móvil)
        encoding_capturado = escanear_dni_camara(cam_id=cam_utilizar)

        # PANTALLA 2: Verificación de identidad y liveness
        print("\n--- PASO 2: VERIFICACIÓN EN VIVO ---")
        coincide, distancia = verificar_en_vivo(
            dni_encoding=encoding_capturado,
            cam_id=cam_utilizar,
            tolerance=0.6,
            timeout_seg=120,
        )

        if not coincide:
            raise ValueError(
                f"Error: no son la misma persona (distancia={distancia:.4f})."
            )

        print(f"OK: misma persona y prueba de vida superada (distancia={distancia:.4f}).")
    except Exception as exc:
        print(str(exc))