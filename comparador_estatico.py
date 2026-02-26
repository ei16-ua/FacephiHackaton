import cv2
import numpy as np
import face_recognition

def normalizar_iluminacion(img):
    """
    Normaliza la iluminación de la imagen para reducir el efecto de sombras.
    Usa CLAHE (Contrast Limited Adaptive Histogram Equalization).
    """
    # Convertir a LAB (separa luminosidad del color)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # Aplicar CLAHE solo al canal de luminosidad
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_norm = clahe.apply(l)

    # Reconstruir la imagen
    lab_norm = cv2.merge([l_norm, a, b])
    img_norm = cv2.cvtColor(lab_norm, cv2.COLOR_LAB2RGB)
    return img_norm

def cargar_encoding_dni(dni_path):
    """
    Carga y detecta la cara en una foto del DNI (foto completa).
    Recorta la mitad izquierda para centrarse en la foto.
    """
    img = face_recognition.load_image_file(dni_path)
    img = normalizar_iluminacion(img)  # Reducir efecto de sombras
    h, w = img.shape[:2]

    # En un DNI español, la foto está en la mitad izquierda.
    img_crop = img[:, :w // 2]

    # Lista de intentos progresivos para detectar caras pequeñas
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
            encodings = face_recognition.face_encodings(attempt_img, known_face_locations=face_locs, num_jitters=3)
            if encodings:
                print(f"[+] Cara del DNI detectada con: {desc}")
                return encodings[0]

    raise ValueError(f"No se detectó ningún rostro en la imagen del DNI: {dni_path}")

def cargar_encoding_selfie(foto_path):
    """
    Carga y detecta la cara en una foto normal (selfie).
    """
    img = face_recognition.load_image_file(foto_path)
    img = normalizar_iluminacion(img)  # Reducir efecto de sombras
    face_locs = face_recognition.face_locations(img, model="hog")
    
    if not face_locs:
        # Intento secundario escalando un poco si la cara es pequeña
        img_large = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        face_locs = face_recognition.face_locations(img_large, model="hog")
        if not face_locs:
            raise ValueError(f"No se detectó ningún rostro en la foto selfie: {foto_path}")
        img = img_large

    encodings = face_recognition.face_encodings(img, known_face_locations=face_locs, num_jitters=3)
    if not encodings:
        raise ValueError(f"No se pudo extraer los rasgos faciales de: {foto_path}")
    
    print(f"[+] Cara del Selfie detectada correctamente.")
    return encodings[0]

def comparar_imagenes(dni_path, selfie_path, tolerance=0.45): #tol=0.6 por defecto
    """
    Compara las dos imágenes y devuelve si coinciden y la distancia.
    """
    print(f"--- INICIANDO COMPARACIÓN ESTÁTICA ---")
    print(f"DNI FRONT: {dni_path}")
    print(f"SELFIE   : {selfie_path}\n")

    # 1. Extraer rasgos del DNI
    encoding_dni = cargar_encoding_dni(dni_path)
    
    # 2. Extraer rasgos del Selfie
    encoding_selfie = cargar_encoding_selfie(selfie_path)

    # 3. Comparar
    distancia = face_recognition.face_distance([encoding_dni], encoding_selfie)[0]
    coincide = distancia <= tolerance

    print("\n--- RESULTADO DE LA COMPARACIÓN ---")
    if coincide:
        print("¡IDENTIDAD CONFIRMADA!")
        print(f"Son la misma persona. Nivel de diferencia: {distancia:.4f} (límite: {tolerance})")
    else:
        print("¡IDENTIDAD RECHAZADA!")
        print(f"No parecen la misma persona. Nivel de diferencia: {distancia:.4f} (límite: {tolerance})")

    return coincide, distancia


if __name__ == "__main__":
    # Nombres de los archivos de prueba para la demo
    # Cambia estos nombres por las imágenes que vayas a usar
    IMAGEN_DNI = "dni1.png"       # Foto completa del frontal del DNI
    IMAGEN_SELFIE = "selfie_checo2.jpeg"  # Foto de la cara de la persona actual

    try:
        comparar_imagenes(IMAGEN_DNI, IMAGEN_SELFIE, tolerance=0.6)
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        print("Asegúrate de cambiar IMAGEN_DNI e IMAGEN_SELFIE por las rutas correctas.")
