import cv2
import numpy as np
import face_recognition
import matplotlib
matplotlib.use('Agg')  # Backend sin ventana
import matplotlib.pyplot as plt

DNI_PATH = "dni2.png"
SELFIE_PATH = "selfie_perilla1.jpg"

# Colores para cada grupo de landmarks (BGR para OpenCV)
# Solo ojos, nariz y labios
LANDMARK_COLORES = {
    'nose_bridge':     (0, 165, 255),    # Naranja - Puente nariz (4 puntos)
    'nose_tip':        (0, 0, 255),      # Rojo - Punta nariz (5 puntos)
    'left_eye':        (0, 255, 0),      # Verde - Ojo izquierdo (6 puntos)
    'right_eye':       (0, 255, 0),      # Verde - Ojo derecho (6 puntos)
    'top_lip':         (255, 0, 255),    # Magenta - Labio superior (12 puntos)
    'bottom_lip':      (255, 0, 128),    # Rosa - Labio inferior (12 puntos)
}

LANDMARK_NOMBRES = {
    'nose_bridge':     'Puente Nariz',
    'nose_tip':        'Punta Nariz',
    'left_eye':        'Ojo Izq',
    'right_eye':       'Ojo Der',
    'top_lip':         'Labio Sup',
    'bottom_lip':      'Labio Inf',
}

# Zonas que nos interesan para deteccion y comparacion
ZONAS_INTERES = {'left_eye', 'right_eye', 'nose_bridge', 'nose_tip', 'top_lip', 'bottom_lip'}

def bbox_desde_landmarks(landmarks, img_shape, margen=0.3):
    """Calcula un bounding box basado solo en ojos, nariz y labios.
    Esto evita que caras alargadas/recortadas afecten la comparacion.
    margen: porcentaje extra alrededor de los puntos (0.3 = 30%).
    """
    puntos = []
    for zona in ZONAS_INTERES:
        if zona in landmarks:
            puntos.extend(landmarks[zona])
    
    if not puntos:
        return None
    
    xs = [p[0] for p in puntos]
    ys = [p[1] for p in puntos]
    
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    # Aniadir margen proporcional
    ancho = max_x - min_x
    alto = max_y - min_y
    margen_x = int(ancho * margen)
    margen_y = int(alto * margen)
    
    h, w = img_shape[:2]
    top = max(0, min_y - margen_y)
    bottom = min(h, max_y + margen_y)
    left = max(0, min_x - margen_x)
    right = min(w, max_x + margen_x)
    
    return (top, right, bottom, left)  # formato face_recognition

def normalizar_iluminacion(img):
    """Reduce el efecto de sombras usando CLAHE."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_norm = clahe.apply(l)
    lab_norm = cv2.merge([l_norm, a, b])
    return cv2.cvtColor(lab_norm, cv2.COLOR_LAB2RGB)

def dibujar_landmarks(img_bgr, landmarks_list, nombre):
    """Dibuja los 68 puntos faciales agrupados por zona en la imagen."""
    img_puntos = img_bgr.copy()
    
    for face_idx, landmarks in enumerate(landmarks_list):
        print(f"\n  Landmarks cara {face_idx + 1} (solo ojos, nariz, labios):")
        for zona, puntos in landmarks.items():
            # Solo dibujar las zonas de interes
            if zona not in ZONAS_INTERES:
                continue
            color = LANDMARK_COLORES.get(zona, (255, 255, 255))
            nombre_zona = LANDMARK_NOMBRES.get(zona, zona)
            print(f"    {nombre_zona}: {len(puntos)} puntos")
            
            # Dibujar puntos
            for idx, (x, y) in enumerate(puntos):
                cv2.circle(img_puntos, (x, y), 3, color, -1)
                # Numerar algunos puntos clave
                cv2.putText(img_puntos, str(idx), (x + 4, y - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            
            # Conectar puntos con lineas
            for i in range(len(puntos) - 1):
                cv2.line(img_puntos, puntos[i], puntos[i + 1], color, 1)
            
            # Cerrar contorno para ojos y labios
            if zona in ('left_eye', 'right_eye', 'top_lip', 'bottom_lip'):
                cv2.line(img_puntos, puntos[-1], puntos[0], color, 1)
    
    return img_puntos

def crear_leyenda(img_bgr):
    """Crea una leyenda de colores en la esquina de la imagen."""
    y_offset = 20
    for zona, color in LANDMARK_COLORES.items():
        nombre_zona = LANDMARK_NOMBRES.get(zona, zona)
        cv2.rectangle(img_bgr, (10, y_offset - 12), (25, y_offset + 2), color, -1)
        cv2.putText(img_bgr, nombre_zona, (30, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        y_offset += 20
    return img_bgr

def graficar_encodings(enc_dni, enc_selfie, distancia):
    """Genera graficos comparando los 128 valores del encoding facial."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    x = np.arange(128)
    
    # 1. Encodings superpuestos
    axes[0].bar(x - 0.2, enc_dni, width=0.4, alpha=0.7, label='DNI', color='steelblue')
    axes[0].bar(x + 0.2, enc_selfie, width=0.4, alpha=0.7, label='Selfie', color='coral')
    axes[0].set_title(f'Comparacion de los 128 valores del encoding facial (distancia={distancia:.4f})')
    axes[0].set_xlabel('Dimension del encoding')
    axes[0].set_ylabel('Valor')
    axes[0].legend()
    axes[0].set_xlim(-1, 128)
    
    # 2. Diferencia por dimension
    diferencias = enc_dni - enc_selfie
    colores = ['red' if abs(d) > 0.1 else 'green' for d in diferencias]
    axes[1].bar(x, np.abs(diferencias), color=colores, alpha=0.8)
    axes[1].set_title('Diferencia absoluta por dimension (rojo = diferencia > 0.1)')
    axes[1].set_xlabel('Dimension del encoding')
    axes[1].set_ylabel('|Diferencia|')
    axes[1].axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='Umbral 0.1')
    axes[1].legend()
    axes[1].set_xlim(-1, 128)
    
    # 3. Top 20 dimensiones con mayor diferencia
    indices_top = np.argsort(np.abs(diferencias))[::-1][:20]
    vals_top = np.abs(diferencias[indices_top])
    axes[2].barh(range(20), vals_top, color='tomato', alpha=0.8)
    axes[2].set_yticks(range(20))
    axes[2].set_yticklabels([f'Dim {i}' for i in indices_top])
    axes[2].set_title('Top 20 dimensiones con mayor diferencia')
    axes[2].set_xlabel('|Diferencia|')
    axes[2].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('debug_encodings_comparacion.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Grafico de encodings guardado: debug_encodings_comparacion.png")
    
    # Estadisticas
    print(f"\n=== ESTADISTICAS DEL ENCODING ===")
    print(f"  Distancia euclidea total: {distancia:.4f}")
    print(f"  Diferencia media por dimension: {np.mean(np.abs(diferencias)):.4f}")
    print(f"  Diferencia maxima: {np.max(np.abs(diferencias)):.4f} (dimension {np.argmax(np.abs(diferencias))})")
    print(f"  Diferencia minima: {np.min(np.abs(diferencias)):.4f} (dimension {np.argmin(np.abs(diferencias))})")
    print(f"  Dimensiones con diferencia > 0.1: {np.sum(np.abs(diferencias) > 0.1)} de 128")
    print(f"  Dimensiones con diferencia > 0.15: {np.sum(np.abs(diferencias) > 0.15)} de 128")

def diagnosticar_imagen(path, nombre):
    print(f"\n=== {nombre}: {path} ===")
    img = face_recognition.load_image_file(path)
    img = normalizar_iluminacion(img)  # Reducir efecto de sombras
    h, w = img.shape[:2]
    print(f"Tamano: {w}x{h}")
    
    # Detectar caras
    face_locs = face_recognition.face_locations(img, model="hog")
    print(f"Caras detectadas: {len(face_locs)}")
    
    # Si no detecta, intentar con escalado
    img_dibujar = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    used_img = img
    
    if not face_locs:
        print("No se detecto cara, intentando con escala x2...")
        img2 = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        face_locs = face_recognition.face_locations(img2, model="hog")
        print(f"Caras detectadas (x2): {len(face_locs)}")
        img_dibujar = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
        used_img = img2
    
    for i, (top, right, bottom, left) in enumerate(face_locs):
        print(f"  Cara {i+1}: top={top}, right={right}, bottom={bottom}, left={left}")
        ancho_cara = right - left
        alto_cara = bottom - top
        print(f"  Tamano cara: {ancho_cara}x{alto_cara} px")
        cv2.rectangle(img_dibujar, (left, top), (right, bottom), (0, 255, 0), 3)
        cv2.putText(img_dibujar, f"Cara {i+1}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # --- DETECTAR Y DIBUJAR LANDMARKS (solo ojos, nariz, labios) ---
    landmarks_list = face_recognition.face_landmarks(used_img, face_locations=face_locs)
    custom_locs = []
    if landmarks_list:
        img_dibujar = dibujar_landmarks(img_dibujar, landmarks_list, nombre)
        img_dibujar = crear_leyenda(img_dibujar)
        
        # Calcular bounding box ajustado basado solo en ojos/nariz/labios
        for face_idx, lm in enumerate(landmarks_list):
            bbox = bbox_desde_landmarks(lm, used_img.shape)
            if bbox:
                custom_locs.append(bbox)
                top, right, bottom, left = bbox
                # Dibujar el bbox ajustado en cyan para diferenciarlo
                cv2.rectangle(img_dibujar, (left, top), (right, bottom), (255, 255, 0), 2)
                cv2.putText(img_dibujar, "Zona comparacion", (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                print(f"  Bbox ajustado (ojos-nariz-labios): top={top}, right={right}, bottom={bottom}, left={left}")
    
    # Si no se pudieron calcular bboxes custom, usar los originales
    if not custom_locs:
        custom_locs = face_locs
        print("  (Usando bbox original, no se pudieron calcular landmarks)")
    
    # Guardar imagen con las caras y landmarks marcados
    output_path = f"debug_{nombre}.jpg"
    # Resize para que no sea enorme
    max_dim = 800
    scale = min(max_dim / img_dibujar.shape[1], max_dim / img_dibujar.shape[0], 1.0)
    if scale < 1.0:
        img_dibujar = cv2.resize(img_dibujar, None, fx=scale, fy=scale)
    cv2.imwrite(output_path, img_dibujar)
    print(f"  Imagen guardada con landmarks: {output_path}")
    
    return face_locs, used_img, custom_locs

print("=" * 50)
print("DIAGNOSTICO DE COMPARACION FACIAL CON LANDMARKS")
print("=" * 50)

locs_dni, img_dni, custom_locs_dni = diagnosticar_imagen(DNI_PATH, "DNI")
locs_selfie, img_selfie, custom_locs_selfie = diagnosticar_imagen(SELFIE_PATH, "SELFIE")

# Si ambos tienen caras, comparar usando bbox ajustado (solo ojos/nariz/labios)
if custom_locs_dni and custom_locs_selfie:
    print(f"\n=== GENERANDO ENCODINGS (zona ojos-nariz-labios) ===")
    enc_dni = face_recognition.face_encodings(img_dni, known_face_locations=custom_locs_dni, num_jitters=3)
    enc_selfie = face_recognition.face_encodings(img_selfie, known_face_locations=custom_locs_selfie, num_jitters=3)
    
    if enc_dni and enc_selfie:
        # Comparar todas las combinaciones
        print(f"\n=== COMPARACIONES ===")
        for i, ed in enumerate(enc_dni):
            for j, es in enumerate(enc_selfie):
                dist = face_recognition.face_distance([ed], es)[0]
                print(f"  DNI cara {i+1} vs SELFIE cara {j+1}: distancia = {dist:.4f}")
                
                # Generar grafico comparativo de los encodings
                graficar_encodings(ed, es, dist)

print(f"\n=== ARCHIVOS GENERADOS ===")
print(f"  debug_DNI.jpg           -> Foto DNI con landmarks (ojos, nariz, labios)")
print(f"  debug_SELFIE.jpg        -> Foto Selfie con landmarks (ojos, nariz, labios)")
print(f"  debug_encodings_comparacion.png -> Grafico comparando los 128 valores del encoding")
print(f"\nSolo se usan ojos, nariz y labios para la deteccion y comparacion.")
print(f"El recuadro cyan muestra la zona exacta usada para generar el encoding.")
print(f"Esto evita problemas con caras alargadas o recortadas en la barbilla.")
