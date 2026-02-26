import cv2

print("Buscando cámaras disponibles...")
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            print(f"  Cámara {i}: OK ({w}x{h})")
        else:
            print(f"  Cámara {i}: abierta pero no lee frames")
        cap.release()
    else:
        print(f"  Cámara {i}: no disponible")

print("\nSi no aparece ninguna cámara, revisa:")
print("1. Configuración > Privacidad > Cámara > activar acceso")
print("2. Que no haya otra app usando la cámara")
print("3. Administrador de dispositivos > que la cámara esté habilitada")
