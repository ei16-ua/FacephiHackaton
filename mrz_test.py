import re
from datetime import datetime
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

# Config, indicar el dia de hoy aquí
T, MODEL = datetime(2026, 2, 23), ocr_predictor(pretrained=True)

# Función para extraer texto, c es el umbral de confianza (0.5 es un buen valor)
def get_txt(img, c=0.5):
    res = MODEL(DocumentFile.from_images(img))
    return [" ".join([w.value for w in l.words if w.confidence > c]) for p in res.pages for b in p.blocks for l in b.lines]
#en el back_text he puesto 0.4 ya que hay veces que no pilla correctamente la información del MRZ
def compare():
    print("Analizando...")
    f_lines, b_lines = get_txt('dni_front_especimen.jpg', 0.5), get_txt('dni_back_especimen.jpg', 0.4)
    f_all = " ".join(f_lines).upper()
    
    # Identificar líneas MRZ( son 3 líneas)
    m_c = [l.replace(" ", "") for l in b_lines if len(l.replace(" ", "")) >= 28 and ("<" in l or "ESP" in l)]
    if len(m_c) < 3: return print("Error: MRZ no detectado")
    
    # Encontrar L1 (buscando IDESP o ID) Ejemplo: IDESPCA00000049999999R<<<<<
    l1 = next((l for l in m_c if "ESP" in l[:10]), m_c[0])

    idx = m_c.index(l1)
    l2 = m_c[idx+1] if len(m_c) > idx+1 else m_c[1]#si no encuentra la segunda línea, coge la segunda
    l3 = m_c[idx+2] if len(m_c) > idx+2 else m_c[-1]#si no encuentra la tercera línea, coge la tercera
    
    m_data = {
        "id": l1[5:14].replace("<",""), "dni": l1[15:23].replace("<",""), #el id y el dni están en la primera línea
        "s": l2[7], "b": f"{l2[4:6]} {l2[2:4]} 20{l2[0:2]}", "e": f"{l2[12:14]} {l2[10:12]} 20{l2[8:10]}", #la fecha de nacimiento y la fecha de caducidad están en la segunda línea
        "n": l3.replace("<", " ").strip() #el nombre está en la tercera línea
    }
    
    dates = re.findall(r'\d{2} \d{2} \d{4}', f_all)#encuentra las fechas en el frontal
    iss = next((d for d in dates if d != m_data["b"] and d != m_data["e"]), "N/A")#encuentra la fecha de emisión
    
    def vd(s, fut):#función para validar fechas
        try: 
            d = datetime.strptime(s.replace(" ",""), "%d%m%Y")#convierte la fecha a formato datetime
            return (d > T) == fut#compara la fecha con la fecha actual
        except: return False

    print(f"\n{'CAMPO':<10} | {'FRONTAL':<12} | {'MRZ':<12} | OK")#imprime la cabecera
    print("-" * 50)#imprime una línea separadora
    dni_f = (re.search(r'\d{8}[A-Z]', f_all) or re.search(r'\d{8}', f_all)).group(0) if re.search(r'\d{8}', f_all) else "N/A"#busca el dni en el frontal
    id_f = re.search(r'[A-Z]{3}\d{6}', f_all).group(0) if re.search(r'[A-Z]{3}\d{6}', f_all) else "N/A"#busca el id en el frontal
    
    for k, name, fv in [("dni", "DNI", dni_f[:8]), ("id", "IDESP", id_f), #compara el dni y el id
    ("s", "SEXO", m_data["s"] if m_data["s"] in f_all else "N/A"), #compara el sexo
                        ("b", "BIRTH", m_data["b"] if m_data["b"] in dates else "N/A"), #compara la fecha de nacimiento
                        ("e", "EXP", m_data["e"] if m_data["e"] in dates else "N/A")]: #compara la fecha de caducidad
        print(f"{name:<10} | {fv:<12} | {m_data[k]:<12} | {'Y' if fv==m_data[k] else 'N'}")#compara los datos del frontal con los del mrz
        
    name_ok = any(x in f_all for x in m_data['n'].replace("K"," ").split() if len(x)>3)#comprueba si el nombre está en el frontal, en el caso de las K, las omite para que no de error
    print(f"NOMBRE     | {'OK' if name_ok else 'VERIFICAR':<12} | {m_data['n']:<12} | {'Y' if name_ok else 'N'}") #comprueba si el nombre está en el frontal
    print(f"\nEmisión < hoy ({iss}): {'Y' if vd(iss, 0) else 'N'}\nCaducidad > hoy ({m_data['e']}): {'Y' if vd(m_data['e'], 1) else 'N'}") #comprueba si la fecha de emisión y caducidad son correctas

if __name__ == "__main__": compare()
