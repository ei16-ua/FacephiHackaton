import re
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
from threading import Thread
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

# --- CONFIGURACIÓN Y MODELO ---
T_TODAY = datetime(2026, 2, 23)
MODEL = ocr_predictor(pretrained=True)

class MRZApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Verificador DNI - Facephi OCR")
        self.root.geometry("700x600")
        self.root.configure(bg="#f3f4f6")
        
        self.style = ttk.Style()
        self.style.theme_use("clam")
        
        # Estilos personalizados
        self.style.configure("Header.TLabel", font=("Segoe UI", 18, "bold"), background="#f3f4f6", foreground="#111827")
        self.style.configure("TButton", font=("Segoe UI", 10), padding=10)
        self.style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))
        self.style.configure("Treeview", font=("Segoe UI", 10), rowheight=25)
        
        self.create_widgets()

    def create_widgets(self):
        # Header
        header_frame = tk.Frame(self.root, bg="#f3f4f6", pady=20)
        header_frame.pack(fill="x")
        
        ttk.Label(header_frame, text="Verificación de Datos DNI vs MRZ", style="Header.TLabel").pack()
        
        # Botón de Procesar
        self.btn_process = ttk.Button(header_frame, text="Iniciar Análisis", command=self.start_analysis)
        self.btn_process.pack(pady=10)

        # Status Bar
        self.status_var = tk.StringVar(value="Cargue las imágenes y pulse 'Iniciar Análisis'")
        self.status_label = tk.Label(self.root, textvariable=self.status_var, bg="#e5e7eb", font=("Segoe UI", 9))
        self.status_label.pack(fill="x", side="bottom")

        # Main Table Frame
        table_frame = tk.Frame(self.root, bg="white", padx=20, pady=20)
        table_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        columns = ("campo", "frontal", "mrz", "ok")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings")
        
        self.tree.heading("campo", text="CAMPO")
        self.tree.heading("frontal", text="FRONTAL")
        self.tree.heading("mrz", text="MRZ")
        self.tree.heading("ok", text="VAL")
        
        self.tree.column("campo", width=100)
        self.tree.column("frontal", width=150)
        self.tree.column("mrz", width=250)
        self.tree.column("ok", width=50, anchor="center")
        
        self.tree.pack(fill="both", expand=True)
        
        # Tags para colores
        self.tree.tag_configure("ok_row", background="#d1fae5", foreground="#065f46")
        self.tree.tag_configure("fail_row", background="#fee2e2", foreground="#991b1b")

        # Temporal Validation Frame
        self.temp_frame = tk.LabelFrame(self.root, text="Validación de Vigencia", bg="white", font=("Segoe UI", 10, "bold"), padx=10, pady=10)
        self.temp_frame.pack(fill="x", padx=20, pady=20)
        
        self.issue_label = tk.Label(self.temp_frame, text="Emisión anterior a hoy: ---", bg="white", font=("Segoe UI", 10))
        self.issue_label.pack(anchor="w")
        
        self.exp_label = tk.Label(self.temp_frame, text="Caducidad posterior a hoy: ---", bg="white", font=("Segoe UI", 10))
        self.exp_label.pack(anchor="w")

    def get_txt(self, img, c=0.5):
        try:
            res = MODEL(DocumentFile.from_images(img))
            return [" ".join([w.value for w in l.words if w.confidence > c]) for p in res.pages for b in p.blocks for l in b.lines]
        except Exception as e:
            messagebox.showerror("Error OCR", f"No se pudo procesar {img}: {str(e)}")
            return []

    def start_analysis(self):
        self.btn_process.config(state="disabled")
        self.status_var.set("Procesando imágenes (doctr)... Espere, por favor.")
        self.tree.delete(*self.tree.get_children())
        
        Thread(target=self.run_logic).start()

    def run_logic(self):
        try:
            f_lines = self.get_txt('dni_front_especimen.jpg', 0.5)
            b_lines = self.get_txt('dni_back_especimen.jpg', 0.4)
            
            f_all = " ".join(f_lines).upper()
            m_c = [l.replace(" ", "") for l in b_lines if len(l.replace(" ", "")) >= 28 and ("<" in l or "ESP" in l)]
            
            if len(m_c) < 3:
                self.root.after(0, lambda: self.status_var.set("Error: MRZ no detectado en el reverso."))
                self.root.after(0, lambda: self.btn_process.config(state="normal"))
                return

            l1 = next((l for l in m_c if "ESP" in l[:10]), m_c[0])
            idx = m_c.index(l1)
            l2 = m_c[idx+1] if len(m_c) > idx+1 else m_c[1]
            l3 = m_c[idx+2] if len(m_c) > idx+2 else m_c[-1]
            
            m_data = {
                "id": l1[5:14].replace("<",""), "dni": l1[15:23].replace("<",""), 
                "s": l2[7], "b": f"{l2[4:6]} {l2[2:4]} 20{l2[0:2]}", "e": f"{l2[12:14]} {l2[10:12]} 20{l2[8:10]}", 
                "n": l3.replace("<", " ").strip()
            }
            
            dates = re.findall(r'\d{2} \d{2} \d{4}', f_all)
            iss = next((d for d in dates if d != m_data["b"] and d != m_data["e"]), "N/A")
            
            def vd(s, fut):
                try: 
                    d = datetime.strptime(s.replace(" ",""), "%d%m%Y")
                    return (d > T_TODAY) == fut
                except: return False

            dni_f = (re.search(r'\d{8}[A-Z]', f_all) or re.search(r'\d{8}', f_all)).group(0) if re.search(r'\d{8}', f_all) else "N/A"
            id_f = re.search(r'[A-Z]{3}\d{6}', f_all).group(0) if re.search(r'[A-Z]{3}\d{6}', f_all) else "N/A"
            
            # Preparar datos para la UI
            checks = [
                ("DNI", dni_f[:8], m_data["dni"]),
                ("IDESP", id_f, m_data["id"]),
                ("SEXO", m_data["s"] if m_data["s"] in f_all else "N/A", m_data["s"]),
                ("NACIM.", m_data["b"] if m_data["b"] in dates else "N/A", m_data["b"]),
                ("CADUC.", m_data["e"] if m_data["e"] in dates else "N/A", m_data["e"]),
            ]
            
            name_ok = any(x in f_all for x in m_data['n'].replace("K"," ").split() if len(x)>3)
            
            # Actualizar UI en el hilo principal
            self.root.after(0, lambda: self.update_ui(checks, m_data['n'], name_ok, iss, vd(iss, 0), vd(m_data['e'], 1)))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Ocurrió un error inesperado: {str(e)}"))
            self.root.after(0, lambda: self.btn_process.config(state="normal"))

    def update_ui(self, checks, full_name, name_ok, iss_date, v_emision, v_caducidad):
        # Limpiar y llenar tabla
        for name, fv, mv in checks:
            tag = "ok_row" if fv == mv else "fail_row"
            self.tree.insert("", "end", values=(name, fv, mv, "Y" if fv == mv else "N"), tags=(tag,))
        
        # Fila de Nombre
        n_tag = "ok_row" if name_ok else "fail_row"
        self.tree.insert("", "end", values=("NOMBRE", "DETECTOR", full_name, "Y" if name_ok else "N"), tags=(n_tag,))
        
        # Validaciones temporales
        color_em = "#065f46" if v_emision else "#991b1b"
        self.issue_label.config(text=f"Emisión anterior a hoy ({iss_date}): {'SÍ (Correcto)' if v_emision else 'NO (Error)'}", fg=color_em)
        
        color_cad = "#065f46" if v_caducidad else "#991b1b"
        self.exp_label.config(text=f"Caducidad posterior a hoy: {'SÍ (Vigente)' if v_caducidad else 'NO (Caducado)'}", fg=color_cad)
        
        self.status_var.set("Análisis completado con éxito.")
        self.btn_process.config(state="normal")

if __name__ == "__main__":
    root = tk.Tk()
    app = MRZApp(root)
    root.mainloop()
