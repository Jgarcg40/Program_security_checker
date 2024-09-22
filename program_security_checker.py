import sys
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import pandas as pd
import joblib
import os
import requests
import time
import warnings
from threading import Thread

# Comando pyinstaller: pyinstaller --onefile --add-data "trained_model_svm.pkl;." --add-data "icon-shield.ico;." --icon=icon-shield.ico --hidden-import imblearn --hidden-import sklearn.feature_extraction.text --hidden-import numpy.core.multiarray --noconsole program_security_checker.py
# Suprimir algunos warnings relacionados con las versiones de python
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn.base')

def resource_path(relative_path): # Rutas absolutas para usar con pyinstaller
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

# Cargar el modelo SVM (el mejor de los 3)
model_path = resource_path("trained_model_svm.pkl")
best_model = joblib.load(model_path)

# Función para mostrar la pantalla de carga cuando se está procesando los CVEs con nvd
def show_loading_screen():
    global loading_screen
    loading_screen = tk.Toplevel(root)
    loading_screen.title("Cargando...")
    loading_screen.geometry("300x100")
    loading_screen.configure(bg="#2C3E50")
    loading_screen.iconbitmap(resource_path("icon-shield.ico"))
    label = tk.Label(loading_screen, text="Cargando los datos, por favor espere...", bg="#2C3E50", fg="white", font=("Arial", 12))
    label.pack(expand=True, pady=20)

# Función para ocultar la pantalla de carga
def hide_loading_screen():
    loading_screen.destroy()

# Funcion para extraer CVEs del nvd
def get_cve_descriptions_from_nvd(program, version):
    base_url = "https://services.nvd.nist.gov/rest/json/cves/2.0/"

    # Normalizar el nombre del programa y la versión para usar la API
    program = program.lower().replace(" ", "%20")
    version = version.lower().replace(" ", "%20")

    # Definir la consulta de búsqueda de palabras clave
    query = f"keywordSearch={program}%20{version}&resultsPerPage=2000"
    url = f"{base_url}?{query}"

    descriptions = []
    cve_ids = []
    severity_scores = []
    max_attempts = 5

    for attempt in range(max_attempts):
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if 'vulnerabilities' in data:
                for item in data['vulnerabilities']:
                    cve_id = item['cve']['id']
                    description = item['cve']['descriptions'][0]['value']
                    severity_score = 0

                    if 'metrics' in item['cve']:
                        metrics = item['cve']['metrics']
                        if 'cvssMetricV2' in metrics and metrics['cvssMetricV2']:
                            severity_score = metrics['cvssMetricV2'][0]['cvssData'].get('baseScore', 0)
                        elif 'cvssMetricV3' in metrics and metrics['cvssMetricV3']:
                            severity_score = metrics['cvssMetricV3'][0]['cvssData'].get('baseScore', 0)

                    descriptions.append(description)
                    cve_ids.append((cve_id, program, version, severity_score))
                    severity_scores.append(severity_score)
            break
        elif response.status_code == 403: #Si devuelve un 403 porque se han hecho demasiadas peticiones espera 60 segundos
            print("Rate limit exceeded. Waiting for 60 seconds...")
            time.sleep(60)
        else:
            print(f"Failed to download data: {response.status_code}. Retrying in 10 seconds...")
            time.sleep(10)
    else:
        print(f"Failed to download data from {url} after several attempts.")

    return descriptions, cve_ids, severity_scores

# Calcular mediante una regla la puntuación de seguridad
def calculate_security_score(severity_scores):
    if not severity_scores:
        return 10.0

    avg_severity = sum(severity_scores) / len(severity_scores)
    score = max(1.0, 10.0 - avg_severity)
    return round(score, 2)


def process_file(file_path):
    show_loading_screen()  # Mostrar pantalla de carga
    with open(file_path, 'r') as file: # Leer archivo
        lines = file.readlines()

    programs_and_versions = [line.strip().split('-') for line in lines]

    all_descriptions = []
    all_cve_ids = []
    all_severity_scores = []
    all_program_version_pairs = []
    not_found_programs = []

    for program, version in programs_and_versions:
        descriptions, cve_ids, severity_scores = get_cve_descriptions_from_nvd(program, version)

        if not descriptions:
            not_found_programs.append((program, version))
        else:
            all_descriptions.extend(descriptions)
            all_cve_ids.extend(cve_ids)
            all_severity_scores.extend(severity_scores)
            all_program_version_pairs.extend([(program, version)] * len(descriptions))

    result_text.config(state=tk.NORMAL)
    result_text.delete(1.0, tk.END)

    # Mostrar programas y versiones no encontrados
    if not_found_programs:
        result_text.insert(tk.END, "\nNo se encontraron CVEs para las siguientes aplicaciones y versiones o no existen:\n", "error")
        for program, version in not_found_programs:
            result_text.insert(tk.END, f"Programa: {program}, Versión: {version}\n", "error")

    if not all_descriptions:
        result_text.insert(tk.END, "\nNo se encontraron CVEs para los programas y versiones especificados.", "error")
    else:
        predictions = best_model.predict(all_descriptions)

        # Esto asocia cada predicción con su programa y versión
        prediction_results = list(zip(predictions, all_program_version_pairs))

        # Esto calcula las probabilidades de ataque
        attack_probabilities = pd.Series(predictions).value_counts(normalize=True) * 100

        # Se agrupan los programas y versiones por tipo de ataque
        attack_sources = {}
        for attack, (program, version) in prediction_results:
            if attack not in attack_sources:
                attack_sources[attack] = set()
            attack_sources[attack].add(f"{program} {version}")

        # Puntuación de seguridad
        security_score = calculate_security_score(all_severity_scores)

        # Muestra los resultados en un recuadro de texto inmutable
        result_text.insert(tk.END, f"\nPuntuación de seguridad: {security_score}/10\n", "score")

        result_text.insert(tk.END, "\nProbabilidades de ataque:\n", "heading")
        for attack, probability in attack_probabilities.items():
            sources = ', '.join(attack_sources[attack])
            result_text.insert(tk.END, f"\n• {attack}: {probability:.2f}%\n   - {sources}\n", "data")

        result_text.insert(tk.END, "\n\nCVEs recolectados:\n\n", "heading")
        for cve_id, program, version, severity_score in all_cve_ids:
            result_text.insert(tk.END,
                               f"• {cve_id}\n   - Programa: {program.replace('%20', ' ')}\n   - Versión: {version.replace('%20', ' ')}\n   - Severidad: {severity_score}\n\n", "data")

    result_text.config(state=tk.DISABLED)
    hide_loading_screen()  # Ocultar pantalla de carga

# Funcion para cargar el .txt con los programas
def load_file():
    file_path = filedialog.askopenfilename(
        filetypes=[("Text files", "*.txt")],
        title="Seleccione un archivo .txt con los programas y versiones"
    )
    if file_path:
        thread = Thread(target=process_file, args=(file_path,))
        thread.start()
    else:
        messagebox.showerror("Error", "No se seleccionó ningún archivo.")


# Crear la interfaz gráfica
root = tk.Tk()
root.title("Análisis de seguridad en aplicaciones")
root.geometry("800x600")
root.configure(bg="#2C3E50")

# icono de la aplicación
root.iconbitmap(resource_path("icon-shield.ico"))

frame = tk.Frame(root, bg="#2C3E50")
frame.pack(pady=20)

# Botón para cargar el archivo
load_button = tk.Button(frame, text="Cargar archivo", command=load_file, bg="#E74C3C", fg="white", font=("Arial", 12, "bold"))
load_button.grid(row=0, column=0, padx=10)

# Label con una breve explicación
explanation_label = tk.Label(frame,
                             text="Suba un archivo .txt con programas y versiones.\nFormato: programa-version en cada línea.",
                             bg="#3498DB", fg="white", font=("Arial", 10), relief="solid", padx=10, pady=5)
explanation_label.grid(row=0, column=1, padx=10)

# Barra de scroll
result_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=25, font=("Arial", 10), bg="#ECF0F1", fg="#2C3E50")
result_text.pack(pady=10)

# Colores de fondo
result_text.tag_configure("error", foreground="#E74C3C", font=("Arial", 10, "bold"))
result_text.tag_configure("score", foreground="#27AE60", font=("Arial", 12, "bold"))
result_text.tag_configure("heading", foreground="#3498DB", font=("Arial", 11, "bold"))
result_text.tag_configure("data", foreground="#2C3E50", font=("Arial", 10))

result_text.config(state=tk.DISABLED)

root.mainloop()
