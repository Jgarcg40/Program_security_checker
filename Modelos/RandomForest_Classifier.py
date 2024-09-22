import pandas as pd
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
import joblib
import os

# Ruta del modelo en el directorio actual
model_path = "./trained_model_rf.pkl"

# Función para cargar y preprocesar datos
def load_and_preprocess_data():
    print("Fase 1: Carga y Preprocesamiento de Datos")

    # Configurar pandas para mostrar todas las columnas
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('display.max_colwidth', None)

    # Cargar el dataset JSON previamente extraido con la API
    with open('nvd_1999_to_2024.json', 'r') as file:
        data = json.load(file)

    # Extraer y normalizar las descripciones en inglés
    records = []
    for item in data:
        cve_id = item['cve']['id']
        descriptions = item['cve']['descriptions']
        for desc in descriptions:
            if desc['lang'] == 'en':
                description = desc['value']
                break
        else:
            description = None

        records.append({
            'cve_id': cve_id,
            'description': description,
        })

    # Convertir la lista de registros en un DataFrame
    cve_data = pd.DataFrame(records)

    # Lista de tipos de ataques
    attack_types = {
        'SQL Injection': r'sql injection',
        'Cross-Site Scripting': r'cross[-\s]site scripting|xss',
        'CSRF': r'csrf|cross[-\s]site request forgery',
        'Command Injection': r'command injection',
        'Buffer Overflow': r'buffer overflow',
        'Remote Code Execution': r'remote code execution|rce',
        'Information Disclosure': r'information disclosure',
        'Stack Overflow': r'stack overflow',
        'Zero-Day Attacks': r'zero[-\s]day',
        'DoS': r'denial of service|dos',
        'Privilege Escalation': r'privilege escalation',
        'Arbitrary Code Execution': r'arbitrary code execution',
        'Code Injection': r'code injection',
        'Spoofing': r'spoofing',
        'Authentication Bypass': r'authentication bypass',
        'Content Spoofing': r'content spoofing',
        'Unauthorized Access': r'unauthorized access',
        'Phishing Attacks': r'phishing',
        'Malware Attacks': r'malware',
        'Ransomware Attacks': r'ransomware',
        'Social Engineering Attacks': r'social engineering',
        'Supply Chain Attacks': r'supply chain',
        'Data Breach': r'data breach',
        'Man-in-the-Middle': r'man[-\s]in[-\s]the[-\s]middle',
        'Brute Force Attack': r'brute force',
        'Injection Attack': r'injection attack',
        'Directory Traversal': r'directory traversal',
        'Broken Authentication': r'broken authentication'
    }

    # Función para etiquetar descripciones
    def label_attack(description):
        for attack, pattern in attack_types.items():
            if re.search(pattern, description, re.IGNORECASE):
                return attack
        return 'Other'

    # Aplicar la función de etiquetado
    cve_data['attack_type'] = cve_data['description'].apply(label_attack)

    # Eliminar las entradas etiquetadas como 'Other' (limpieza)
    cve_data = cve_data[cve_data['attack_type'] != 'Other']

    # Mostrar las primeras filas del DataFrame etiquetado y filtrado
    print(cve_data.head())
    return cve_data

def train_and_evaluate_model(cve_data):
    print("Fase 2: Entrenamiento del Modelo")

    # Dividir en conjunto de entrenamiento y prueba
    X = cve_data['description']
    y = cve_data['attack_type']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear un pipeline que convierta el texto en TF-IDF y luego Random forest con SMOTE para clases con pocos datos
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('smote', SMOTE(k_neighbors=2)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Definir los hiperparámetros para la búsqueda
    param_grid = {
        'rf__n_estimators': [100, 150],  # Dos valores para el número de árboles
        'rf__max_depth': [10, 20],  # Dos valores para la profundidad máxima
        'rf__min_samples_split': [2, 5],  # Dos valores para el mínimo de muestras para dividir
        'rf__min_samples_leaf': [1, 2],  # Dos valores para el mínimo de muestras en una hoja
        'rf__max_features': ['auto', 'sqrt']  # Dos opciones para el número de características
    }

    # Realizar la búsqueda de cuadrícula con validación cruzada
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    # Guardar el mejor modelo entrenado
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, model_path)
    print("Modelo entrenado y guardado en", model_path)

    # Evaluar el modelo
    y_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=1))

# Verificar si el modelo ya está entrenado
if not os.path.exists(model_path):
    cve_data = load_and_preprocess_data()
    best_model = train_and_evaluate_model(cve_data)
else:
    print("Modelo encontrado. Cargando modelo...")
    best_model = joblib.load(model_path)

print("Fase 3: Realizar Predicciones con Datos de Prueba")

# Datos de prueba
test_cves = [
    "SQL injection vulnerability in product A version 1.0 allows attackers to execute arbitrary SQL commands.",
    "Cross-Site Scripting (XSS) vulnerability in product A version 1.0 allows remote attackers to inject arbitrary web script.",
    "Buffer overflow in product B version 2.0 allows remote attackers to execute arbitrary code.",
    "Remote code execution vulnerability in product B version 2.0 allows attackers to take control of the affected system.",
    "Information disclosure vulnerability in product C version 3.0 allows attackers to read sensitive information.",
    "Command injection vulnerability in product C version 3.0 allows remote attackers to execute arbitrary commands.",
    "Denial of Service (DoS) vulnerability in product D version 4.0 allows remote attackers to cause a denial of service.",
    "Privilege escalation vulnerability in product D version 4.0 allows local attackers to gain elevated privileges.",
    "Cross-Site Request Forgery (CSRF) vulnerability in product E version 5.0 allows remote attackers to hijack the authentication of users.",
    "Zero-Day vulnerability in product E version 5.0 allows attackers to exploit unknown vulnerabilities.",
    "Phishing attack vulnerability in product F version 1.2 allows attackers to steal user credentials.",
    "Malware injection vulnerability in product F version 1.2 allows remote attackers to install malicious software.",
    "Ransomware attack vulnerability in product G version 3.5 allows attackers to encrypt user data and demand ransom.",
    "Social engineering attack vulnerability in product G version 3.5 allows attackers to deceive users into revealing confidential information.",
    "Supply chain attack vulnerability in product H version 2.1 allows attackers to compromise the software during its distribution.",
    "Code injection vulnerability in product I version 4.2 allows remote attackers to execute arbitrary code.",
    "Authentication bypass vulnerability in product J version 1.3 allows attackers to bypass authentication mechanisms.",
    "Spoofing attack vulnerability in product K version 2.7 allows attackers to impersonate legitimate users.",
    "Arbitrary code execution vulnerability in product L version 5.0 allows attackers to execute arbitrary code on the affected system.",
    "Stack overflow vulnerability in product M version 6.1 allows remote attackers to execute arbitrary code.",
    "SQL injection in web application X version 3.2 allows attackers to bypass authentication.",
    "XSS vulnerability in web application Y version 4.1 permits attackers to steal cookies.",
    "Buffer overflow in application Z version 5.4 leads to arbitrary code execution.",
    "Remote code execution in server software A version 2.3.1 allows complete system compromise.",
    "Information disclosure in database software B version 6.2 reveals user passwords.",
    "Command injection in IoT device C version 1.0 allows root access.",
    "DoS vulnerability in firewall D version 3.1.5 causes system crash.",
    "Privilege escalation in mobile app E version 2.0 allows admin access.",
    "CSRF vulnerability in content management system F version 7.3.4 leads to account takeover.",
    "Zero-Day vulnerability in software G version 1.2.3 allows unknown exploits.",
    "Phishing attack in email service H version 9.0 tricks users into revealing credentials.",
    "Malware attack in software update I version 4.5.6 installs ransomware.",
    "Ransomware attack in company network J version 3.2.1 encrypts all files.",
    "Social engineering attack in HR software K version 6.5.3 manipulates employees.",
    "Supply chain attack in manufacturing software L version 5.0.0 compromises the distribution process.",
    "Code injection in online game M version 2.1.7 allows remote execution.",
    "Authentication bypass in financial software N version 1.3.9 bypasses security checks.",
    "Spoofing attack in VPN software O version 4.2.8 impersonates users.",
    "Arbitrary code execution in desktop application P version 3.3.3 allows full control.",
    "Stack overflow in web server Q version 2.8.9 leads to crash.",
    "SQL injection in shopping cart R version 3.0.0 allows data theft.",
    "XSS vulnerability in forum software S version 4.2.1 permits script execution.",
    "Buffer overflow in media player T version 1.9.6 leads to crash.",
    "Remote code execution in cloud service U version 3.3.2 allows data breach.",
    "Information disclosure in accounting software V version 2.4.5 exposes financial records.",
    "Command injection in router firmware W version 1.0.0 allows remote access.",
    "DoS vulnerability in antivirus software X version 5.2.3 causes system slowdown.",
    "Privilege escalation in office suite Y version 4.3.7 grants admin privileges.",
    "CSRF vulnerability in blogging platform Z version 2.9.1 allows unauthorized actions.",
    "Zero-Day vulnerability in enterprise software A version 6.1.4 remains unpatched.",
    "Phishing attack in social media app B version 3.7.5 tricks users into sharing info.",
    "Malware injection in game C version 2.5.0 installs spyware.",
    "Ransomware attack in healthcare system D version 1.1.8 locks patient records.",
    "Social engineering attack in customer service software E version 2.4.9 fools employees.",
    "Supply chain attack in logistics software F version 3.0.2 compromises shipments.",
    "Code injection in remote desktop software G version 1.5.6 executes commands.",
    "Authentication bypass in payment gateway H version 4.0.3 skips security checks.",
    "Spoofing attack in messaging app I version 3.6.1 impersonates contacts.",
    "Arbitrary code execution in e-commerce platform J version 2.7.4 allows full takeover.",
    "Stack overflow in database server K version 1.2.9 causes service crash.",
    "SQL injection in content management system L version 4.5.8 leaks database.",
    "XSS vulnerability in photo sharing app M version 3.2.2 executes scripts.",
    "Buffer overflow in printer firmware N version 2.0.0 leads to remote control.",
    "Remote code execution in backup software O version 4.1.3 breaches data.",
    "Information disclosure in email client P version 3.5.6 exposes emails.",
    "Command injection in smart home device Q version 1.7.2 allows remote commands.",
    "DoS vulnerability in video conferencing app R version 2.6.4 causes call drops.",
    "Privilege escalation in project management tool S version 4.8.1 grants root access.",
    "CSRF vulnerability in wiki software T version 3.1.2 permits unauthorized edits.",
    "Zero-Day vulnerability in collaboration software U version 2.5.7 remains exploitable.",
    "Phishing attack in banking app V version 3.4.1 deceives users into providing credentials.",
    "Malware injection in browser plugin W version 1.0.9 installs keylogger.",
    "Ransomware attack in manufacturing system X version 2.3.4 halts production.",
    "Social engineering attack in sales software Y version 1.6.3 manipulates sales reps.",
    "Supply chain attack in ERP software Z version 3.8.5 compromises processes.",
    "Code injection in web hosting service A version 2.4.0 allows arbitrary code.",
    "Authentication bypass in single sign-on system B version 1.2.5 skips login.",
    "Spoofing attack in email gateway C version 3.0.7 impersonates senders.",
    "Arbitrary code execution in accounting tool D version 4.9.1 allows control.",
    "Stack overflow in CRM software E version 2.1.6 crashes system.",
    "SQL injection in web analytics tool F version 3.3.3 leaks user data.",
    "XSS vulnerability in video platform G version 1.5.7 executes malicious scripts.",
    "Buffer overflow in IoT hub H version 2.6.8 leads to control.",
    "Remote code execution in monitoring tool I version 4.2.0 breaches security.",
    "Information disclosure in HR tool J version 3.1.9 exposes employee data.",
    "Command injection in firewall firmware K version 2.3.4 allows root commands.",
    "DoS vulnerability in online storage L version 1.4.6 causes service outage.",
    "Privilege escalation in photo editor M version 3.5.1 grants admin rights.",
    "CSRF vulnerability in news site N version 4.7.8 allows unauthorized posts.",
    "Zero-Day vulnerability in document management system O version 2.1.0 remains unpatched.",
    "Phishing attack in online banking P version 3.6.5 deceives customers.",
    "Malware injection in coding platform Q version 1.9.8 installs backdoor.",
    "Ransomware attack in supply chain management R version 2.7.3 halts operations.",
    "Social engineering attack in booking system S version 4.1.9 manipulates users.",
    "Supply chain attack in software distribution T version 3.4.7 compromises packages.",
    "Code injection in VPN service U version 1.0.3 allows remote code.",
    "Authentication bypass in encryption tool V version 2.8.6 skips security.",
    "Spoofing attack in email client W version 3.2.4 impersonates users.",
    "Arbitrary code execution in server management tool X version 1.3.7 grants control.",
    "Stack overflow in messaging server Y version 2.9.5 causes crash.",
    "SQL injection in customer portal Z version 4.2.3 leaks data.",
    "XSS vulnerability in blogging tool A version 3.6.1 executes scripts."
]
# Realizar predicciones
predictions = best_model.predict(test_cves)

# Mostrar las predicciones
for i, attack in enumerate(predictions):
    print(f"CVE {i + 1}: {attack}")
