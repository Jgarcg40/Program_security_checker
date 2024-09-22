import requests
import json
import time

# Esta aplicación recolecta todas las CVEs existentes de la API de nvd
base_url = "https://services.nvd.nist.gov/rest/json/cves/2.0/"
output_file = "nvd_1999_to_2024.json"

all_cves = []
results_per_page = 2000
start_index = 0

max_attempts = 5  # Número máximo de intentos en caso de error
total_cves = 252703  # Número total de CVEs conocido

while start_index < total_cves:
    url = f"{base_url}?resultsPerPage={results_per_page}&startIndex={start_index}"
    print(f"Downloading {url}")

    for attempt in range(max_attempts):
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if 'vulnerabilities' in data:
                all_cves.extend(data['vulnerabilities'])
            break
        elif response.status_code == 403: #Si devuelve un 403 porque se han hecho demasiadas peticiones espera 60 segundos
            print("Rate limit exceeded. Waiting for 60 seconds...")
            time.sleep(60)  # Esperar 60 segundos antes de reintentar
        else:
            print(f"Failed to download data: {response.status_code}. Retrying in 10 seconds...")
            time.sleep(10)  # Esperar 10 segundos antes de reintentar
    else:
        print(f"Failed to download data from {url} after several attempts.")
        break  # Salir del bucle si falla repetidamente

    start_index += results_per_page
    time.sleep(1)

# Guardar los datos en un archivo JSON
with open(output_file, 'w') as f:
    json.dump(all_cves, f, indent=2)

print(f"All data downloaded to {output_file}")

