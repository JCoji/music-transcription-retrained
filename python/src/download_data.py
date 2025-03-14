import requests
import zipfile
import os
from tqdm import tqdm

# URL do głównego pliku ZIP
main_zip_url = "https://zenodo.org/api/records/3371780/files-archive"

# Ścieżka do folderu głównego projektu (jeden poziom wyżej niż src)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Główny folder na dane
data_folder = os.path.join(project_root, "data", "guitar_set")

# Ścieżka do zapisania głównego pliku ZIP
main_zip_path = os.path.join(data_folder, "main_guitar_set.zip")

# Tymczasowy folder na rozpakowanie głównego ZIP
extract_main_path = os.path.join(data_folder, "temp")

# Lista wewnętrznych plików ZIP i ich folderów docelowych
internal_zips = [
    {"name": "annotation.zip", "extract_to": os.path.join(data_folder, "annotations")},
    {"name": "audio_hex-pickup_debleeded.zip", "extract_to": os.path.join(data_folder, "audio_hex_pickup_debleeded")},
    {"name": "audio_hex-pickup_original.zip", "extract_to": os.path.join(data_folder, "audio_hex_pickup_original")},
    {"name": "audio_mono-mic.zip", "extract_to": os.path.join(data_folder, "audio_mono_mic")},
    {"name": "audio_mono-pickup_mix.zip", "extract_to": os.path.join(data_folder, "audio_mono_pickup_mix")}
]

# Utwórz folder data, jeśli nie istnieje
os.makedirs(data_folder, exist_ok=True)

# Pobierz główny plik ZIP z paskiem postępu
print("Pobieranie głównego pliku ZIP...")
response = requests.get(main_zip_url, stream=True)
total_size = int(response.headers.get('content-length', 0))
block_size = 8192

# Użyj tqdm do wyświetlenia paska postępu
with open(main_zip_path, 'wb') as f:
    for chunk in tqdm(response.iter_content(chunk_size=block_size), total=total_size // block_size, unit='KB', unit_scale=True):
        f.write(chunk)
print("Główny plik ZIP został pobrany.")

# Rozpakuj główny plik ZIP do tymczasowego folderu
print("Rozpakowywanie głównego pliku ZIP...")
os.makedirs(extract_main_path, exist_ok=True)
with zipfile.ZipFile(main_zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_main_path)
print("Główny plik ZIP został rozpakowany.")

# Rozpakuj każdy wewnętrzny plik ZIP do odpowiednich folderów
for internal_zip in internal_zips:
    zip_name = internal_zip["name"]
    extract_to = internal_zip["extract_to"]
    zip_path = os.path.join(extract_main_path, zip_name)

    # Utwórz folder docelowy, jeśli nie istnieje
    os.makedirs(extract_to, exist_ok=True)

    # Rozpakuj plik ZIP
    print(f"Rozpakowywanie {zip_name} do {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"{zip_name} został rozpakowany do {extract_to}.\n")
