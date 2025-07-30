import os
import numpy as np
import matplotlib.pyplot as plt
import shutil

# --- PATH ---
# AGGIORNATO: Percorso del nuovo dataset "Infineon"
base_path = r"C:\Users\User\Desktop\Project\Infineon\Dataset"
# AGGIORNATO: Percorso della cartella di output per il nuovo dataset
output_path = r"C:\Users\User\Desktop\Project\Infineon\Photos"

# --- FUNZIONE PER PULIRE LA CARTELLA DI OUTPUT ---
def clean_output_folder(folder_path):
    """
    Rimuove il contenuto della cartella specificata e la ricrea.
    ATTENZIONE: Questo cancellerà PERMANENTEMENTE il contenuto!
    """
    if os.path.exists(folder_path):
        print(f"Cleaning existing output folder: {folder_path}")
        shutil.rmtree(folder_path)
    os.makedirs(folder_path, exist_ok=True) # Ricrea la cartella vuota

# Pulisci la cartella prima di iniziare l'elaborazione
clean_output_folder(output_path)

# --- CONFIGURAZIONE ---
# Altezza MASSIMA fissa delle immagini in righe (corrisponde a 5 secondi di dati)
# L'altezza reale dell'immagine sarà l'altezza dei dati, fino a questo limite.
MAX_IMAGE_HEIGHT = 125 
# Larghezza fissa delle immagini in colonne (40 bins per RX0 + 40 bins per RX1 = 80 colonne)
IMAGE_WIDTH_MERGED = 80 

# Etichette valide nel formato originale dei file (le più specifiche prima per priorità)
RAW_VALID_LABELS = [
    "right arm up",
    "left arm up",
    "right leg up",
    "left leg up",
    "still position"
]

# Mappa per convertire le etichette raw in un formato pulito
LABEL_MAPPING = {
    "right arm up": "right_arm_up",
    "left arm up": "left_arm_up",
    "right leg up": "right_leg_up",
    "left leg up": "left_leg_up",
    "still position": "still_position"
}

# --- FUNZIONE PER PRE-PROCESSARE I DATI NPY (adattata per forma Infineon) ---
def process_infineon_npy_data(file_path):
    """
    Carica un file .npy di Infineon, calcola la magnitudine,
    seleziona il primo canale interno e i primi 40 range bins.
    """
    try:
        data = np.load(file_path)
        # Assumiamo che la forma sia (N_frames, 8, 128)
        if data.ndim == 3 and data.shape[1] >= 1 and data.shape[2] >= 40:
            # Calcola la magnitudine dei dati complessi e seleziona il primo "canale interno" (indice 0 della seconda dimensione)
            # e i primi 40 "range bins" (0:40 della terza dimensione).
            processed_data = np.abs(data)[:, 0, 0:40] # Risultato forma (N_frames, 40)
            return processed_data
        else:
            print(f"Errore: Formato dati non atteso ({data.shape}) o insufficiente per {os.path.basename(file_path)}. Atteso 3D (X, 8, 128) o simile.")
            return None
    except Exception as e:
        print(f"Errore durante il caricamento o l'elaborazione di {os.path.basename(file_path)}: {e}")
        return None

# --- GESTIONE E RAGGRUPPAMENTO FILE ---
all_files_in_dataset = os.listdir(base_path)
file_groups = {} 

for f in all_files_in_dataset:
    if f.endswith(".npy"):
        parts = f.rsplit('_', 2) 
        if len(parts) < 3: 
            print(f"Skipping malformed file name: {f}")
            continue

        acquisition_base_name = "_".join(parts[:-2]) 
        rx_suffix_full = parts[-1] 

        rx_key = None
        # Per Infineon, consideriamo solo RX0 e RX1 per questo script.
        # RX2 verrà gestito in un altro script se necessario.
        for suffix in ['rx0.npy', 'rx1.npy']: # Modificato per cercare solo RX0 e RX1
            if suffix in rx_suffix_full:
                rx_key = suffix.split('.')[0]
                break
        
        if not rx_key:
            # Se è un file RX2 o un tipo sconosciuto, lo saltiamo qui
            if 'rx2.npy' in rx_suffix_full:
                 print(f"Skipping file {f}: This script processes only RX0/RX1.")
            else:
                 print(f"Skipping file {f}: Cannot determine RX antenna type or not RX0/RX1 for this script.")
            continue

        if acquisition_base_name not in file_groups:
            # Inizializza solo con RX0 e RX1 per questo script
            file_groups[acquisition_base_name] = {'rx0': None, 'rx1': None, 'label': None}

        file_groups[acquisition_base_name][rx_key] = f

        found_label = None
        for label in sorted(RAW_VALID_LABELS, key=len, reverse=True): 
            if label.lower() in acquisition_base_name.lower(): 
                found_label = label
                break
        
        if found_label:
            file_groups[acquisition_base_name]['label'] = LABEL_MAPPING[found_label.lower()]
        else:
            print(f"Warning: No valid action label found for '{acquisition_base_name}'. This group will be skipped.")
            if acquisition_base_name in file_groups:
                del file_groups[acquisition_base_name]
            continue

print(f"Found {len(file_groups)} unique acquisition groups to potentially process.")

# --- ELABORAZIONE E GENERAZIONE IMMAGINI ---
generated_image_count = 0

for base_name, files_info in file_groups.items():
    rx0_file_name = files_info.get('rx0') # Usiamo .get per evitare KeyError se assente
    rx1_file_name = files_info.get('rx1') # Usiamo .get per evitare KeyError se assente
    current_label = files_info['label']

    if not current_label:
        continue 
    
    if rx0_file_name and rx1_file_name:
        data_rx0 = process_infineon_npy_data(os.path.join(base_path, rx0_file_name))
        data_rx1 = process_infineon_npy_data(os.path.join(base_path, rx1_file_name))
        
        if data_rx0 is None or data_rx1 is None:
            print(f"Skipping group '{base_name}' due to processing error or unsupported data format for RX0 or RX1.")
            continue

        # **VERIFICA ALTEZZA DEI DATI PRIMA DI CONCATENARE**
        if data_rx0.shape[0] != data_rx1.shape[0]:
            print(f"ERROR: Inconsistent heights for RX0 ({data_rx0.shape[0]}) and RX1 ({data_rx1.shape[0]}) in group '{base_name}'. Skipping.")
            continue

        current_data_height = data_rx0.shape[0]

        # Se i dati superano l'altezza massima desiderata, li tronchiamo
        if current_data_height > MAX_IMAGE_HEIGHT:
            data_rx0 = data_rx0[0:MAX_IMAGE_HEIGHT, :]
            data_rx1 = data_rx1[0:MAX_IMAGE_HEIGHT, :]
            current_data_height = MAX_IMAGE_HEIGHT # Aggiorna l'altezza corrente dopo il troncamento

        # Concatena i dati delle due antenne orizzontalmente
        merged_data = np.hstack((data_rx0, data_rx1))

        # Verifica che la larghezza combinata sia quella attesa (80)
        if merged_data.shape[1] != IMAGE_WIDTH_MERGED:
            print(f"ERROR: Merged data width is {merged_data.shape[1]}, but expected {IMAGE_WIDTH_MERGED} for group '{base_name}'. Skipping.")
            continue
        
        # Non è più necessario creare una matrice di zeri e copiare.
        # Utilizziamo direttamente i merged_data per l'immagine.
        final_image_data = merged_data 

        # --- Creazione e Salvataggio Immagine ---
        DPI = 100 
        
        # Calcola la dimensione della figura in pollici in base ai pixel e al DPI
        # L'altezza della figura si adatta all'altezza effettiva dei dati
        fig_width_inches = IMAGE_WIDTH_MERGED / float(DPI)
        fig_height_inches = current_data_height / float(DPI) 

        plt.figure(figsize=(fig_width_inches, fig_height_inches), dpi=DPI)
        
        plt.imshow(final_image_data, cmap='gray', aspect='auto') 
        plt.axis('off') 

        output_file_name_clean = base_name.replace(' ', '_')
        output_file_full_path = os.path.join(output_path, f"{current_label}.{output_file_name_clean}_Infineon_part1.png")

        print(f"Saving: {output_file_full_path}")

        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(output_file_full_path, bbox_inches='tight', pad_inches=0)
        plt.close() 
        generated_image_count += 1
    else:
        # Se mancano RX0 o RX1 per un gruppo, lo saltiamo
        print(f"Skipping group '{base_name}': Missing required RX0 or RX1 file for this script.")

print(f"\nElaboration completed! Total images generated: {generated_image_count}")
if generated_image_count > 0:
    # Qui l'altezza può variare fino a MAX_IMAGE_HEIGHT
    print(f"All generated images have a consistent width of {IMAGE_WIDTH_MERGED} pixels and a height of up to {MAX_IMAGE_HEIGHT} pixels.")