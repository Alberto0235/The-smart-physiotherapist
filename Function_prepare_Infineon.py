import os
import numpy as np
import matplotlib.pyplot as plt
import shutil

# --- PATH ---
# AGGIORNATO: Percorso del nuovo dataset "Infineon"
base_path = r"C:\Users\Alb3r\Desktop\HAEEAI project\Project\Infineon\Dataset"
# AGGIORNATO: Percorso della cartella di output per il nuovo dataset
output_path = r"C:\Users\Alb3r\Desktop\HAEEAI project\Project\Infineon\Photos"

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
# Altezza fissa delle immagini in righe (corrisponde a 5 secondi di dati)
IMAGE_HEIGHT = 125 
# Larghezza fissa delle immagini in colonne (40 bins per RX0 + 40 bins per RX1 = 80 colonne)
IMAGE_WIDTH_MERGED = 80 

# Etichette valide nel formato originale dei file (le più specifiche prima per priorità)
# Queste sono le stringhe che il codice cercherà nei nomi dei file per identificare la classe.
RAW_VALID_LABELS = [
    "right arm up",
    "left arm up",
    "right leg up",
    "left leg up",
    "still position"
]

# Mappa per convertire le etichette raw in un formato pulito per i nomi delle classi/file
# es. "right arm up" -> "right_arm_up" per Edge Impulse.
LABEL_MAPPING = {
    "right arm up": "right_arm_up",
    "left arm up": "left_arm_up",
    "right leg up": "right_leg_up",
    "left leg up": "left_leg_up",
    "still position": "still_position"
}

# --- GESTIONE E RAGGRUPPAMENTO FILE ---
all_files_in_dataset = os.listdir(base_path)
file_groups = {} # Mappa per raggruppare i file per acquisizione e identificare le etichette

for f in all_files_in_dataset:
    if f.endswith(".npy"):
        # Estrai la parte base del nome del file per raggruppare (escludendo _Infineon_rxN.npy)
        # Dividiamo dal terzo underscore partendo dalla fine: [base]_[Infineon]_[rxN.npy]
        parts = f.rsplit('_', 2) 
        if len(parts) < 3: # Assicurati che il formato del nome sia corretto
            print(f"Skipping malformed file name: {f}")
            continue

        # Ora il base_name esclude '_Infineon' e '_rxN.npy'
        acquisition_base_name = "_".join(parts[:-2]) 
        rx_suffix_full = parts[-1] # Es: rx2.npy

        # Estrai 'rx0', 'rx1' o 'rx2' dal suffisso
        rx_key = None
        for suffix in ['rx0.npy', 'rx1.npy', 'rx2.npy']:
            if suffix in rx_suffix_full:
                rx_key = suffix.split('.')[0]
                break
        
        if not rx_key:
            print(f"Skipping file {f}: Cannot determine RX antenna type.")
            continue

        # Inizializza il gruppo se non esiste
        if acquisition_base_name not in file_groups:
            file_groups[acquisition_base_name] = {'rx0': None, 'rx1': None, 'rx2': None, 'label': None}

        # Assegna il file al suo slot RX
        file_groups[acquisition_base_name][rx_key] = f

        # Estrai l'etichetta dell'azione/classe dal nome base
        found_label = None
        for label in sorted(RAW_VALID_LABELS, key=len, reverse=True): # Cerca l'etichetta più specifica prima
            if label.lower() in acquisition_base_name.lower():
                found_label = label
                break
        
        if found_label:
            file_groups[acquisition_base_name]['label'] = LABEL_MAPPING[found_label.lower()]
        else:
            print(f"Warning: No valid action label found for '{acquisition_base_name}'. This group will be skipped.")
            # Rimuovi il gruppo se non si trova un'etichetta valida
            if acquisition_base_name in file_groups:
                del file_groups[acquisition_base_name]
            continue

print(f"Found {len(file_groups)} unique acquisition groups to potentially process.")

# --- ELABORAZIONE E GENERAZIONE IMMAGINI ---
generated_image_count = 0

for base_name, files_info in file_groups.items():
    rx0_file_name = files_info['rx0']
    rx1_file_name = files_info['rx1']
    current_label = files_info['label']

    if not current_label:
        continue 
    
    if rx0_file_name and rx1_file_name:
        try:
            # Carica i dati .npy
            data_rx0 = np.load(os.path.join(base_path, rx0_file_name))
            data_rx1 = np.load(os.path.join(base_path, rx1_file_name))
        except Exception as e:
            print(f"Error loading files for group '{base_name}': {e}. Skipping this group.")
            continue

        # --- MODIFICA FONDAMENTALE PER GESTIRE LA FORMA (X, 8, 128) ---
        # Assumiamo che la forma sia (frames, canali_interni, range_bins_completi)
        # E che vogliamo prendere il primo canale interno e i primi 40 range bins
        
        # Calcola la magnitudine dei dati complessi
        data_abs_rx0 = np.abs(data_rx0) # Forma (N_frames, 8, 128)
        data_abs_rx1 = np.abs(data_rx1) # Forma (N_frames, 8, 128)

        # Seleziona il primo "canale interno" (indice 0 della seconda dimensione)
        # e i primi 40 "range bins" (0:40 della terza dimensione).
        # Questo riduce la forma a (N_frames, 40) per ogni RX.
        data_float_rx0 = data_abs_rx0[:, 0, 0:40]
        data_float_rx1 = data_abs_rx1[:, 0, 0:40]
        
        # Concatena i dati delle due antenne orizzontalmente
        # Questo creerà una matrice di (N_righe x 80 colonne)
        merged_data = np.hstack((data_float_rx0, data_float_rx1))

        current_rows = merged_data.shape[0]
        current_cols = merged_data.shape[1] 

        # Creiamo una matrice vuota della dimensione finale desiderata (125x80)
        final_image_data = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH_MERGED))

        rows_to_copy = min(current_rows, IMAGE_HEIGHT)
        cols_to_copy = min(current_cols, IMAGE_WIDTH_MERGED)
        
        final_image_data[0:rows_to_copy, 0:cols_to_copy] = merged_data[0:rows_to_copy, 0:cols_to_copy]

        # --- Creazione e Salvataggio Immagine ---
        DPI = 100 
        
        fig_width_inches = IMAGE_WIDTH_MERGED / float(DPI)
        fig_height_inches = IMAGE_HEIGHT / float(DPI)

        plt.figure(figsize=(fig_width_inches, fig_height_inches), dpi=DPI)
        
        plt.imshow(final_image_data, cmap='gray', aspect='auto') 
        plt.axis('off') 

        # Prepara il nome del file di output per Edge Impulse
        # Esempio: right_arm_up.Alberto_Right_arm_up_Fisio_20250528-145949_Infineon_part1.png
        output_file_name_clean = base_name.replace(' ', '_')
        # AGGIORNATO: suffisso '_Infineon' nel nome del file di output
        output_file_full_path = os.path.join(output_path, f"{current_label}.{output_file_name_clean}_Infineon_part1.png")

        print(f"Saving: {output_file_full_path}")

        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(output_file_full_path, bbox_inches='tight', pad_inches=0)
        plt.close() 
        generated_image_count += 1
    else:
        # Se mancano RX0 o RX1 per un gruppo, lo saltiamo
        print(f"Skipping group '{base_name}': Missing required RX0 or RX1 file.")

print(f"\nElaboration completed! Total images generated: {generated_image_count}")