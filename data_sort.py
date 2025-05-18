import os
import shutil
import random
from pathlib import Path

# Configuración
ORIGIN_DIR = "Dataset"  # carpeta que contiene 'cry' y 'not_cry'
TARGET_DIR = "data"      # carpeta raíz destino

SPLITS = {
    "train": 0.7,
    "validation": 0.10,
    "test": 0.20
}

CLASSES = ["cry", "not_cry"]

random.seed(42)  # para resultados reproducibles

# Crear carpetas destino
for split in SPLITS.keys():
    for cls in CLASSES:
        split_dir = os.path.join(TARGET_DIR, split, cls)
        os.makedirs(split_dir, exist_ok=True)

# Procesar cada clase
for cls in CLASSES:
    class_path = os.path.join(ORIGIN_DIR, cls)
    files = os.listdir(class_path)
    files = [f for f in files if os.path.isfile(os.path.join(class_path, f))]
    random.shuffle(files)

    total = len(files)
    train_end = int(SPLITS["train"] * total)
    val_end = train_end + int(SPLITS["validation"] * total)

    split_files = {
        "train": files[:train_end],
        "validation": files[train_end:val_end],
        "test": files[val_end:]
    }

    # Copiar archivos a carpetas correspondientes
    for split, file_list in split_files.items():
        for fname in file_list:
            src = os.path.join(class_path, fname)
            dst = os.path.join(TARGET_DIR, split, cls, fname)
            shutil.copy2(src, dst)  # usa copy2 para mantener metadata (como timestamps)

print("✅ Archivos organizados correctamente.")