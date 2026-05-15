
"""
Carga del dataset Oral Diseases (Kaggle: salmansajid05/oral-diseases)
para uso con CNN y algoritmos bioinspirados.
"""

import os
import shutil
import zipfile
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import kaggle

# ─── Configuración ────────────────────────────────────────────────────────────
DATASET_NAME  = "salmansajid05/oral-diseases"
DATASET_DIR   = "dataset/oral-diseases"
DATASET_FLAT  = "dataset/oral-diseases-flat"   # estructura limpia para Keras
IMG_SIZE      = (224, 224)
BATCH_SIZE    = 32
SEED          = 42

# Mapeo: nombre de la carpeta en el ZIP → nombre de clase final
CLASS_FOLDERS = {
    "Calculus":            "Calculus",
    "Gingivitis":          "Gingivitis",
    "Mouth Ulcer":         "Mouth Ulcer",
    "Tooth Discoloration": "Tooth Discoloration",
    "Data caries":         "Caries",
}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


# ─── 1. Descarga del dataset ──────────────────────────────────────────────────
def download_dataset():
    """Descarga el dataset desde Kaggle y lo extrae manejando rutas de Windows."""
    if os.path.isdir(DATASET_DIR) and any(os.scandir(DATASET_DIR)):
        print(f"[INFO] Dataset raw ya existe en '{DATASET_DIR}', se omite la descarga.")
    else:
        os.makedirs(DATASET_DIR, exist_ok=True)
        print(f"[INFO] Descargando '{DATASET_NAME}'...")
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(DATASET_NAME, path=DATASET_DIR, unzip=False)

        zip_path = os.path.join(DATASET_DIR, "oral-diseases.zip")
        print(f"[INFO] Extrayendo '{zip_path}'...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.infolist():
                # Eliminar espacios al final de cada segmento (bug en el ZIP original)
                sanitized = "/".join(part.rstrip() for part in member.filename.split("/"))
                target = os.path.join(DATASET_DIR, sanitized.replace("/", os.sep))
                if member.is_dir():
                    os.makedirs(target, exist_ok=True)
                else:
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    with zf.open(member) as src, open(target, "wb") as dst:
                        dst.write(src.read())
        os.remove(zip_path)
        print(f"[INFO] Dataset extraído en '{DATASET_DIR}'.")

    _build_flat_dataset()


# ─── 2. Reorganizar en estructura plana  ─────────────────────────────────────
def _build_flat_dataset():
    """
    Crea DATASET_FLAT con una carpeta por clase que contiene todas sus imágenes.
    Estructura resultante:
        dataset/oral-diseases-flat/
            Calculus/
            Gingivitis/
            Mouth Ulcer/
            Tooth Discoloration/
            Caries/
    """
    if os.path.isdir(DATASET_FLAT) and any(os.scandir(DATASET_FLAT)):
        print(f"[INFO] Dataset flat ya existe en '{DATASET_FLAT}', se omite la reorganización.")
        return

    print(f"[INFO] Reorganizando dataset en '{DATASET_FLAT}'...")
    for src_folder, class_name in CLASS_FOLDERS.items():
        src_dir = os.path.join(DATASET_DIR, src_folder)
        dst_dir = os.path.join(DATASET_FLAT, class_name)
        os.makedirs(dst_dir, exist_ok=True)

        count = 0
        for root, _, files in os.walk(src_dir):
            for fname in files:
                if os.path.splitext(fname)[1].lower() in IMAGE_EXTS:
                    src_path = os.path.join(root, fname)
                    # Prefijo de ruta relativa para evitar colisiones de nombres
                    rel = os.path.relpath(root, src_dir).replace(os.sep, "_")
                    dst_name = f"{rel}_{fname}" if rel != "." else fname
                    shutil.copy2(src_path, os.path.join(dst_dir, dst_name))
                    count += 1
        print(f"  {class_name}: {count} imagenes")

    print(f"[INFO] Reorganizacion completada en '{DATASET_FLAT}'.")


# ─── 3. Crear generadores de datos ───────────────────────────────────────────
def load_dataset(data_root: str = None, use_augmentation: bool = True):
    """
    Carga el dataset y devuelve (train_gen, val_gen, class_names).

    Parametros
    ----------
    data_root        : Ruta al directorio raiz con las carpetas de clases.
                       Si es None se usa DATASET_FLAT.
    use_augmentation : Aplica augmentacion al conjunto de entrenamiento.

    Retorna
    -------
    train_gen   : generador de entrenamiento
    val_gen     : generador de validacion
    class_names : lista de nombres de clases
    """
    if data_root is None:
        download_dataset()
        data_root = DATASET_FLAT

    print(f"[INFO] Cargando datos desde: {data_root}")

    if use_augmentation:
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            validation_split=0.2,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
        )
    else:
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            validation_split=0.2,
        )

    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
    )

    train_gen = train_datagen.flow_from_directory(
        data_root,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        seed=SEED,
    )

    val_gen = val_datagen.flow_from_directory(
        data_root,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        seed=SEED,
    )

    class_names = list(train_gen.class_indices.keys())
    num_classes = len(class_names)

    print(f"[INFO] Clases ({num_classes}): {class_names}")
    print(f"[INFO] Muestras de entrenamiento : {train_gen.samples}")
    print(f"[INFO] Muestras de validacion    : {val_gen.samples}")

    return train_gen, val_gen, class_names


# ─── 4. Version tf.data (opcional, mas eficiente en GPU) ─────────────────────
def load_dataset_tfdata(data_root: str = None):
    """
    Alternativa con tf.data.Dataset para mayor rendimiento.
    Retorna (train_ds, val_ds, class_names).
    """
    if data_root is None:
        download_dataset()
        data_root = DATASET_FLAT

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_root,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_root,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    class_names = train_ds.class_names
    print(f"[INFO] Clases ({len(class_names)}): {class_names}")

    normalization = tf.keras.layers.Rescaling(1.0 / 255)
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.map(lambda x, y: (normalization(x), y), num_parallel_calls=AUTOTUNE)
    val_ds   = val_ds.map(lambda x, y: (normalization(x), y),   num_parallel_calls=AUTOTUNE)

    train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
    val_ds   = val_ds.cache().prefetch(AUTOTUNE)

    return train_ds, val_ds, class_names


# ─── Prueba rapida ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Opcion A: ImageDataGenerator (compatible con Keras clasico)
    train_gen, val_gen, class_names = load_dataset()

    # Opcion B: tf.data (mas eficiente, descomenta si prefieres esta version)
    # train_ds, val_ds, class_names = load_dataset_tfdata()

    print("\n[OK] Dataset listo para usar en la CNN.")
