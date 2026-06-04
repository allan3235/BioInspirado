"""
Carga del dataset Oral Diseases (Kaggle: salmansajid05/oral-diseases)
para uso con CNN y algoritmos bioinspirados — versión PyTorch.

Estructura del flat dataset generado:
    dataset/oral-diseases-flat/
        train/
            Calculus/ | Caries/ | Gingivitis/ | Hypodontia/
            Mouth Ulcer/ | Tooth Discoloration/
        val/
            Calculus/ | Caries/ | Gingivitis/ | Hypodontia/
            Mouth Ulcer/ | Tooth Discoloration/
"""

import math
import os
import random
import shutil
import zipfile

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from ManejoDeDatos.basededatos import Imagen, ImagenUso, obtener_sesion

_n_train: int = 0


def get_n_train() -> int:
    return _n_train


# ─── Rutas ────────────────────────────────────────────────────────────────────
_REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR  = os.path.join(_REPO_ROOT, "dataset", "oral-diseases")
DATASET_FLAT = os.path.join(_REPO_ROOT, "dataset", "oral-diseases-flat")

YOLO_BASE = os.path.join(
    DATASET_DIR,
    "Caries_Gingivitus_ToothDiscoloration_Ulcer-yolo_annotated-Dataset",
    "Caries_Gingivitus_ToothDiscoloration_Ulcer-yolo_annotated-Dataset",
    "Data",
)

# ─── Configuración ────────────────────────────────────────────────────────────
DATASET_NAME = "salmansajid05/oral-diseases"
IMG_SIZE     = (224, 224)
BATCH_SIZE   = 32
SEED         = 42
VAL_SPLIT    = 0.2

IMAGE_EXTS   = {".jpg", ".jpeg", ".png", ".bmp"}
EXCLUDE_DIRS = {"preview"}

# ─── Datasets adicionales (caries) ────────────────────────────────────────────
DATASET_CAV_TEE_NAME    = "maazmakhdoom/dental-cavity-detection-dataset"
DATASET_CAV_TEE_DIR     = os.path.join(_REPO_ROOT, "dataset", "dental-cavity-cav-tee")

DATASET_CAVITY_YML_NAME = "shahjahanabdullatif/datasetcavitydetection"
DATASET_CAVITY_YML_DIR  = os.path.join(_REPO_ROOT, "dataset", "dental-cavity-yml")

FOLDER_CLASSES = {
    "Calculus":   os.path.join(DATASET_DIR, "Calculus",   "Calculus"),
    "Gingivitis": os.path.join(DATASET_DIR, "Gingivitis", "Gingivitis"),
    "Hypodontia": os.path.join(DATASET_DIR, "hypodontia", "hypodontia"),
}

FOLDER_AND_YOLO_CLASSES = {
    "Caries":              (os.path.join(DATASET_DIR, "Data caries",         "Data caries"),         0),
    "Mouth Ulcer":         (os.path.join(DATASET_DIR, "Mouth Ulcer",         "Mouth Ulcer"),          1),
    "Tooth Discoloration": (os.path.join(DATASET_DIR, "Tooth Discoloration", "Tooth Discoloration"),  2),
}


# ─── 1. Descarga del dataset ──────────────────────────────────────────────────
def _download_kaggle_dataset(name: str, dest_dir: str) -> None:
    if os.path.isdir(dest_dir) and any(os.scandir(dest_dir)):
        print(f"[INFO] Dataset '{name}' ya existe en '{dest_dir}'.")
        return
    import kaggle
    os.makedirs(dest_dir, exist_ok=True)
    print(f"[INFO] Descargando '{name}'...")
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(name, path=dest_dir, unzip=True)
    print(f"[INFO] Dataset '{name}' descargado en '{dest_dir}'.")


def download_dataset():
    if os.path.isdir(DATASET_DIR) and any(os.scandir(DATASET_DIR)):
        print(f"[INFO] Dataset raw ya existe en '{DATASET_DIR}'.")
    else:
        import kaggle
        os.makedirs(DATASET_DIR, exist_ok=True)
        print(f"[INFO] Descargando '{DATASET_NAME}'...")
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(DATASET_NAME, path=DATASET_DIR, unzip=False)

        zip_path = os.path.join(DATASET_DIR, "oral-diseases.zip")
        print(f"[INFO] Extrayendo '{zip_path}'...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.infolist():
                sanitized = "/".join(p.rstrip() for p in member.filename.split("/"))
                target = os.path.join(DATASET_DIR, sanitized.replace("/", os.sep))
                if member.is_dir():
                    os.makedirs(target, exist_ok=True)
                else:
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    with zf.open(member) as src, open(target, "wb") as dst:
                        dst.write(src.read())
        os.remove(zip_path)
        print(f"[INFO] Dataset extraído en '{DATASET_DIR}'.")

    _download_kaggle_dataset(DATASET_CAV_TEE_NAME, DATASET_CAV_TEE_DIR)
    _download_kaggle_dataset(DATASET_CAVITY_YML_NAME, DATASET_CAVITY_YML_DIR)
    _build_flat_dataset()


# ─── 2. Construcción del flat dataset ────────────────────────────────────────
def _build_flat_dataset():
    train_dir = os.path.join(DATASET_FLAT, "train")
    if os.path.isdir(train_dir) and any(os.scandir(train_dir)):
        print(f"[INFO] Dataset flat ya existe en '{DATASET_FLAT}'.")
        return

    print(f"[INFO] Construyendo dataset flat en '{DATASET_FLAT}'...")
    random.seed(SEED)

    staging_dir = os.path.join(DATASET_FLAT, "_yolo_crops_tmp")

    for class_name, src_dir in FOLDER_CLASSES.items():
        images = _collect_folder_images(src_dir)
        _shuffle_split_copy(images, class_name)

    for class_name, (src_dir, yolo_id) in FOLDER_AND_YOLO_CLASSES.items():
        folder_imgs = _collect_folder_images(src_dir)
        class_staging = os.path.join(staging_dir, class_name.replace(" ", "_"))
        yolo_crops = _generate_yolo_crops(yolo_id, class_staging)

        extra_crops = []
        if class_name == "Caries":
            extra_crops += _generate_yolo_obb_crops(
                DATASET_CAV_TEE_DIR,
                os.path.join(staging_dir, "caries_cav_tee"),
            )
            extra_crops += _generate_yolo_std_crops(
                DATASET_CAVITY_YML_DIR,
                class_id=0,
                staging_dir=os.path.join(staging_dir, "caries_yml"),
            )

        _shuffle_split_copy(folder_imgs + yolo_crops + extra_crops, class_name)

    if os.path.isdir(staging_dir):
        shutil.rmtree(staging_dir)

    print(f"[INFO] Dataset flat listo en '{DATASET_FLAT}'.")


def _collect_folder_images(src_dir: str) -> list:
    images = []
    for root, dirs, files in os.walk(src_dir):
        dirs[:] = [d for d in dirs if d.lower() not in EXCLUDE_DIRS]
        for fname in files:
            if os.path.splitext(fname)[1].lower() in IMAGE_EXTS:
                images.append(os.path.join(root, fname))
    return images


def _generate_yolo_crops(yolo_class_id: int, staging_dir: str) -> list:
    os.makedirs(staging_dir, exist_ok=True)
    crops = []

    for split in ("train", "val"):
        images_dir = os.path.join(YOLO_BASE, "images", split)
        labels_dir = os.path.join(YOLO_BASE, "labels", split)

        if not os.path.isdir(images_dir):
            continue

        for img_fname in sorted(os.listdir(images_dir)):
            if os.path.splitext(img_fname)[1].lower() not in IMAGE_EXTS:
                continue

            label_path = os.path.join(
                labels_dir, os.path.splitext(img_fname)[0] + ".txt"
            )
            if not os.path.isfile(label_path):
                continue

            img = cv2.imread(os.path.join(images_dir, img_fname))
            if img is None:
                continue

            h_img, w_img = img.shape[:2]

            with open(label_path) as f:
                for ann_idx, line in enumerate(f):
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    try:
                        cid = int(parts[0])
                    except ValueError:
                        continue
                    if cid != yolo_class_id:
                        continue

                    cx, cy, bw, bh = map(float, parts[1:])
                    x1 = max(0, int((cx - bw / 2) * w_img))
                    y1 = max(0, int((cy - bh / 2) * h_img))
                    x2 = min(w_img, int((cx + bw / 2) * w_img))
                    y2 = min(h_img, int((cy + bh / 2) * h_img))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    stem = os.path.splitext(img_fname)[0]
                    dst = os.path.join(staging_dir, f"{split}_{stem}_ann{ann_idx}.jpg")
                    cv2.imwrite(dst, img[y1:y2, x1:x2])
                    crops.append(dst)

    return crops


def _generate_yolo_obb_crops(dataset_dir: str, staging_dir: str) -> list:
    if not os.path.isdir(dataset_dir):
        print(f"[ADVERTENCIA] CAV-TEE no encontrado en '{dataset_dir}', se omite.")
        return []

    os.makedirs(staging_dir, exist_ok=True)
    crops = []

    for split in ("train", "valid", "val", "test"):
        images_dir = os.path.join(dataset_dir, split, "images")
        labels_dir = os.path.join(dataset_dir, split, "labels")
        if not os.path.isdir(images_dir):
            images_dir = os.path.join(dataset_dir, "images", split)
            labels_dir = os.path.join(dataset_dir, "labels", split)
        if not os.path.isdir(images_dir):
            continue

        for img_fname in sorted(os.listdir(images_dir)):
            if os.path.splitext(img_fname)[1].lower() not in IMAGE_EXTS:
                continue
            label_path = os.path.join(labels_dir, os.path.splitext(img_fname)[0] + ".txt")
            if not os.path.isfile(label_path):
                continue

            img = cv2.imread(os.path.join(images_dir, img_fname))
            if img is None:
                continue
            h_img, w_img = img.shape[:2]

            with open(label_path) as f:
                for ann_idx, line in enumerate(f):
                    parts = line.strip().split()
                    if len(parts) not in (6, 9):
                        continue
                    try:
                        int(parts[0])
                    except ValueError:
                        continue

                    if len(parts) == 6:
                        cx, cy, bw, bh, angle = map(float, parts[1:])
                        cos_a = abs(math.cos(math.radians(angle)))
                        sin_a = abs(math.sin(math.radians(angle)))
                        aabb_w = bw * cos_a + bh * sin_a
                        aabb_h = bw * sin_a + bh * cos_a
                        x1 = max(0, int((cx - aabb_w / 2) * w_img))
                        y1 = max(0, int((cy - aabb_h / 2) * h_img))
                        x2 = min(w_img, int((cx + aabb_w / 2) * w_img))
                        y2 = min(h_img, int((cy + aabb_h / 2) * h_img))
                    else:
                        coords = list(map(float, parts[1:]))
                        xs = [coords[i] * w_img for i in range(0, 8, 2)]
                        ys = [coords[i] * h_img for i in range(1, 8, 2)]
                        x1 = max(0, int(min(xs)))
                        y1 = max(0, int(min(ys)))
                        x2 = min(w_img, int(max(xs)))
                        y2 = min(h_img, int(max(ys)))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    stem = os.path.splitext(img_fname)[0]
                    dst = os.path.join(staging_dir, f"{split}_{stem}_ann{ann_idx}.jpg")
                    cv2.imwrite(dst, img[y1:y2, x1:x2])
                    crops.append(dst)

    print(f"  [CAV-TEE OBB] {len(crops)} crops extraídos de '{dataset_dir}'")
    return crops


def _generate_yolo_std_crops(dataset_dir: str, class_id: int, staging_dir: str) -> list:
    if not os.path.isdir(dataset_dir):
        print(f"[ADVERTENCIA] Dataset YML no encontrado en '{dataset_dir}', se omite.")
        return []

    os.makedirs(staging_dir, exist_ok=True)
    crops = []

    for split in ("train", "valid", "val", "test"):
        images_dir = os.path.join(dataset_dir, split, "images")
        labels_dir = os.path.join(dataset_dir, split, "labels")
        if not os.path.isdir(images_dir):
            images_dir = os.path.join(dataset_dir, "images", split)
            labels_dir = os.path.join(dataset_dir, "labels", split)
        if not os.path.isdir(images_dir):
            continue

        for img_fname in sorted(os.listdir(images_dir)):
            if os.path.splitext(img_fname)[1].lower() not in IMAGE_EXTS:
                continue
            label_path = os.path.join(labels_dir, os.path.splitext(img_fname)[0] + ".txt")
            if not os.path.isfile(label_path):
                continue

            img = cv2.imread(os.path.join(images_dir, img_fname))
            if img is None:
                continue
            h_img, w_img = img.shape[:2]

            with open(label_path) as f:
                for ann_idx, line in enumerate(f):
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    try:
                        cid = int(parts[0])
                    except ValueError:
                        continue
                    if cid != class_id:
                        continue

                    cx, cy, bw, bh = map(float, parts[1:])
                    x1 = max(0, int((cx - bw / 2) * w_img))
                    y1 = max(0, int((cy - bh / 2) * h_img))
                    x2 = min(w_img, int((cx + bw / 2) * w_img))
                    y2 = min(h_img, int((cy + bh / 2) * h_img))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    stem = os.path.splitext(img_fname)[0]
                    dst = os.path.join(staging_dir, f"{split}_{stem}_ann{ann_idx}.jpg")
                    cv2.imwrite(dst, img[y1:y2, x1:x2])
                    crops.append(dst)

    print(f"  [Cavity YML] {len(crops)} crops extraídos de '{dataset_dir}'")
    return crops


def _shuffle_split_copy(images: list, class_name: str):
    random.shuffle(images)
    split_idx = int(len(images) * (1 - VAL_SPLIT))
    splits = {"train": images[:split_idx], "val": images[split_idx:]}

    for split, paths in splits.items():
        dst_dir = os.path.join(DATASET_FLAT, split, class_name)
        os.makedirs(dst_dir, exist_ok=True)
        prefix = class_name.replace(" ", "_")
        for i, src in enumerate(paths):
            ext = os.path.splitext(src)[1].lower() or ".jpg"
            shutil.copy2(src, os.path.join(dst_dir, f"{prefix}_{i:05d}{ext}"))

    total = len(images)
    print(f"  [{class_name}] {split_idx} train / {total - split_idx} val  (total: {total})")


# ─── 3. Dataset desde carpetas (ImageFolder) ─────────────────────────────────

def load_dataset(use_augmentation: bool = True):
    """
    Prepara el flat dataset y retorna (train_dataset, val_dataset, class_names).
    Los datasets son compatibles con torch.utils.data.DataLoader.
    """
    download_dataset()

    train_dir = os.path.join(DATASET_FLAT, "train")
    val_dir   = os.path.join(DATASET_FLAT, "val")

    base_transform = T.Compose([
        T.Resize(IMG_SIZE),
        T.ToTensor(),
    ])

    if use_augmentation:
        train_transform = T.Compose([
            T.Resize(IMG_SIZE),
            T.RandomHorizontalFlip(),
            T.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            T.ToTensor(),
        ])
    else:
        train_transform = base_transform

    train_dataset = ImageFolder(train_dir, transform=train_transform)
    val_dataset   = ImageFolder(val_dir,   transform=base_transform)

    class_names = train_dataset.classes
    print(f"[INFO] Clases ({len(class_names)}): {class_names}")
    print(f"[INFO] Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    return train_dataset, val_dataset, class_names


# ─── 4. Carga desde Base de Datos ────────────────────────────────────────────

def _consultar_clases_y_conteos(uso: str):
    sesion = obtener_sesion()
    resultados = (
        sesion.query(Imagen.enfermedad)
        .join(ImagenUso, ImagenUso.imagen_id == Imagen.id)
        .filter(ImagenUso.uso == uso)
        .all()
    )
    sesion.close()
    enfermedades = [e for (e,) in resultados]
    return sorted(set(enfermedades)), len(enfermedades)


def _consultar_ids(uso: str):
    sesion = obtener_sesion()
    ids = (
        sesion.query(Imagen.id)
        .join(ImagenUso, ImagenUso.imagen_id == Imagen.id)
        .filter(ImagenUso.uso == uso)
        .all()
    )
    sesion.close()
    return [i for (i,) in ids]


def _decodificar_imagen(datos_binarios: bytes) -> np.ndarray:
    """Convierte bytes a array numpy RGB redimensionado a IMG_SIZE."""
    arr = np.frombuffer(datos_binarios, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    return img.astype(np.float32) / 255.0


class BDDataset(Dataset):
    """
    Dataset PyTorch que carga imágenes desde PostgreSQL.
    Almacena los bytes comprimidos en RAM y decodifica en __getitem__
    para minimizar el uso de memoria sin abrir una conexión por muestra.
    """

    def __init__(self, ids: list, clase_a_idx: dict):
        CHUNK = 200
        self._raw: list[bytes] = []
        self._labels: list[int] = []

        for start in range(0, len(ids), CHUNK):
            bloque = ids[start: start + CHUNK]
            sesion = obtener_sesion()
            registros = (
                sesion.query(Imagen.imagen, Imagen.enfermedad)
                .filter(Imagen.id.in_(bloque))
                .all()
            )
            sesion.close()
            for datos, enfermedad in registros:
                self._raw.append(bytes(datos))
                self._labels.append(clase_a_idx[enfermedad])

    def __len__(self):
        return len(self._raw)

    def __getitem__(self, idx):
        img = _decodificar_imagen(self._raw[idx])          # (H, W, C) float32 [0,1]
        img = torch.from_numpy(img).permute(2, 0, 1)       # → (C, H, W)
        return img, self._labels[idx]


def load_dataset_bd():
    """
    Carga el dataset desde PostgreSQL y retorna (train_dataset, val_dataset, class_names).
    Los datasets son instancias de BDDataset y son compatibles con DataLoader.
    """
    print("[INFO] Consultando clases y conteos desde la BD...")
    class_names, n_train = _consultar_clases_y_conteos("entrenar")
    _,           n_val   = _consultar_clases_y_conteos("validar")
    clase_a_idx = {c: i for i, c in enumerate(class_names)}
    num_clases  = len(class_names)

    print(f"[INFO] Clases ({num_clases}): {class_names}")
    print(f"[INFO] Train: {n_train} | Val: {n_val}")

    train_ids = _consultar_ids("entrenar")
    val_ids   = _consultar_ids("validar")

    global _n_train
    _n_train = n_train

    print("[INFO] Cargando imágenes de entrenamiento desde BD...")
    train_dataset = BDDataset(train_ids, clase_a_idx)
    print("[INFO] Cargando imágenes de validación desde BD...")
    val_dataset   = BDDataset(val_ids, clase_a_idx)

    return train_dataset, val_dataset, class_names


# ─── Prueba rápida ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    train_ds, val_ds, class_names = load_dataset()
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    imgs, labels = next(iter(train_loader))
    print(f"\n[OK] Dataset listo — batch shape: {imgs.shape}, labels: {labels[:5]}")
