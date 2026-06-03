"""
Script para subir todas las imagenes del flat dataset a PostgreSQL.
Lee de dataset/oral-diseases-flat/ y las inserta en las tablas imagenes e imagenes_uso.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ManejoDeDatos.basededatos import Imagen, ImagenUso, iniciar_bd, obtener_sesion

DATASET_FLAT = os.path.join(_REPO_ROOT, "dataset", "oral-diseases-flat")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

USO_MAP = {
    "train": "entrenar",
    "val":   "validar",
}


def subir_imagenes():
    iniciar_bd()
    sesion = obtener_sesion()

    total = 0

    for split, uso in USO_MAP.items():
        split_dir = os.path.join(DATASET_FLAT, split)

        if not os.path.isdir(split_dir):
            print(f"[ADVERTENCIA] No se encontro la carpeta: {split_dir}")
            continue

        for enfermedad in os.listdir(split_dir):
            enfermedad_dir = os.path.join(split_dir, enfermedad)

            if not os.path.isdir(enfermedad_dir):
                continue

            archivos = [
                f for f in os.listdir(enfermedad_dir)
                if os.path.splitext(f)[1].lower() in IMAGE_EXTS
            ]

            for archivo in archivos:
                ruta = os.path.join(enfermedad_dir, archivo)

                with open(ruta, "rb") as f:
                    datos = f.read()

                imagen = Imagen(imagen=datos, enfermedad=enfermedad)
                sesion.add(imagen)
                sesion.flush()

                sesion.add(ImagenUso(imagen_id=imagen.id, uso=uso))
                total += 1

            print(f"  [{uso}] {enfermedad}: {len(archivos)} imagenes")

    sesion.commit()
    sesion.close()
    print(f"\n[OK] {total} imagenes subidas a la base de datos.")


if __name__ == "__main__":
    subir_imagenes()
