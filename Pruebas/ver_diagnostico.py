"""
ver_diagnostico.py — Descarga todo: imágenes, modelo .pt, hiperparámetros y métricas CSV.
Uso:
    python ver_diagnostico.py                  # último modelo guardado
    python ver_diagnostico.py --modelo 42      # modelo específico
    python ver_diagnostico.py --exp 5          # todos los modelos del experimento 5
    python ver_diagnostico.py --url "postgresql://..."  # URL externa
"""

import csv
import os
import sys
import argparse

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "final_docker"))

from dotenv import load_dotenv
load_dotenv(os.path.join(_ROOT, ".env"))
load_dotenv(os.path.join(_ROOT, "final_docker", "ManejoDeDatos", ".env"))

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from final_docker.ManejoDeDatos.basededatos import (
    ResultadoDiagnostico, Modelo, Metrica, Hiperparametro,
    obtener_diagnosticos_experimento,
)

IMAGENES = [
    ("curva_perdida",    "curva_perdida.png"),
    ("curva_precision",  "curva_precision.png"),
    ("matriz_confusion", "matriz_confusion.png"),
    ("curva_recall_f1",  "curva_recall_f1.png"),
]


def descargar_modelo(sesion: Session, modelo_id: int, carpeta: str) -> None:
    modelo = sesion.get(Modelo, modelo_id)
    if modelo is None:
        print(f"  [!] Modelo {modelo_id} no encontrado")
        return

    # Archivo .pt
    if modelo.archivo:
        ruta = os.path.join(carpeta, f"modelo_{modelo_id}.pt")
        with open(ruta, "wb") as f:
            f.write(modelo.archivo)
        print(f"  modelo pytorch ->{ruta}")
    else:
        print(f"  [!] modelo_{modelo_id}: sin archivo serializado en BD")

    # Hiperparámetros ->txt
    hip = sesion.execute(
        select(Hiperparametro).where(Hiperparametro.modelo_id == modelo_id)
    ).scalar_one_or_none()
    if hip:
        ruta = os.path.join(carpeta, "hiperparametros.txt")
        with open(ruta, "w") as f:
            f.write(f"modelo_id          : {modelo_id}\n")
            f.write(f"tasa_aprendizaje   : {hip.tasa_aprendizaje}\n")
            f.write(f"tamano_lote        : {hip.tamano_lote}\n")
            f.write(f"optimizador        : {hip.optimizador}\n")
            f.write(f"epocas             : {hip.epocas}\n")
            f.write(f"aumento_datos      : {hip.aumento_datos}\n")
            f.write(f"neuronas_densas    : {hip.neuronas_densas}\n")
            f.write(f"tasa_dropout       : {hip.tasa_dropout}\n")
            f.write(f"filtros_conv       : {hip.filtros_conv}\n")
        print(f"  hiperparámetros ->{ruta}")

    # Métricas por época ->CSV
    metricas = sesion.execute(
        select(Metrica).where(Metrica.modelo_id == modelo_id).order_by(Metrica.epoca)
    ).scalars().all()
    if metricas:
        ruta = os.path.join(carpeta, "metricas.csv")
        with open(ruta, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoca", "loss", "val_loss", "accuracy", "val_accuracy"])
            for m in metricas:
                writer.writerow([
                    m.epoca,
                    m.perdida_entrenamiento,
                    m.perdida_validacion,
                    m.precision_entrenamiento,
                    m.precision_validacion,
                ])
        print(f"  métricas CSV   ->{ruta}  ({len(metricas)} épocas)")


def descargar_imagenes(diag: ResultadoDiagnostico, carpeta: str) -> None:
    for attr, nombre in IMAGENES:
        datos = getattr(diag, attr)
        if datos:
            ruta = os.path.join(carpeta, nombre)
            with open(ruta, "wb") as f:
                f.write(datos)
            print(f"  {nombre:<30} ->{ruta}")


def descargar_todo(sesion: Session, modelo_id: int, carpeta: str) -> None:
    os.makedirs(carpeta, exist_ok=True)
    print(f"\n-- Modelo {modelo_id} -> '{carpeta}'")

    descargar_modelo(sesion, modelo_id, carpeta)

    diag = sesion.execute(
        select(ResultadoDiagnostico).where(ResultadoDiagnostico.modelo_id == modelo_id)
    ).scalar_one_or_none()
    if diag:
        descargar_imagenes(diag, carpeta)
    else:
        print(f"  [!] Sin diagnóstico gráfico para modelo {modelo_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url",    type=str, help="URL de la BD (sobreescribe .env)")
    parser.add_argument("--modelo", type=int, help="ID del modelo")
    parser.add_argument("--exp",    type=int, help="ID del experimento (descarga todos sus modelos)")
    args = parser.parse_args()

    db_url = args.url or os.getenv("base_datos")
    if not db_url:
        print("ERROR: Proporciona --url o define base_datos en el .env")
        sys.exit(1)

    engine = create_engine(db_url)
    sesion = Session(engine)
    try:
        if args.exp:
            modelos = sesion.execute(
                select(Modelo).where(Modelo.experimento_id == args.exp).order_by(Modelo.fecha)
            ).scalars().all()
            if not modelos:
                print(f"Sin modelos para experimento {args.exp}")
                return
            print(f"Experimento {args.exp}: {len(modelos)} modelo(s)")
            for m in modelos:
                carpeta = f"diagnosticos/exp{args.exp}_modelo{m.id}"
                descargar_todo(sesion, m.id, carpeta)

        elif args.modelo:
            descargar_todo(sesion, args.modelo, f"diagnosticos/modelo{args.modelo}")

        else:
            # Último modelo guardado
            ultimo = sesion.execute(
                select(Modelo).order_by(Modelo.id.desc()).limit(1)
            ).scalar_one_or_none()
            if ultimo is None:
                print("No hay modelos en la base de datos.")
                return
            descargar_todo(sesion, ultimo.id, f"diagnosticos/modelo{ultimo.id}")

    finally:
        sesion.close()


if __name__ == "__main__":
    main()
