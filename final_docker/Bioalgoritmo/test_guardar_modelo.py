"""
Prueba de 1 época: verifica que el modelo se serializa y guarda en la BD.
Corre con:
    docker-compose -f docker-compose.test.yml run --rm worker python test_guardar_modelo.py
"""

import sys
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # CPU para prueba rápida

# Carga el .env raíz antes que cualquier import de basededatos,
# así la cadena de BD local tiene prioridad sobre la del .env de Docker.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv(os.path.join(_REPO_ROOT, ".env"), override=True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ManejoDeDatos.basededatos import iniciar_bd, obtener_sesion, Modelo
from ManejoDeDatos.data_loader import load_dataset_bd
from bio_optimizer_core import (
    build_model,
    preparar_datasets,
    serializar_modelo,
    guardar_individuo_bd,
    crear_experimento,
)


INDIVIDUO_PRUEBA = {
    "epochs":        1,
    "batch_size":    16,
    "learning_rate": 1e-3,
    "optimizer":     "adam",
    "dense_units":   64,
    "dropout_rate":  0.3,
    "filters":       8,
    "use_augmentation": False,
}


def ok(msg):  print(f"  [OK]   {msg}")
def fallo(msg, exc=None):
    print(f"  [FALLO] {msg}")
    if exc:
        print(f"          {exc}")


def main():
    print("\n=== TEST: guardar modelo 1 época ===\n")

    # 1. BD
    try:
        iniciar_bd()
        sesion = obtener_sesion()
        ok("Conexión a BD")
    except Exception as e:
        fallo("No se pudo conectar a la BD", e)
        sys.exit(1)

    # 2. Dataset
    try:
        from ManejoDeDatos.data_loader import get_n_train
        train_ds, val_ds, class_names = load_dataset_bd()
        n_train = get_n_train()
        num_clases = len(class_names)
        ok(f"Dataset cargado — {n_train} imágenes train, {num_clases} clases: {class_names}")
        if n_train == 0:
            fallo("BD no tiene imágenes de entrenamiento. Carga las imágenes primero.")
            sys.exit(1)
    except Exception as e:
        fallo("Error cargando dataset desde BD", e)
        sys.exit(1)

    # 3. Preparar datasets y modelo
    try:
        ds_train, ds_val, steps = preparar_datasets(INDIVIDUO_PRUEBA, train_ds, val_ds)
        ok(f"Datasets preparados — steps_per_epoch={steps}")
        modelo = build_model(INDIVIDUO_PRUEBA, num_clases)
        ok("Modelo construido")
    except Exception as e:
        fallo("Error preparando datasets o construyendo modelo", e)
        sys.exit(1)

    # 4. Entrenamiento 1 época
    try:
        history = modelo.fit(
            ds_train,
            validation_data=ds_val,
            epochs=1,
            steps_per_epoch=steps,
            verbose=1,
        )
        val_acc = history.history["val_accuracy"][-1]
        ok(f"Entrenamiento completado — val_accuracy época final: {val_acc:.4f}")
    except Exception as e:
        fallo("Error en model.fit()", e)
        sys.exit(1)

    # 5. Serializar
    try:
        modelo_bytes = serializar_modelo(modelo)
        if modelo_bytes is None:
            fallo("serializar_modelo() devolvió None")
            sys.exit(1)
        ok(f"Modelo serializado — {len(modelo_bytes) / 1024:.1f} KB")
    except Exception as e:
        fallo("Error serializando modelo", e)
        sys.exit(1)

    # 6. Guardar en BD
    try:
        experimento_id = crear_experimento(sesion, "TEST_1_EPOCA")
        modelo_id = guardar_individuo_bd(
            sesion, experimento_id, INDIVIDUO_PRUEBA, history,
            modelo_bytes=modelo_bytes,
        )
        if modelo_id == 0:
            fallo("guardar_individuo_bd() devolvió 0 (encolado para reintento — fallo en BD)")
            sys.exit(1)
        ok(f"Guardado en BD — modelo_id={modelo_id}, experimento_id={experimento_id}")
    except Exception as e:
        fallo("Error guardando en BD", e)
        sys.exit(1)

    # 7. Verificar que el binario quedó en la BD
    try:
        registro = sesion.get(Modelo, modelo_id)
        if registro is None:
            fallo(f"No se encontró modelo_id={modelo_id} en la BD")
            sys.exit(1)
        if not registro.archivo:
            fallo("El campo 'archivo' está vacío — el binario NO se guardó")
            sys.exit(1)
        ok(f"Binario confirmado en BD — {len(registro.archivo) / 1024:.1f} KB en columna 'archivo'")
    except Exception as e:
        fallo("Error verificando BD", e)
        sys.exit(1)

    sesion.close()
    print("\n=== RESULTADO: TODO CORRECTO — el modelo se guarda bien ===\n")


if __name__ == "__main__":
    main()
