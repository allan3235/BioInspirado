import io
import queue
import random
import sys
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


def validar_parametros_ga(prob_mutacion: float, prob_cruce: float) -> None:
    if not (0.0 <= prob_mutacion <= 1.0):
        raise ValueError(f"PROB_MUTACION debe estar en [0.0, 1.0], valor recibido: {prob_mutacion}")
    if not (0.0 <= prob_cruce <= 1.0):
        raise ValueError(f"PROB_CRUCE debe estar en [0.0, 1.0], valor recibido: {prob_cruce}")


_OPTIMIZERS_MAP = {
    "adam":    tf.keras.optimizers.Adam,
    "sgd":     tf.keras.optimizers.SGD,
    "rmsprop": tf.keras.optimizers.RMSprop,
    "adamw":   tf.keras.optimizers.AdamW,
}


def build_model(individuo: dict, num_classes: int) -> tf.keras.Model:
    """
    CNN propia: dos bloques Conv→BN→MaxPool, GlobalAvgPool, Dense, Dropout, salida.
    Es el modelo principal que optimizan GA y PSO.
    """
    optimizer_name = individuo["optimizer"]
    if optimizer_name not in _OPTIMIZERS_MAP:
        raise ValueError(
            f"Optimizador '{optimizer_name}' no soportado. Válidos: {set(_OPTIMIZERS_MAP.keys())}"
        )

    filters = individuo.get("filters", 32)

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters, (3, 3), activation="relu",
                               input_shape=(224, 224, 3), padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(filters * 2, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(filters * 4, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(individuo["dense_units"], activation="relu"),
        tf.keras.layers.Dropout(individuo["dropout_rate"]),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])
    optimizer_cls = _OPTIMIZERS_MAP[optimizer_name]
    opt_kwargs = {"learning_rate": individuo["learning_rate"]}
    if optimizer_name == "adamw":
        opt_kwargs["weight_decay"] = 1e-4
    model.compile(
        optimizer=optimizer_cls(**opt_kwargs),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )
    return model


def build_model_vgg16(individuo: dict, num_classes: int) -> tf.keras.Model:
    """VGG16 transfer learning — modelo de comparación/baseline."""
    optimizer_name = individuo["optimizer"]
    if optimizer_name not in _OPTIMIZERS_MAP:
        raise ValueError(
            f"Optimizador '{optimizer_name}' no soportado. Válidos: {set(_OPTIMIZERS_MAP.keys())}"
        )

    base_model = tf.keras.applications.VGG16(
        weights="imagenet", include_top=False, input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(individuo["dense_units"], activation="relu"),
        tf.keras.layers.Dropout(individuo["dropout_rate"]),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])

    optimizer_cls = _OPTIMIZERS_MAP[optimizer_name]
    opt_kwargs = {"learning_rate": individuo["learning_rate"]}
    if optimizer_name == "adamw":
        opt_kwargs["weight_decay"] = 1e-4
    model.compile(
        optimizer=optimizer_cls(**opt_kwargs),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )
    return model


# ─── Persistencia ────────────────────────────────────────────────────────────

def _import_bd_models():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from ManejoDeDatos.basededatos import Experimento, Modelo, Metrica, Hiperparametro
    return Experimento, Modelo, Metrica, Hiperparametro


def _nueva_sesion():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from ManejoDeDatos.basededatos import obtener_sesion
    return obtener_sesion()


# ─── Cola de reintentos ───────────────────────────────────────────────────────

@dataclass
class _TareaGuardado:
    experimento_id: int
    individuo: dict
    historia: dict
    arquitectura: str
    modelo_bytes: Optional[bytes] = None


_cola_reintentos: queue.Queue = queue.Queue()
_worker_iniciado = False
_worker_lock = threading.Lock()


def _insertar_en_bd(sesion, tarea: _TareaGuardado) -> int:
    """Lógica de inserción compartida entre la ruta normal y los reintentos."""
    _, Modelo, Metrica, Hiperparametro = _import_bd_models()

    modelo = Modelo(
        experimento_id=tarea.experimento_id,
        arquitectura=tarea.arquitectura,
        fecha=datetime.utcnow(),
    )
    if tarea.modelo_bytes:
        modelo.archivo = tarea.modelo_bytes
        modelo.formato = "keras"
    sesion.add(modelo)
    sesion.flush()

    sesion.add(Hiperparametro(
        modelo_id=modelo.id,
        tasa_aprendizaje=tarea.individuo.get("learning_rate"),
        tamano_lote=tarea.individuo.get("batch_size"),
        optimizador=tarea.individuo.get("optimizer"),
        epocas=tarea.individuo.get("epochs"),
        aumento_datos=tarea.individuo.get("use_augmentation"),
        neuronas_densas=tarea.individuo.get("dense_units"),
        tasa_dropout=tarea.individuo.get("dropout_rate"),
        filtros_conv=tarea.individuo.get("filters"),
    ))

    hist = tarea.historia
    for epoch_idx in range(len(hist.get("loss", []))):
        sesion.add(Metrica(
            modelo_id=modelo.id,
            epoca=epoch_idx + 1,
            perdida_entrenamiento=_safe_get(hist, "loss", epoch_idx),
            perdida_validacion=_safe_get(hist, "val_loss", epoch_idx),
            precision_entrenamiento=_safe_get(hist, "accuracy", epoch_idx),
            precision_validacion=_safe_get(hist, "val_accuracy", epoch_idx),
        ))

    sesion.commit()
    return modelo.id


def _worker_reintentos():
    pendientes: list = []
    while True:
        time.sleep(60)
        while True:
            try:
                pendientes.append(_cola_reintentos.get_nowait())
            except queue.Empty:
                break

        if not pendientes:
            continue

        etiqueta = "modelo(s)" if pendientes else ""
        print(f"[reintento BD] {len(pendientes)} {etiqueta} pendiente(s) — intentando guardar...")
        exitosas = []
        for i, tarea in enumerate(pendientes):
            sesion = None
            try:
                sesion = _nueva_sesion()
                modelo_id = _insertar_en_bd(sesion, tarea)
                exitosas.append(i)
                tipo = "mejor modelo" if tarea.modelo_bytes else "individuo"
                print(f"[reintento BD] OK ({tipo}) — experimento_id={tarea.experimento_id} modelo_id={modelo_id}")
            except Exception as e:
                tipo = "mejor modelo" if tarea.modelo_bytes else "individuo"
                print(f"[reintento BD] Error ({tipo}) experimento_id={tarea.experimento_id} — reintentando en 60s: {e}")
                if sesion:
                    try:
                        sesion.rollback()
                    except Exception:
                        pass
            finally:
                if sesion:
                    try:
                        sesion.close()
                    except Exception:
                        pass

        for i in sorted(exitosas, reverse=True):
            pendientes.pop(i)


def _iniciar_worker():
    global _worker_iniciado
    with _worker_lock:
        if not _worker_iniciado:
            t = threading.Thread(target=_worker_reintentos, daemon=True, name="bd-retry-worker")
            t.start()
            _worker_iniciado = True


def _encolar(tarea: _TareaGuardado):
    _cola_reintentos.put(tarea)
    _iniciar_worker()
    tipo = "mejor modelo" if tarea.modelo_bytes else "individuo"
    print(f"[guardar BD] Encolado para reintento ({tipo}) — experimento_id={tarea.experimento_id}")


# ─── API pública de persistencia ─────────────────────────────────────────────

def crear_experimento(sesion, nombre: str) -> int:
    Experimento, _, _, _ = _import_bd_models()
    experimento = Experimento(nombre=nombre, fecha=datetime.utcnow())
    sesion.add(experimento)
    sesion.commit()
    sesion.refresh(experimento)
    return experimento.id


def serializar_modelo(keras_model: tf.keras.Model) -> Optional[bytes]:
    """Serializa un modelo Keras a bytes .keras. Retorna None si falla."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            keras_model.save(tmp_path)
            with open(tmp_path, "rb") as f:
                return f.read()
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
    except Exception as e:
        print(f"[serializar_modelo] Error: {e}")
        return None


def guardar_individuo_bd(sesion, experimento_id: int, individuo: dict, history,
                         arquitectura: str = "CNN",
                         modelo_bytes: Optional[bytes] = None) -> int:
    hist = history.history if hasattr(history, "history") else history
    tarea = _TareaGuardado(
        experimento_id=experimento_id,
        individuo=dict(individuo),
        historia=dict(hist),
        arquitectura=arquitectura,
        modelo_bytes=modelo_bytes,
    )
    if sesion is None:
        _encolar(tarea)
        return 0
    try:
        return _insertar_en_bd(sesion, tarea)
    except Exception as e:
        print(f"[guardar_individuo_bd] Error: {e}")
        try:
            sesion.rollback()
        except Exception:
            pass
        # Reintento inmediato con sesión nueva (la original puede estar muerta)
        try:
            sesion_nueva = _nueva_sesion()
            resultado = _insertar_en_bd(sesion_nueva, tarea)
            sesion_nueva.close()
            return resultado
        except Exception as e2:
            print(f"[guardar_individuo_bd] Reintento fallido: {e2}")
            _encolar(tarea)
            return 0


def _safe_get(hist: dict, key: str, idx: int):
    vals = hist.get(key, [])
    return float(vals[idx]) if idx < len(vals) else None


# ─── Preparación de datasets por individuo ───────────────────────────────────

def preparar_datasets(individuo: dict, train_ds, val_ds):
    """
    Aplica batch_size y use_augmentation del individuo a los datasets crudos
    retornados por load_dataset_bd (sin batch, sin augmentation).
    """
    batch_size = individuo.get("batch_size", 32)
    use_aug    = individuo.get("use_augmentation", True)
    AUTOTUNE   = tf.data.AUTOTUNE

    ds_train = train_ds
    if use_aug:
        augment = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomTranslation(0.1, 0.1),
        ])
        ds_train = ds_train.map(
            lambda x, y: (augment(x, training=True), y),
            num_parallel_calls=AUTOTUNE,
        )

    ds_train = ds_train.batch(batch_size).prefetch(AUTOTUNE)
    ds_val   = val_ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds_train, ds_val


# ─── Evaluación ──────────────────────────────────────────────────────────────

def evaluate_individual(individuo, train_ds, val_ds, num_classes, experimento_id, sesion) -> float:
    try:
        ds_train, ds_val = preparar_datasets(individuo, train_ds, val_ds)
        model = build_model(individuo, num_classes)
        history = model.fit(
            ds_train,
            validation_data=ds_val,
            epochs=individuo["epochs"],
            verbose=2,
        )
        try:
            guardar_individuo_bd(sesion, experimento_id, individuo, history)
        except Exception as bd_err:
            print(f"[evaluate_individual] Advertencia BD: {bd_err}")
        return float(max(history.history["val_accuracy"]))
    except Exception as err:
        print(f"[evaluate_individual] Error: {err}")
        return 0.0


# ─── Algoritmo Genético ───────────────────────────────────────────────────────

class GeneticAlgorithm:

    def __init__(self, search_space, poblacion_size, num_generaciones,
                 prob_mutacion, prob_cruce, seed=42):
        self.search_space = search_space
        self.poblacion_size = poblacion_size
        self.num_generaciones = num_generaciones
        self.prob_mutacion = prob_mutacion
        self.prob_cruce = prob_cruce
        self.seed = seed
        self._rng = random.Random(seed)

    def inicializar_poblacion(self):
        rng = random.Random(self.seed)
        return [
            {key: rng.choice(vals) for key, vals in self.search_space.items()}
            for _ in range(self.poblacion_size)
        ]

    def evaluar_poblacion(self, poblacion, train_ds, val_ds, num_classes, experimento_id, sesion):
        return [
            evaluate_individual(ind, train_ds, val_ds, num_classes, experimento_id, sesion)
            for ind in poblacion
        ]

    def seleccion_torneo(self, poblacion, fitness_list, k=2):
        indices = self._rng.sample(range(len(poblacion)), min(k, len(poblacion)))
        best_idx = max(indices, key=lambda i: fitness_list[i])
        return dict(poblacion[best_idx])

    def cruce_uniforme(self, padre1, padre2):
        hijo1, hijo2 = {}, {}
        for key in padre1:
            if self._rng.random() < self.prob_cruce:
                hijo1[key] = padre1[key]; hijo2[key] = padre2[key]
            else:
                hijo1[key] = padre2[key]; hijo2[key] = padre1[key]
        return hijo1, hijo2

    def mutacion(self, individuo):
        return {
            key: (self._rng.choice(self.search_space[key])
                  if self._rng.random() < self.prob_mutacion else val)
            for key, val in individuo.items()
        }

    def run(self, train_ds, val_ds, num_classes, experimento_id, sesion):
        poblacion = self.inicializar_poblacion()
        fitness_list = self.evaluar_poblacion(
            poblacion, train_ds, val_ds, num_classes, experimento_id, sesion
        )

        mejor_idx = int(np.argmax(fitness_list))
        mejor_individuo = dict(poblacion[mejor_idx])
        mejor_fitness = fitness_list[mejor_idx]
        fitness_history = [mejor_fitness]

        for _ in range(self.num_generaciones - 1):
            nueva_poblacion = [dict(mejor_individuo)]
            while len(nueva_poblacion) < self.poblacion_size:
                p1 = self.seleccion_torneo(poblacion, fitness_list)
                p2 = self.seleccion_torneo(poblacion, fitness_list)
                h1, h2 = self.cruce_uniforme(p1, p2)
                nueva_poblacion.append(self.mutacion(h1))
                if len(nueva_poblacion) < self.poblacion_size:
                    nueva_poblacion.append(self.mutacion(h2))

            poblacion = nueva_poblacion[:self.poblacion_size]
            fitness_list = self.evaluar_poblacion(
                poblacion, train_ds, val_ds, num_classes, experimento_id, sesion
            )

            gen_mejor_idx = int(np.argmax(fitness_list))
            if fitness_list[gen_mejor_idx] > mejor_fitness:
                mejor_fitness = fitness_list[gen_mejor_idx]
                mejor_individuo = dict(poblacion[gen_mejor_idx])
            fitness_history.append(mejor_fitness)

        return mejor_individuo, fitness_history


# ─── PSO ─────────────────────────────────────────────────────────────────────

class ParticleSwarmOptimizer:

    def __init__(self, search_space, poblacion_size, num_generaciones,
                 w_inercia=0.5, c1_cognitivo=1.5, c2_social=1.5, seed=42):
        self.search_space = search_space
        self.poblacion_size = poblacion_size
        self.num_generaciones = num_generaciones
        self.w_inercia = w_inercia
        self.c1_cognitivo = c1_cognitivo
        self.c2_social = c2_social
        self.seed = seed
        self._rng = random.Random(seed)

    def inicializar_enjambre(self):
        rng = random.Random(self.seed)
        enjambre = []
        for _ in range(self.poblacion_size):
            pos = {key: rng.choice(vals) for key, vals in self.search_space.items()}
            vel = {key: rng.uniform(-1.0, 1.0) for key in self.search_space}
            enjambre.append({"pos": dict(pos), "vel": dict(vel),
                             "pbest_pos": dict(pos), "pbest_fit": 0.0})
        return enjambre

    def discretizar_posicion(self, pos_continua):
        discreto = {}
        for key, val in pos_continua.items():
            opciones = self.search_space[key]
            try:
                discreto[key] = min(opciones, key=lambda v: abs(float(v) - float(val)))
            except (TypeError, ValueError):
                discreto[key] = opciones[0]
        return discreto

    def actualizar_velocidad(self, particula, gbest_pos):
        nueva_vel = {}
        for key in self.search_space:
            v = particula["vel"][key]
            try:
                pos   = float(particula["pos"][key])
                pbest = float(particula["pbest_pos"][key])
                gbest = float(gbest_pos[key])
                r1, r2 = self._rng.random(), self._rng.random()
                nueva_vel[key] = (self.w_inercia * v
                                  + self.c1_cognitivo * r1 * (pbest - pos)
                                  + self.c2_social   * r2 * (gbest - pos))
            except (TypeError, ValueError):
                nueva_vel[key] = 0.0
        return nueva_vel

    def actualizar_posicion(self, particula):
        nueva_pos_continua = {}
        for key in self.search_space:
            try:
                nueva_pos_continua[key] = float(particula["pos"][key]) + particula["vel"][key]
            except (TypeError, ValueError):
                nueva_pos_continua[key] = particula["pos"][key]
        return self.discretizar_posicion(nueva_pos_continua)

    def run(self, train_ds, val_ds, num_classes, experimento_id, sesion):
        enjambre = self.inicializar_enjambre()

        gbest_pos = None
        gbest_fit = -1.0
        for particula in enjambre:
            fit = evaluate_individual(
                particula["pos"], train_ds, val_ds, num_classes, experimento_id, sesion
            )
            particula["pbest_fit"] = fit
            if fit > gbest_fit:
                gbest_fit = fit
                gbest_pos = dict(particula["pos"])

        fitness_history = [gbest_fit]

        for _ in range(self.num_generaciones - 1):
            for particula in enjambre:
                particula["vel"] = self.actualizar_velocidad(particula, gbest_pos)
                nueva_pos = self.actualizar_posicion(particula)
                particula["pos"] = nueva_pos

                fit = evaluate_individual(
                    nueva_pos, train_ds, val_ds, num_classes, experimento_id, sesion
                )
                if fit > particula["pbest_fit"]:
                    particula["pbest_fit"] = fit
                    particula["pbest_pos"] = dict(nueva_pos)
                if fit > gbest_fit:
                    gbest_fit = fit
                    gbest_pos = dict(nueva_pos)

            fitness_history.append(gbest_fit)

        return gbest_pos, fitness_history


# ─── Visualización ───────────────────────────────────────────────────────────

def plot_fitness_history(fitness_history: list, algoritmo: str) -> None:
    generaciones = list(range(1, len(fitness_history) + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(generaciones, fitness_history, marker="o", linewidth=2, color="steelblue")
    plt.xlabel("Generación / Iteración")
    plt.ylabel("Fitness máximo (val_accuracy)")
    plt.title(f"Evolución del Fitness — {algoritmo}")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_training_curves(history) -> None:
    hist = history.history if hasattr(history, "history") else history
    epochs = range(1, len(hist.get("loss", [])) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, hist.get("accuracy", []),     label="Entrenamiento", color="steelblue")
    ax1.plot(epochs, hist.get("val_accuracy", []), label="Validación",    color="orange")
    ax1.set_xlabel("Época"); ax1.set_ylabel("Accuracy")
    ax1.set_title("Accuracy por época"); ax1.legend(); ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.plot(epochs, hist.get("loss", []),     label="Entrenamiento", color="steelblue")
    ax2.plot(epochs, hist.get("val_loss", []), label="Validación",    color="orange")
    ax2.set_xlabel("Época"); ax2.set_ylabel("Loss")
    ax2.set_title("Loss por época"); ax2.legend(); ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(model, val_ds, class_names: list) -> None:
    if len(class_names) != 6:
        raise ValueError(f"Se esperaban 6 clases, se detectaron {len(class_names)}.")

    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    y_true, y_pred = [], []
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1).tolist())
        y_true.extend(np.argmax(labels.numpy(), axis=1).tolist())

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicción"); plt.ylabel("Real")
    plt.title("Matriz de Confusión — Mejor Individuo")
    plt.tight_layout()
    plt.show()
