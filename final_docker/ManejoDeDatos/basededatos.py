import os
from datetime import datetime, timedelta
from typing import List, Optional

from dotenv import load_dotenv
from sqlalchemy import create_engine, ForeignKey, String, DateTime, LargeBinary, Text, JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, Session

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

engine = create_engine(
    os.getenv("base_datos"),
    pool_pre_ping=True,
    pool_recycle=1800,
)


class Base(DeclarativeBase):
    pass


class Experimento(Base):
    __tablename__ = "experimentos"

    id: Mapped[int] = mapped_column(primary_key=True)
    nombre: Mapped[str] = mapped_column(String(100), nullable=False)
    descripcion: Mapped[Optional[str]] = mapped_column(String(255))
    fecha: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    modelos: Mapped[List["Modelo"]] = relationship(back_populates="experimento")
    tareas: Mapped[List["TareaEvaluacion"]] = relationship(back_populates="experimento")


class Modelo(Base):
    __tablename__ = "modelos"

    id: Mapped[int] = mapped_column(primary_key=True)
    experimento_id: Mapped[int] = mapped_column(ForeignKey("experimentos.id"))
    arquitectura: Mapped[str] = mapped_column(String(100), nullable=False)
    archivo: Mapped[Optional[bytes]] = mapped_column(LargeBinary)
    formato: Mapped[Optional[str]] = mapped_column(String(10))
    fecha: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    experimento: Mapped["Experimento"] = relationship(back_populates="modelos")
    metricas: Mapped[List["Metrica"]] = relationship(back_populates="modelo")
    hiperparametros: Mapped[Optional["Hiperparametro"]] = relationship(back_populates="modelo")
    diagnostico: Mapped[Optional["ResultadoDiagnostico"]] = relationship(back_populates="modelo")


class Metrica(Base):
    __tablename__ = "metricas"

    id: Mapped[int] = mapped_column(primary_key=True)
    modelo_id: Mapped[int] = mapped_column(ForeignKey("modelos.id"))
    epoca: Mapped[int] = mapped_column(nullable=False)
    perdida_entrenamiento: Mapped[Optional[float]]
    perdida_validacion: Mapped[Optional[float]]
    precision_entrenamiento: Mapped[Optional[float]]
    precision_validacion: Mapped[Optional[float]]

    modelo: Mapped["Modelo"] = relationship(back_populates="metricas")


class Hiperparametro(Base):
    __tablename__ = "hiperparametros"

    id: Mapped[int] = mapped_column(primary_key=True)
    modelo_id: Mapped[int] = mapped_column(ForeignKey("modelos.id"))
    tasa_aprendizaje: Mapped[Optional[float]]
    tamano_lote: Mapped[Optional[int]]
    optimizador: Mapped[Optional[str]] = mapped_column(String(50))
    epocas: Mapped[Optional[int]]
    aumento_datos: Mapped[Optional[bool]]
    neuronas_densas: Mapped[Optional[int]]
    neuronas_finales: Mapped[Optional[int]]
    tasa_dropout: Mapped[Optional[float]]
    filtros_conv: Mapped[Optional[int]]

    modelo: Mapped["Modelo"] = relationship(back_populates="hiperparametros")


class ResultadoDiagnostico(Base):
    __tablename__ = "resultados_diagnostico"

    id: Mapped[int] = mapped_column(primary_key=True)
    modelo_id: Mapped[int] = mapped_column(ForeignKey("modelos.id"))
    curva_perdida: Mapped[Optional[bytes]] = mapped_column(LargeBinary)
    curva_precision: Mapped[Optional[bytes]] = mapped_column(LargeBinary)
    matriz_confusion: Mapped[Optional[bytes]] = mapped_column(LargeBinary)
    curva_roc: Mapped[Optional[bytes]] = mapped_column(LargeBinary)
    formato: Mapped[str] = mapped_column(String(10), default="png")
    fecha: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    modelo: Mapped["Modelo"] = relationship(back_populates="diagnostico")


class EscenarioBusqueda(Base):
    """
    Cola distribuida. Cada contenedor Docker toma un escenario pendiente,
    lo ejecuta y sube el resultado. El campo `estado` es el semáforo.
    """
    __tablename__ = "escenarios_busqueda"

    id: Mapped[int] = mapped_column(primary_key=True)
    nombre: Mapped[str] = mapped_column(String(150), nullable=False)

    algoritmo: Mapped[str] = mapped_column(String(10), nullable=False)
    seed: Mapped[int] = mapped_column(default=42)
    poblacion_size: Mapped[int] = mapped_column(nullable=False)
    num_generaciones: Mapped[int] = mapped_column(nullable=False)

    prob_mutacion: Mapped[Optional[float]]
    prob_cruce: Mapped[Optional[float]]

    w_inercia: Mapped[Optional[float]]
    c1_cognitivo: Mapped[Optional[float]]
    c2_social: Mapped[Optional[float]]

    search_space_json: Mapped[dict] = mapped_column(JSON, nullable=False)

    estado: Mapped[str] = mapped_column(String(20), default="pendiente")
    nodo_id: Mapped[Optional[str]] = mapped_column(String(150))
    fecha_inicio: Mapped[Optional[datetime]] = mapped_column(DateTime)
    fecha_fin: Mapped[Optional[datetime]] = mapped_column(DateTime)

    experimento_id: Mapped[Optional[int]] = mapped_column(ForeignKey("experimentos.id"), nullable=True)
    mejor_fitness: Mapped[Optional[float]]
    mejor_individuo_json: Mapped[Optional[dict]] = mapped_column(JSON)
    fitness_history_json: Mapped[Optional[list]] = mapped_column(JSON)
    mensaje_error: Mapped[Optional[str]] = mapped_column(Text)


class TareaEvaluacion(Base):
    __tablename__ = "tareas_evaluacion"

    id: Mapped[int] = mapped_column(primary_key=True)
    experimento_id: Mapped[int] = mapped_column(ForeignKey("experimentos.id"))
    algoritmo: Mapped[str] = mapped_column(String(10), nullable=False)
    generacion: Mapped[Optional[int]]
    individuo_idx: Mapped[Optional[int]]
    hiperparametros_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    estado: Mapped[str] = mapped_column(String(20), default="pendiente")
    nodo_id: Mapped[Optional[str]] = mapped_column(String(100))
    fitness: Mapped[Optional[float]]
    fecha_inicio: Mapped[Optional[datetime]] = mapped_column(DateTime)
    fecha_fin: Mapped[Optional[datetime]] = mapped_column(DateTime)
    mensaje_error: Mapped[Optional[str]] = mapped_column(Text)

    experimento: Mapped["Experimento"] = relationship(back_populates="tareas")


class Imagen(Base):
    __tablename__ = "imagenes"

    id: Mapped[int] = mapped_column(primary_key=True)
    imagen: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    enfermedad: Mapped[str] = mapped_column(String(100), nullable=False)
    nombre_archivo: Mapped[Optional[str]] = mapped_column(String(255))
    fecha: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    usos: Mapped[List["ImagenUso"]] = relationship(back_populates="imagen")


class ImagenUso(Base):
    __tablename__ = "imagenes_uso"

    id: Mapped[int] = mapped_column(primary_key=True)
    imagen_id: Mapped[int] = mapped_column(ForeignKey("imagenes.id"))
    uso: Mapped[str] = mapped_column(String(50), nullable=False)

    imagen: Mapped["Imagen"] = relationship(back_populates="usos")


def iniciar_bd():
    Base.metadata.create_all(engine)


def obtener_sesion() -> Session:
    return Session(engine)


def cargar_escenarios(sesion: Session, escenarios: list) -> None:
    for cfg in escenarios:
        sesion.add(EscenarioBusqueda(**cfg))
    sesion.commit()


def tomar_escenario(sesion: Session, nodo_id: str) -> Optional["EscenarioBusqueda"]:
    """SELECT FOR UPDATE SKIP LOCKED — dos contenedores nunca toman el mismo escenario."""
    from sqlalchemy import select
    stmt = (
        select(EscenarioBusqueda)
        .where(EscenarioBusqueda.estado == "pendiente")
        .limit(1)
        .with_for_update(skip_locked=True)
    )
    escenario = sesion.execute(stmt).scalar_one_or_none()
    if escenario is not None:
        escenario.estado = "en_proceso"
        escenario.nodo_id = nodo_id
        escenario.fecha_inicio = datetime.utcnow()
        sesion.commit()
    return escenario


def completar_escenario(sesion: Session, escenario_id: int, experimento_id: int,
                        mejor_fitness: float, mejor_individuo: dict,
                        fitness_history: Optional[list] = None) -> None:
    escenario = sesion.get(EscenarioBusqueda, escenario_id)
    if escenario:
        escenario.estado = "completado"
        escenario.experimento_id = experimento_id
        escenario.mejor_fitness = mejor_fitness
        escenario.mejor_individuo_json = mejor_individuo
        escenario.fitness_history_json = fitness_history
        escenario.fecha_fin = datetime.utcnow()
        sesion.commit()


def fallar_escenario(sesion: Session, escenario_id: int, mensaje: str) -> None:
    escenario = sesion.get(EscenarioBusqueda, escenario_id)
    if escenario:
        escenario.estado = "error"
        escenario.mensaje_error = mensaje
        escenario.fecha_fin = datetime.utcnow()
        sesion.commit()


def reintentar_escenarios_colgados(sesion: Session, timeout_minutos: int = 120) -> int:
    """Rescata escenarios 'en_proceso' que llevan más de timeout_minutos sin terminar."""
    from sqlalchemy import select

    limite = datetime.utcnow() - timedelta(minutes=timeout_minutos)
    stmt = (
        select(EscenarioBusqueda)
        .where(
            EscenarioBusqueda.estado == "en_proceso",
            EscenarioBusqueda.fecha_inicio < limite,
        )
        .with_for_update(skip_locked=True)
    )
    colgados = sesion.execute(stmt).scalars().all()
    for escenario in colgados:
        escenario.estado = "pendiente"
        escenario.mensaje_error = (
            f"Reencolado: nodo '{escenario.nodo_id}' no respondió en "
            f"{timeout_minutos} min (inicio: {escenario.fecha_inicio})"
        )
        escenario.nodo_id = None
        escenario.fecha_inicio = None
    if colgados:
        sesion.commit()
    return len(colgados)


def consultar_mejores_escenarios(sesion: Session, limite: int = 10) -> list:
    from sqlalchemy import select
    stmt = (
        select(EscenarioBusqueda)
        .where(EscenarioBusqueda.estado == "completado")
        .order_by(EscenarioBusqueda.mejor_fitness.desc())
        .limit(limite)
    )
    return sesion.execute(stmt).scalars().all()


def guardar_diagnostico(sesion: Session, modelo_id: int,
                        curva_perdida: Optional[bytes] = None,
                        curva_precision: Optional[bytes] = None,
                        matriz_confusion: Optional[bytes] = None,
                        curva_roc: Optional[bytes] = None,
                        formato: str = "png") -> "ResultadoDiagnostico":
    diag = ResultadoDiagnostico(
        modelo_id=modelo_id,
        curva_perdida=curva_perdida,
        curva_precision=curva_precision,
        matriz_confusion=matriz_confusion,
        curva_roc=curva_roc,
        formato=formato,
    )
    sesion.add(diag)
    sesion.commit()
    return diag


def encolar_tareas(sesion: Session, experimento_id: int, algoritmo: str,
                   poblacion: list, generacion: int) -> None:
    for idx, individuo in enumerate(poblacion):
        sesion.add(TareaEvaluacion(
            experimento_id=experimento_id,
            algoritmo=algoritmo,
            generacion=generacion,
            individuo_idx=idx,
            hiperparametros_json=individuo,
            estado="pendiente",
        ))
    sesion.commit()
