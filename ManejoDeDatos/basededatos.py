import os
from datetime import datetime
from typing import List, Optional

from dotenv import load_dotenv
from sqlalchemy import create_engine, ForeignKey, String, DateTime, LargeBinary, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, Session

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

engine = create_engine(os.getenv("base_datos"))


class Base(DeclarativeBase):
    pass


class Experimento(Base):
    __tablename__ = "experimentos"

    id: Mapped[int] = mapped_column(primary_key=True)
    nombre: Mapped[str] = mapped_column(String(100), nullable=False)
    descripcion: Mapped[Optional[str]] = mapped_column(String(255))
    fecha: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    modelos: Mapped[List["Modelo"]] = relationship(back_populates="experimento")


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


class Imagen(Base):
    __tablename__ = "imagenes"

    id: Mapped[int] = mapped_column(primary_key=True)
    imagen: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    enfermedad: Mapped[str] = mapped_column(String(100), nullable=False)
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


if __name__ == "__main__":
    iniciar_bd()
    print("Tablas creadas correctamente")
