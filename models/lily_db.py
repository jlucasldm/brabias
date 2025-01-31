import enum

from config import settings
from sqlalchemy import VARCHAR, TEXT, ForeignKey
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
import asyncio

from models.annotations import big_intpk, datetime_default_now, text, longtext, varchar

DATABASE_URL = (
    f"mysql+asyncmy://{settings.lily.USERNAME}:"
    f"{settings.lily.PASSWORD}@"
    f"{settings.lily.HOST}:"
    f"{settings.lily.PORT}/lily"
)

async_engine = create_async_engine(DATABASE_URL, echo=False, future=True)

LilyAsyncSession = async_sessionmaker(
    async_engine, expire_on_commit=False, autoflush=False
)


class CategoriaSocialEnum(enum.Enum):
    RACA = "raca"
    GENERO = "genero"
    RELIGIAO = "religiao"
    REGIAO = "regiao"


class BaseModel(DeclarativeBase):
    pass


class Roteiro(BaseModel):
    __tablename__ = "roteiro"

    id: Mapped[big_intpk]
    titulo: Mapped[str] = mapped_column(VARCHAR(250))
    conteudo: Mapped[longtext]
    data_publicacao: Mapped[datetime_default_now]
    data_criacao: Mapped[datetime_default_now]

    chunks: Mapped[list["RoteiroChunk"]] = relationship(
        "RoteiroChunk", back_populates="roteiro", lazy="subquery"
    )
    contexto: Mapped[list["Contexto"]] = relationship(
        "Contexto", back_populates="roteiro", lazy="subquery"
    )
    roteiro_contexto: Mapped[list["RoteiroContexto"]] = relationship(
        "RoteiroContexto", back_populates="roteiro", lazy="subquery"
    )


class RoteiroChunk(BaseModel):
    __tablename__ = "roteiro_chunk"

    id: Mapped[big_intpk]
    roteiro_id: Mapped[int] = mapped_column(ForeignKey("roteiro.id"))
    chunk: Mapped[str] = mapped_column(TEXT)
    data_criacao: Mapped[datetime_default_now]

    roteiro: Mapped["Roteiro"] = relationship(
        "Roteiro", back_populates="chunks", lazy="subquery"
    )
    contexto: Mapped["Contexto"] = relationship(
        "Contexto", back_populates="chunk", lazy="subquery"
    )
    roteiro_contexto: Mapped[list["RoteiroContexto"]] = relationship(
        "RoteiroContexto", back_populates="chunk", lazy="subquery"
    )


class RoteiroContexto(BaseModel):
    __tablename__ = "roteiro_contexto"

    id: Mapped[big_intpk]
    roteiro_id: Mapped[int] = mapped_column(ForeignKey("roteiro.id"))
    roteiro_chunk_id: Mapped[int] = mapped_column(ForeignKey("roteiro_chunk.id"))
    trecho: Mapped[text]
    termo_alvo: Mapped[varchar]
    categoria_social: Mapped[varchar]
    contexto_classificado: Mapped[varchar]
    par_contraste: Mapped[text]
    instancia_suprimida: Mapped[text]
    data_criacao: Mapped[datetime_default_now]

    roteiro: Mapped["Roteiro"] = relationship(
        "Roteiro", back_populates="roteiro_contexto", lazy="subquery"
    )

    chunk: Mapped["RoteiroChunk"] = relationship(
        "RoteiroChunk", back_populates="roteiro_contexto", lazy="subquery"
    )


class ToLDBR(BaseModel):
    __tablename__ = "ToLD-BR"

    id: Mapped[big_intpk]
    texto: Mapped[text]
    homofobia: Mapped[float]
    obscenidade: Mapped[float]
    insulto: Mapped[float]
    racismo: Mapped[float]
    misoginia: Mapped[float]
    xenofobia: Mapped[float]
    categoria_social: Mapped[CategoriaSocialEnum | None]

    annotations: Mapped[list["ToLDBRAnnotation"]] = relationship(
        "ToLDBRAnnotation", back_populates="toldbr", lazy="subquery"
    )


class ToLDBRAnnotation(BaseModel):
    __tablename__ = "ToLD-BR-annotation"

    id: Mapped[big_intpk]
    toldbr_id: Mapped[int] = mapped_column(ForeignKey("ToLD-BR.id"))
    original: Mapped[text]
    termo_alvo: Mapped[varchar]
    categoria_social: Mapped[varchar]
    contexto_classificado: Mapped[varchar]
    par_contraste: Mapped[text]
    instancia_suprimida: Mapped[text]
    justificativa: Mapped[text | None]

    toldbr: Mapped["ToLDBR"] = relationship(
        "ToLDBR", back_populates="annotations", lazy="subquery"
    )


class Contexto(BaseModel):
    __tablename__ = "contexto"

    id: Mapped[big_intpk]
    roteiro_id: Mapped[int] = mapped_column(ForeignKey("roteiro.id"))
    chunk_id: Mapped[int] = mapped_column(ForeignKey("roteiro_chunk.id"))
    contexto: Mapped[str] = mapped_column(TEXT)

    roteiro: Mapped["Roteiro"] = relationship(
        "Roteiro", back_populates="contexto", lazy="subquery"
    )
    chunk: Mapped["RoteiroChunk"] = relationship(
        "RoteiroChunk", back_populates="contexto", lazy="subquery"
    )


async def create_all():
    async with async_engine.begin() as conn:
        await conn.run_sync(BaseModel.metadata.create_all)


if __name__ == "__main__":
    asyncio.run(create_all())
