import os

import pandas as pd
import asyncio
from pathlib import Path

from spacy.cli import download
from sqlalchemy import select

from models.lily_db import LilyAsyncSession, Roteiro, RoteiroChunk, ToLDBR

import spacy

# download("pt_core_news_sm")
nlp = spacy.load("pt_core_news_sm")


def chunk_text(text, max_tokens=1000):
    doc = nlp(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sent in doc.sents:
        sent_length = len(sent)
        if current_length + sent_length > max_tokens:
            chunks.append(" ".join([str(s) for s in current_chunk]))
            current_chunk = []
            current_length = 0
        current_chunk.append(sent)
        current_length += sent_length

    if current_chunk:
        chunks.append(" ".join([str(s) for s in current_chunk]))

    return chunks


async def build_told_br():
    async with LilyAsyncSession() as session:
        df = pd.read_csv("ToLD-BR.csv")

        for index, row in df.iterrows():
            contexto = ToLDBR(
                texto=row["text"],
                homofobia=row["homophobia"],
                obscenidade=row["obscene"],
                insulto=row["insult"],
                racismo=row["racism"],
                misoginia=row["misogyny"],
                xenofobia=row["xenophobia"],
            )

            session.add(contexto)
            await session.commit()


async def build_roteiros():
    async with LilyAsyncSession() as session:
        path = Path("../roteiros")

        for file in path.iterdir():
            with open(file, "r") as f:
                roteiro = f.read()

                file_name = os.path.basename(file)
                file = os.path.splitext(file_name)

                roteiro = Roteiro(titulo=file[0], conteudo=roteiro)

                session.add(roteiro)
                await session.commit()


async def build_chunks():
    async with LilyAsyncSession() as session:
        roteiros = (await session.execute(select(Roteiro))).scalars().all()

        for roteiro in roteiros:
            chunks = chunk_text(roteiro.conteudo)

            for chunk in chunks:
                roteiro_chunk = RoteiroChunk(roteiro_id=roteiro.id, chunk=chunk)
                session.add(roteiro_chunk)

            await session.commit()


if __name__ == "__main__":
    asyncio.run(build_told_br())
