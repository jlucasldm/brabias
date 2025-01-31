import asyncio
import json

from sqlalchemy import select

from config import settings
from models.lily_db import (
    LilyAsyncSession,
    RoteiroChunk,
    RoteiroContexto,
)
from provider.open_ai_provider import OpenAiProvider

API_KEY = settings.openai.API_KEY


async def roteiro():
    openai = OpenAiProvider(model="gpt-4o-mini", max_tokens=10000, temperature=0)

    # pegar o primeiro roteiro do banco

    async with LilyAsyncSession() as session:
        print("[INFO] Getting RoteiroChunk")
        roteiro_chunk = (
            (await session.execute(select(RoteiroChunk).where(RoteiroChunk.id > 1263)))
            .scalars()
            .all()
        )

        system_prompt = openai.get_roteiro_system_prompt()

        for chunk in roteiro_chunk:
            print("[INFO] Annotating chunk with id: ", chunk.id)
            user_prompt = {"role": "user", "content": chunk.chunk}

            response = openai.get_response([system_prompt, user_prompt])

            text = response.choices[0].message.content

            text = text.replace("```json", "")
            text = text.replace("```", "")
            json_text = json.loads(text, strict=False)

            for resposta in json_text:
                print("[INFO] Annotating RoteiroContexto with id: ", chunk.id)
                roteiro_contexto = RoteiroContexto(
                    roteiro_id=chunk.roteiro_id,
                    roteiro_chunk_id=chunk.id,
                    trecho=resposta["trecho"],
                    termo_alvo=resposta["termo_alvo"],
                    categoria_social=resposta["categoria_social"],
                    contexto_classificado=resposta["contexto_classificado"],
                    par_contraste=resposta["par_contraste"],
                    instancia_suprimida=resposta["instancia_suprimida"],
                )

                session.add(roteiro_contexto)
                await session.commit()


if __name__ == "__main__":
    asyncio.run(roteiro())
