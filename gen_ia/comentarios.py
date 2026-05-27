import os
import time
from dotenv import load_dotenv
import mlflow
from openai import OpenAI, Client
from typing import List, Literal
from pydantic import BaseModel, Field

load_dotenv()

mlflow.set_tracking_uri("sqlite:///../mlflow.db")
mlflow.set_experiment("Análise de Comentários")
mlflow.openai.autolog()

# Definição do LLM client
CLIENT = OpenAI(
    api_key=os.getenv("API_KEY_OPEN_ROUTER"),
    base_url="https://openrouter.ai/api/v1"
)
MODELO = "openai/gpt-oss-120b:free"


# 1. Definição do esquema de dados (Schema) de forma estrita usando Pydantic
class AnaliseComentario(BaseModel):
    sentimento: Literal["positivo", "negativo", "neutro"] = Field(
        description="O sentimento geral extraído do comentário."
    )
    topicos: List[str] = Field(
        description="Lista de tópicos principais abordados no comentário."
    )
    resumo: str = Field(
        description="Um resumo curto do comentário."
    )


def analisar_comentario(texto: str, client: Client, modelo: str) -> dict:
    prompt = f"Analise o seguinte comentário: '{texto}'"

    try:
        # 2. Utilizamos o método .parse() da API (Structured Outputs)
        resposta = client.beta.chat.completions.parse(
            model=modelo,
            messages=[
                {
                    "role": "system",
                    "content": "Você é um especialista em extração de dados e análise de sentimentos."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            # 3. Passamos o modelo Pydantic diretamente aqui
            response_format=AnaliseComentario,
            temperature=0
        )

        # 4. Acessamos o objeto Pydantic já parseado e validado
        analise_pydantic = resposta.choices[0].message.parsed

        # Retornamos como dicionário (ou você pode retornar o próprio objeto analise_pydantic)
        return analise_pydantic.model_dump()

    except Exception as e:
        # O Pydantic ou a API levantarão exceções claras se algo falhar
        return {
            "erro": str(e)
        }


if __name__ == "__main__":

    comentarios = [
        "O atendimento foi excelente e muito rápido.",
        "O sistema trava toda hora e perdi meu trabalho.",
        "A interface ficou bonita, mas está muito lenta.",
        "Gostei bastante da experiência.",
        "O suporte nunca responde e o aplicativo apresenta erro no login.",
    ]

    for comentario in comentarios:
        resultado = analisar_comentario(comentario, CLIENT, MODELO)

        print("=" * 70)
        print("COMENTÁRIO:")
        print(comentario)

        time.sleep(1)
