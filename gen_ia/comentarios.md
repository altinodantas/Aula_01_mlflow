# Material de Apoio: `gen_ia/comentarios.py`

Este material explica, passo a passo, o script `gen_ia/comentarios.py` como um apoio para estudantes que estão aprendendo a usar LLMs, MLflow e Pydantic para análise de comentários.

---

## 1. Importações e preparação de ambiente

```python
import os
import time
from dotenv import load_dotenv
import mlflow
from openai import OpenAI, Client
from typing import List, Literal
from pydantic import BaseModel, Field
```

- `os` e `dotenv`: para carregar variáveis de ambiente e acessar chaves secretas.
- `time`: usado para pausar a execução entre chamadas.
- `mlflow`: para rastrear e registrar execuções de modelos.
- `OpenAI` e `Client`: cliente usado para chamar a API de LLM.
- `typing`: anotações de tipos para o schema de saída.
- `pydantic`: validação estrita de dados usando modelos.

> Esse bloco inicial prepara o ambiente e as ferramentas necessárias para autenticação, execução do agente e validação de resultados.

---

## 2. Configuração do MLflow e do modelo

```python
load_dotenv()

mlflow.set_tracking_uri("sqlite:///../mlflow.db")
mlflow.set_experiment("Análise de Comentários")
mlflow.openai.autolog()
```

- `load_dotenv()`: lê o arquivo `.env` e carrega variáveis de ambiente.
- `set_tracking_uri(...)`: define onde MLflow salva os dados de experimento. Aqui é um banco SQLite localizado um nível acima.
- `set_experiment(...)`: define o nome do experimento em que os runs serão agrupados.
- `mlflow.openai.autolog()`: habilita o autologging para chamadas OpenAI, registrando automaticamente parâmetros, métricas e artefatos.

> Com essa configuração, a execução do script ficará registrada no MLflow sem precisar de logs manuais adicionais.

---

## 3. Criação do cliente OpenAI e definição do modelo

```python
CLIENT = OpenAI(
    api_key=os.getenv("API_KEY_OPEN_ROUTER"),
    base_url="https://openrouter.ai/api/v1"
)
MODELO = "openai/gpt-oss-120b:free"
```

- `CLIENT`: instancia o cliente OpenAI usando a chave de API retirada das variáveis de ambiente.
- `base_url`: indica o endpoint do provedor de LLM.
- `MODELO`: string que identifica o modelo que será utilizado nas requisições.

> Esse cliente é a ponte entre o código Python e o serviço de que prover o modelo generativo.

---

## 4. Definição do schema de saída com Pydantic

```python
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
```

- `AnaliseComentario`: modelo Pydantic que descreve a estrutura esperada da resposta do LLM.
- `sentimento`: valor categórico estrito que deve ser `positivo`, `negativo` ou `neutro`.
- `topicos`: lista de tópicos detectados no comentário.
- `resumo`: texto resumido da mensagem.

> Usar um schema estrito ajuda a garantir que a resposta atenderá ao formato esperado, evitando erros posteriores.

---

## 5. Função principal de análise de comentários

```python
def analisar_comentario(texto: str, client: Client, modelo: str) -> dict:
    prompt = f"Analise o seguinte comentário: '{texto}'"

    try:
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
            response_format=AnaliseComentario,
            temperature=0
        )

        analise_pydantic = resposta.choices[0].message.parsed

        return analise_pydantic.model_dump()

    except Exception as e:
        return {
            "erro": str(e)
        }
```

- `prompt`: texto enviado ao modelo com a instrução clara de analisar o comentário.
- `client.beta.chat.completions.parse(...)`: chamada de API que pede ao modelo para gerar uma resposta estruturada e valida-la automaticamente com Pydantic.
- `response_format=AnaliseComentario`: instrui a API a formatar a saída conforme o schema.
- `temperature=0`: configura o modelo para respostas determinísticas e menos criativas.
- `analise_pydantic.model_dump()`: converte o objeto Pydantic validado em dicionário.
- bloco `except`: captura erros de validação ou da API e retorna uma mensagem de erro.

> Esse é o núcleo do script: a função transforma texto livre em uma análise estruturada e validada.

---

## 6. Execução do script

```python
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
```

- `if __name__ == "__main__"`: garante que o código só execute quando o arquivo é rodado diretamente.
- `comentarios`: exemplos de texto usados para teste.
- loop `for`: analisa cada comentário e imprime informações na tela.
- `time.sleep(1)`: pausa 1 segundo entre chamadas para evitar sobrecarga da API.

> Este bloco demonstra como usar a função de análise em prática, com exemplos reais de comentários.

---

## 7. Observações finais

- O script combina LLMs com validação de saída estruturada, um padrão moderno para aplicações confiáveis.
- Pydantic garante que o resultado contenha `sentimento`, `topicos` e `resumo` no formato correto.
- O autolog do MLflow facilita o rastreamento do uso do modelo e de métricas de chamadas.
- A instrução `temperature=0` é útil quando queremos respostas estáveis e consistentes.

> Este material é um guia para estudantes entenderem os principais componentes e o fluxo de `gen_ia/comentarios.py`.
