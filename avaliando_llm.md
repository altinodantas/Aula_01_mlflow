# Material de Apoio: `avaliando_llm.py`

Este material explica cada bloco do script `avaliando_llm.py` de forma didática, como suporte a estudantes que estão aprendendo a avaliar modelos de linguagem com MLflow e OpenAI.

---

## 1. Importação de bibliotecas

```python
import os
from dotenv import load_dotenv
import mlflow
from openai import OpenAI
from mlflow.genai import scorer
from mlflow.genai.scorers import Correctness, Guidelines
```

- `os`: usado para acessar variáveis de ambiente.
- `load_dotenv`: carrega variáveis de ambiente do arquivo `.env`.
- `mlflow`: biblioteca para rastrear experimentos e avaliações.
- `OpenAI`: cliente para interagir com a API do OpenAI.
- `scorer`: decorador do MLflow para criar funções de avaliação personalizadas.
- `Correctness` e `Guidelines`: scorers prontos do MLflow para avaliar respostas de LLMs.

> Este bloco prepara as ferramentas para autenticação, execução do agente e avaliação do modelo.

---

## 2. Carregar variáveis de ambiente e configurar o MLflow

```python
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("API_KEY_OPENAI")

# Use different env variable when using a different LLM provider
mlflow.set_experiment("Evaluation Quickstart")
```

- `load_dotenv()`: lê o arquivo `.env` e define as variáveis de ambiente no processo.
- `os.getenv("API_KEY_OPENAI")`: recupera a chave de API do OpenAI.
- `set_experiment(...)`: define o experimento no MLflow onde os resultados serão registrados.

Foi necessário usar uma chave de API da OpenIA porque, para as métricas Correctness e Guidelines, o MLFlow utiliza a estratégia de LLM as Judge. Outros provedores são suportados.

> Definir o experimento permite organizar avaliações e comparar resultados no UI do MLflow.

---

## 3. Configuração do cliente OpenAI

```python
client = OpenAI(
    api_key=os.getenv("API_KEY_OPEN_ROUTER"),
    base_url="https://openrouter.ai/api/v1"
)
```

- `api_key`: define a chave de API usada para autenticar as chamadas.
- `base_url`: define o endpoint da API, neste caso `openrouter.ai`.

Conforme em exemplos anteriores, o client das requisições ao LLM foi configurado para usar modelos da OpenRouter. Por questões de custo.

> Essa configuração torna possível enviar pedidos de `chat completions` para o modelo.

---

## 4. Implementação do agente

```python
def my_agent(question: str) -> str | None:
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b:free",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer questions concisely.",
            },
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content
```

- `my_agent(...)`: função que envia a pergunta ao modelo e retorna a resposta gerada.
- `model`: identifica qual modelo será usado.
- `messages`: histórico de conversas enviado ao modelo. Aqui há um sistema (`system`) e uma mensagem do usuário (`user`).
- `response.choices[0].message.content`: extrai o texto da primeira resposta.

> Este é o núcleo do agente: ele transforma uma pergunta em uma chamada à API e devolve a resposta do LLM.

---

## 5. Função de previsão para avaliação

```python
# Wrapper function for evaluation
def qa_predict_fn(question: str) -> str:
    return my_agent(question)
```

- `qa_predict_fn(...)`: função wrapper que adapta o agente para a interface esperada pelo MLflow GenAI.

> O MLflow usa essa função para gerar previsões durante a avaliação.

---

## 6. Conjunto de dados de avaliação

```python
eval_dataset = [
    {
        "inputs": {"question": "What is the capital of France?"},
        "expectations": {"expected_response": "Paris"},
    },
    {
        "inputs": {"question": "Who was the first person to build an airplane?"},
        "expectations": {"expected_response": "Wright Brothers"},
    },
    {
        "inputs": {"question": "Who wrote Romeo and Juliet?"},
        "expectations": {"expected_response": "William Shakespeare"},
    },
]
```

- `eval_dataset`: lista de exemplos usados para testar o agente.
- Cada item tem `inputs` (a pergunta) e `expectations` (a resposta esperada).

> Esse conjunto simples serve para avaliar se o agente responde corretamente a perguntas de conhecimento geral.

---

## 7. Definindo scorers de avaliação

```python
@scorer
def is_concise(outputs: str) -> bool:
    return len(outputs.split()) <= 5

scorers = [
    Correctness(),
    Guidelines(name="is_english", guidelines="The answer must be in English"),
    is_concise,
]
```

- `@scorer`: cria um scorer personalizado do MLflow.
- `is_concise(...)`: verifica se a resposta tem até 5 palavras.
- `Correctness()`: scorer padrão para medir se a resposta está correta com base em expectativas.
- `Guidelines(...)`: garante que a resposta siga uma regra, neste caso estar em inglês.

> Juntar scorers permite avaliar o modelo em diferentes dimensões: exatidão, idioma e formato da resposta.

---

## 8. Executando a avaliação

```python
if __name__ == "__main__":
    results = mlflow.genai.evaluate(
        data=eval_dataset,
        predict_fn=qa_predict_fn,
        scorers=scorers,
    )
```

- `if __name__ == "__main__"`: garante que a avaliação só seja executada quando o script for rodado diretamente.
- `mlflow.genai.evaluate(...)`: função do MLflow para avaliar o agente com base nos dados e scorers definidos.

> Esse bloco inicia a avaliação e registra os resultados no experimento do MLflow.

---

## 9. Observações finais

- O script mostra um fluxo básico de avaliação de modelos de linguagem com MLflow GenAI.
- As variáveis de ambiente e o `.env` são essenciais para manter chaves de API fora do código. Lembre-se de criar um .env ou configurar `API_KEY_OPENAI` e `API_KEY_OPEN_ROUTER`.
- O `scorer` personalizado permite criar regras específicas para o comportamento esperado.
- A avaliação automatizada ajuda a comparar diferentes agentes e modelos de forma organizada.

> Este material é pensado para apoiar estudantes na compreensão de avaliações de LLMs usando MLflow e OpenAI.
