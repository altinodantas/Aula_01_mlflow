import os
from dotenv import load_dotenv
import mlflow
from openai import OpenAI
from mlflow.genai import scorer
from mlflow.genai.scorers import Correctness, Guidelines

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("API_KEY_OPENAI")

# Use different env variable when using a different LLM provider
mlflow.set_experiment("Evaluation Quickstart")

# Your agent implementation
client = OpenAI(
    api_key=os.getenv("API_KEY_OPEN_ROUTER"),
    base_url="https://openrouter.ai/api/v1"
)


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


# Wrapper function for evaluation
def qa_predict_fn(question: str) -> str:
    return my_agent(question)


# Evaluation dataset
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


# Scorers
@scorer
def is_concise(outputs: str) -> bool:
    return len(outputs.split()) <= 5


scorers = [
    Correctness(),
    Guidelines(name="is_english", guidelines="The answer must be in English"),
    is_concise,
]

# Run evaluation
if __name__ == "__main__":
    results = mlflow.genai.evaluate(
        data=eval_dataset,
        predict_fn=qa_predict_fn,
        scorers=scorers,
    )