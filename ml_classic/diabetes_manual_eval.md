# Material de Apoio: `diabetes_manual_eval.py`

Este material explica cada bloco do script `diabetes_manual_eval.py`, pensado para estudantes que querem entender o fluxo de treinamento, avaliação e registro de um modelo clássico com MLflow.

---

## 1. Importação de bibliotecas

```python
import pandas as pd
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split, TunedThresholdClassifierCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np

import mlflow
```

- `pandas`: leitura e manipulação de dados tabulares.
- `infer_signature`: cria uma assinatura de modelo para MLflow, facilitando a validação de entradas e saídas.
- `train_test_split`: divide o conjunto de dados em treino e teste.
- `TunedThresholdClassifierCV`: ajusta o limiar de decisão de um classificador.
- `SVC`: classificador de máquina de vetores de suporte.
- `StandardScaler`: normalização de atributos para média 0 e variância 1.
- `numpy`: manipulação de arrays numéricos.
- `mlflow`: rastreamento de experimentos e gerenciamento de modelos.

> Esse bloco carrega o conjunto de ferramentas para preparação de dados, modelo, avaliação e registro.

---

## 2. Configuração do MLflow

```python
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("aula_01_diabetes_experimento")
```

- `set_tracking_uri(...)`: define o local onde o MLflow grava o histórico de experimentos. Aqui é usado um banco SQLite local chamado `mlflow.db`.
- `set_experiment(...)`: nome do experimento que agrupará os runs neste script.

> Essa configuração garante que os resultados sejam organizados e acessíveis na interface do MLflow.

---

## 3. Carregamento dos dados

```python
dados = pd.read_csv("../data/diabetes.csv")
feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
X = dados[feature_cols]
y = dados.label.to_numpy()
```

- `pd.read_csv(...)`: carrega o arquivo CSV com dados de diabetes.
- `feature_cols`: lista de colunas usadas como atributos de entrada.
- `X`: matriz de recursos selecionados.
- `y`: vetor de rótulos binários (`label`).

> Separar os dados em `X` e `y` é o passo inicial para treinar um classificador.

---

## 4. Normalização dos dados

```python
scale = StandardScaler()
X = scale.fit_transform(X)
X = np.array(X)
```

- `StandardScaler()`: cria o objeto de normalização.
- `fit_transform(X)`: calcula a média e desvio padrão no conjunto de treino e aplica a transformação.
- `np.array(X)`: garante que o conjunto de dados esteja no formato NumPy.

> A normalização é importante para modelos como o SVM, que são sensíveis à escala das variáveis.

---

## 5. Divisão treino/teste

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
```

- `test_size=0.3`: usa 30% dos dados para teste e 70% para treino.
- `random_state=1`: garante que a divisão seja reproduzível.

> Separar os dados em treino e teste evita a avaliação ilusória do modelo com dados usados no treinamento.

---

## 6. Treinamento do modelo

```python
clf = SVC(probability=True)
clf = clf.fit(X_train,y_train)
```

- `SVC(probability=True)`: cria um classificador SVM que calcula probabilidades.
- `fit(...)`: treina o modelo com os dados de treino.

> O SVC é um classificador robusto usado frequentemente em problemas de classificação binária.

---

## 7. Criação do conjunto de avaliação

```python
eval_data = pd.DataFrame(X_test, columns=feature_cols)
eval_data["label"] = y_test
```

- `pd.DataFrame(...)`: reconstrói o conjunto de teste como DataFrame com os nomes das colunas.
- `eval_data["label"]`: adiciona a coluna de rótulos reais para a avaliação.

> O MLflow espera um DataFrame com as _features_ e a coluna de alvo para avaliar o modelo.

---

## 8. Ajuste de limiar com `TunedThresholdClassifierCV`

```python
classifier_tuned = TunedThresholdClassifierCV(
    clf, scoring="balanced_accuracy"
).fit(X_train, y_train)
```

- `TunedThresholdClassifierCV(...)`: envolve o classificador para encontrar o melhor limiar de decisão.
- `scoring="balanced_accuracy"`: usa acurácia balanceada como critério de ajuste.
- `fit(...)`: ajusta o limiar com base no conjunto de treino.

> Ajustar o limiar pode melhorar métricas importantes quando há desbalanceamento entre classes.

---

## 9. Registro e avaliação no MLflow

```python
with mlflow.start_run():
    # Log model
    signature = infer_signature(X_test, classifier_tuned.predict(X_test))
    model_info = mlflow.sklearn.log_model(classifier_tuned, name="model", signature=signature)

    result = mlflow.models.evaluate(
        model=model_info.model_uri,
        data=eval_data,
        targets="label",
        model_type="classifier"
    )

    print(f"Accuracy: {result.metrics['accuracy_score']:.3f}")
    print(f"F1 Score: {result.metrics['f1_score']:.3f}")
    print(f"ROC AUC: {result.metrics['roc_auc']:.3f}")
```

- `mlflow.start_run()`: inicia um novo run no experimento.
- `infer_signature(...)`: cria uma assinatura do modelo a partir das entradas e saídas de exemplo.
- `mlflow.sklearn.log_model(...)`: registra o modelo ajustado no MLflow.
- `mlflow.models.evaluate(...)`: avalia o modelo com o conjunto de teste e calcula métricas automaticamente.
- `print(...)`: exibe as métricas de avaliação no console.

> Esse bloco finaliza o experimento e documenta o desempenho do modelo de forma reproduzível.

---

## 10. Observações finais

- O script mostra um fluxo clássico de machine learning com MLflow: preparar dados, treinar, validar e registrar.
- O uso de `infer_signature` ajuda a manter a compatibilidade entre os dados e o modelo.
- `TunedThresholdClassifierCV` é valioso quando o objetivo é ajustar a sensibilidade do classificador para métricas específicas.
- `mlflow.models.evaluate` oferece uma avaliação padronizada que facilita comparações entre modelos.

> Este material foi feito para ajudar estudantes a entenderem cada etapa do código em `diabetes_manual_eval.py`.
