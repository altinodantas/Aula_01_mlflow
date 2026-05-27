# Material de Apoio: `keras_regression_autolog.py`

Este arquivo explica o código do script `keras_regression_autolog.py` em blocos relevantes, com foco no uso de Keras, normalização de dados e integração com MLflow para autologging.

---

## 1. Importação de bibliotecas

```python
import mlflow
import numpy as np
from keras import Input
from tensorflow import keras
from sklearn.metrics import r2_score
import pandas as pd
```

- `mlflow`: biblioteca usada para rastrear experimentos de machine learning.
- `numpy`: biblioteca para operações numéricas e manipulação de arrays.
- `keras` e `Input`: usados para construir a rede neural.
- `r2_score`: métrica de regressão para avaliar o modelo no conjunto de teste.
- `pandas`: usado para carregar e manipular dados tabulares.

> Esta seção carrega as dependências necessárias para leitura de dados, criação do modelo e monitoramento do experimento.

---

## 2. Configuração do MLflow

```python
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("aula_01_keras_autolog_regression")

# Enable autologging
mlflow.tensorflow.autolog()
```

- `set_tracking_uri(...)`: define onde o MLflow irá salvar os dados de rastreamento. Neste caso, um banco de dados SQLite local chamado `mlflow.db`.
- `set_experiment(...)`: define o nome do experimento no MLflow. Todos os runs registrados aqui serão agrupados neste experimento.
- `mlflow.tensorflow.autolog()`: habilita o autologging para TensorFlow/Keras. Isso faz com que o MLflow registre automaticamente parâmetros, métricas, gráficos de treino e artefatos do modelo.

> Com o autolog ativado, não é necessário chamar manualmente `mlflow.log_param` para muitos itens padrão.

---

## 3. Função de normalização

```python

def norm(x):
    return (x - train_stats['mean']) / train_stats['std']
```

- Esta função recebe um DataFrame e aplica normalização usando média e desvio padrão calculados no conjunto de treino.
- A normalização é importante para ajudar o modelo a aprender mais rápido e com maior estabilidade.

> A função assume que `train_stats` já existe no escopo global.

---

## 4. Carregamento e limpeza dos dados

```python
dados = pd.read_csv("data/mpg_data.csv")
dados = dados.dropna()

origin = dados.pop('Origin')
dados['USA'] = ((origin == 1) * 1.0).copy()
dados['Europe'] = ((origin == 2) * 1.0).copy()
dados['Japan'] = ((origin == 3) * 1.0).copy()
```

- `pd.read_csv(...)`: lê o arquivo CSV com os dados de consumo de combustível.
- `dropna()`: remove linhas que contenham valores ausentes.
- `dados.pop('Origin')`: remove a coluna `Origin` do DataFrame e guarda esse vetor separadamente.
- As três colunas `USA`, `Europe` e `Japan` são criadas com codificação one-hot para representar a origem do carro.

> A transformação one-hot torna a variável categórica `Origin` utilizável pelo modelo numérico.

---

## 5. Separação em treino e teste

```python
train_dataset = dados.sample(frac=0.8, random_state=0)
test_dataset = dados.drop(train_dataset.index)

train_labels = train_dataset.pop('MPG').to_numpy()
test_labels = test_dataset.pop('MPG').to_numpy()
```

- `sample(frac=0.8, random_state=0)`: seleciona aleatoriamente 80% dos dados para treino.
- `drop(train_dataset.index)`: usa o restante dos registros como conjunto de teste.
- `pop('MPG')`: remove a coluna alvo `MPG` dos conjuntos de entrada e armazena os valores em `train_labels` e `test_labels`.

> Separar treino e teste permite avaliar o desempenho do modelo em dados que ele nunca viu durante o treinamento.

---

## 6. Estatísticas do conjunto de treino

```python
train_stats = train_dataset.describe()
train_stats = train_stats.transpose()

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

normed_train_data = np.array(normed_train_data)
normed_test_data = np.array(normed_test_data)
```

- `describe()`: calcula estatísticas de cada coluna numérica do conjunto de treino, como média e desvio padrão.
- `transpose()`: organiza essas estatísticas para facilitar o acesso por coluna.
- `norm(...)`: normaliza os dados de treino e teste usando as estatísticas do treino.
- `np.array(...)`: converte os DataFrames normalizados em arrays NumPy, formato esperado pelo Keras.

> A normalização com base apenas no conjunto de treino evita vazamento de informação do conjunto de teste.

---

## 7. Definição do modelo Keras

```python
model = keras.Sequential([
    Input(shape=(9,)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])

optimizer = keras.optimizers.RMSprop(0.001)

model.compile(loss='mse',
              optimizer=optimizer,
              metrics=['mae'])
```

- `Sequential([...])`: cria uma rede neural sequencial.
- `Input(shape=(9,))`: define que o modelo espera 9 variáveis de entrada.
- `Dense(64, activation="relu")`: primeira camada totalmente conectada com 64 neurônios e função de ativação ReLU.
- `Dense(32, activation="relu")`: segunda camada oculta com 32 neurônios.
- `Dense(1)`: camada de saída para regressão, produzindo um valor contínuo.
- `RMSprop(0.001)`: otimizador com taxa de aprendizagem 0.001.
- `compile(...)`: define a função de perda como `mse` (erro quadrático médio) e adiciona `mae` (erro absoluto médio) como métrica de monitoramento.

> Este modelo é adequado para regressão numérica sobre o consumo de combustível (MPG).

---

## 8. Treinamento e registro do experimento

```python
mlflow.enable_system_metrics_logging()
with mlflow.start_run():
    EPOCHS = 200
    history = model.fit(normed_train_data, train_labels,
                        epochs=EPOCHS,
                        verbose=0,
                        validation_split=0.2)

    predictions = model.predict(normed_test_data)
    r2 = r2_score(test_labels, predictions)
    mlflow.log_metric("test_r2", r2)

    print("R2:", r2)
```

- `enable_system_metrics_logging()`: adiciona métricas de sistema ao log, como uso de CPU e memória.
- `mlflow.start_run()`: inicia um novo run do MLflow; tudo dentro do `with` será registrado nesse run.
- `model.fit(...)`: treina o modelo por 200 épocas, usando 20% dos dados de treino como validação.
- `model.predict(...)`: gera previsões com o conjunto de teste normalizado.
- `r2_score(...)`: calcula a métrica R² para avaliar a qualidade da regressão.
- `mlflow.log_metric(...)`: registra o valor de `test_r2` no MLflow.
- `print(...)`: exibe o resultado da métrica no console.

> Ao final, o MLflow armazena automaticamente parâmetros, métricas, gráficos e o modelo treinado, graças ao autolog.

---

## 9. Observações finais

- O uso de `mlflow.tensorflow.autolog()` reduz muito o trabalho manual de monitoramento.
- A normalização e a codificação one-hot são etapas essenciais para trabalhar com dados tabulares em redes neurais.
- A divisão treino/teste garante que o modelo seja avaliado de forma mais realista.
- A métrica `R2` complementa `mae` e `mse`, mostrando a proporção de variação explicada pelo modelo.

> Este material foi pensado para ajudar estudantes a compreenderem cada bloco funcional do script e a relação entre preparação de dados, definição de modelo e registro de experimentos.
