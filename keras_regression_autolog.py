import mlflow
import numpy as np
from keras import Input
from tensorflow import keras
from sklearn.metrics import r2_score
import pandas as pd

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("aula_01_keras_autolog")

# Enable autologging
mlflow.tensorflow.autolog()


def norm(x):
    return (x - train_stats['mean']) / train_stats['std']


dados = pd.read_csv("data/mpg_data.csv")
dados = dados.dropna()

origin = dados.pop('Origin')
dados['USA'] = ((origin == 1) * 1.0).copy()
dados['Europe'] = ((origin == 2) * 1.0).copy()
dados['Japan'] = ((origin == 3) * 1.0).copy()

train_dataset = dados.sample(frac=0.8, random_state=0)
test_dataset = dados.drop(train_dataset.index)

train_labels = train_dataset.pop('MPG').to_numpy()
test_labels = test_dataset.pop('MPG').to_numpy()

train_stats = train_dataset.describe()
train_stats = train_stats.transpose()

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

normed_train_data = np.array(normed_train_data)
normed_test_data = np.array(normed_test_data)

# Define model
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

# Training with automatic logging
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
