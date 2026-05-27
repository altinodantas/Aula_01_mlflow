import mlflow
import numpy as np
from tensorflow import keras

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("aula_01_keras_autolog")

# Enable autologging
mlflow.tensorflow.autolog()

# Prepare sample data
X_train = np.random.rand(1000, 20)
y_train = np.random.randint(0, 2, 1000)

# Define model
model = keras.Sequential([
    keras.layers.Dense(64, activation="relu", input_shape=(20,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid"),
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Training with automatic logging
with mlflow.start_run():
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)