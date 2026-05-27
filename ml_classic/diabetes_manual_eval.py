import pandas as pd
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split, TunedThresholdClassifierCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np

import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("aula_01_diabetes_experimento")

dados = pd.read_csv("../data/diabetes.csv")
feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
X = dados[feature_cols]
y = dados.label.to_numpy()

scale = StandardScaler()
X = scale.fit_transform(X)
X = np.array(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

clf = SVC(probability=True)
clf = clf.fit(X_train,y_train)

eval_data = pd.DataFrame(X_test, columns=feature_cols)
eval_data["label"] = y_test

# Tentando otimizar o limiar
classifier_tuned = TunedThresholdClassifierCV(
    clf, scoring="balanced_accuracy"
).fit(X_train, y_train)

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