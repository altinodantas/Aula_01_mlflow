import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np

import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("aula_01_diabetes_experimento")

# Enable autologging for scikit-learn
mlflow.sklearn.autolog()

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

y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("ConfusionMatrix:\n",metrics.confusion_matrix(y_test, y_pred))