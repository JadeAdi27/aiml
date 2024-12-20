import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('breast_cancer.csv') df
 df = df.iloc[:, :-1] df
df.head()
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
from sklearn.tree import DecisionTreeClassifier dt_classifier.fit(x_train, y_train)
predictions = dt_classifier.predict(x_test) prob_predictions = dt_classifier.predict_proba(x_test)
from sklearn.metrics import accuracy_score, confusion_matrix ,classification_report print("Training accuracy Score is : ", accuracy_score(y_train,
dt_classifier.predict(x_train)))
print("Training Confusion Matrix is : \n", confusion_matrix(y_train, dt_classifier.predict(x_train)))
print("Testing Confusion Matrix is : \n", confusion_matrix(y_test, dt_classifier.predict(x_test)))
print(classification_report(y_test,dt_classifier.predict(x_test)))
