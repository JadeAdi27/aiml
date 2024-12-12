import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ### Step 2: Load the CSV file into pandas DataFrame and clean the data
df = pd.read_csv("breast_cancer.csv")
df = df.iloc[:, :-1]  # Remove the last column if not needed
print(df.shape)
print(df.head())

# ### Step 3: Create the Feature Matrix and Target Vector and check the first 5 rows
x = df.iloc[:, 2:].values  # Features
y = df["diagnosis"].values  # Target
print(x[:2])
print(y[:5])

# ### Step 4: Split the data into training set and test set
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=500
)
print(x_train.shape)  # (455, 30)
print(x_test.shape)   # (114, 30)
print(y_train.shape)
print(y_test.shape)
print((y_train == 'M').sum())  # Count of malignant in train set
print((y_train == 'B').sum())  # Count of benign in train set

# Baseline model, accuracy, confusion matrix, classification report
# Baseline model accuracy = (more frequent class occurrences) / (total elements)
baseline_pred = ["B"] * len(y_train)  # Predict 'B' for all training samples
baseline_accuracy = accuracy_score(y_train, baseline_pred)
print(f"Baseline model accuracy: {baseline_accuracy:.2f}")

# ### Step 5: Instantiate a Gaussian Naive Bayes model and train it
nb_model = GaussianNB()
nb_model.fit(x_train, y_train)
print(f"Training accuracy: {nb_model.score(x_train, y_train):.2f}")
print(f"Testing accuracy: {nb_model.score(x_test, y_test):.2f}")

# Confusion matrix for training data
train_conf_matrix = confusion_matrix(y_train, nb_model.predict(x_train))
print("Training Confusion Matrix:")
print(train_conf_matrix)

# Confusion matrix for test data
test_conf_matrix = confusion_matrix(y_test, nb_model.predict(x_test))
print("Testing Confusion Matrix:")
print(test_conf_matrix)

# Classification report for training data
print("Training Classification Report:")
print(classification_report(y_train, nb_model.predict(x_train)))

# Classification report for test data
print("Testing Classification Report:")
print(classification_report(y_test, nb_model.predict(x_test)))
