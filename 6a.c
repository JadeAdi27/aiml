import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load and prepare data
dataframe = pd.read_csv('pima-indians-diabetes.csv')
X = dataframe.iloc[:, :8]
y = dataframe.iloc[:, 8]
features_train, features_test, target_train, target_test = train_test_split(X, y, test_size=0.33, random_state=0)

# Define the Keras model
network = Sequential()
network.add(Dense(units=8, activation="relu", input_shape=(features_train.shape[1],)))
network.add(Dense(units=8, activation="relu"))
network.add(Dense(units=1, activation="sigmoid"))

# Compile the model
network.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = network.fit(features_train, target_train, epochs=20, batch_size=100, validation_data=(features_test, target_test), verbose=1)

# Plot training and test loss
training_loss = history.history["loss"]
test_loss = history.history["val_loss"]
epoch_count = range(1, len(training_loss) + 1)
plt.plot(epoch_count, training_loss, "r--")
plt.plot(epoch_count, test_loss, "b-")
plt.legend(["Training Loss", "Test Loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# Evaluate accuracy
_, train_accuracy = network.evaluate(features_train, target_train)
_, test_accuracy = network.evaluate(features_test, target_test)
print(f'Training Accuracy: {train_accuracy * 100:.2f}')
print(f'Test Accuracy: {test_accuracy * 100:.2f}')

# Predict using the Keras model
predicted_target = network.predict(features_test)
for i in range(10):
    print(predicted_target[i])

# Plot training and test accuracy
training_accuracy = history.history["accuracy"]
test_accuracy = history.history["val_accuracy"]
plt.plot(epoch_count, training_accuracy, "r--")
plt.plot(epoch_count, test_accuracy, "b-")
plt.legend(["Training Accuracy", "Test Accuracy"])
plt.xlabel("Epoch")
plt.ylabel("Accuracy Score")
plt.show()
