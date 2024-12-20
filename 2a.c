#importing Libraries import numpy as np import pandas as pd
importmatplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
data = pd.read_csv('student.csv') print(data.shape)
data.head()

math = data['Math'].values
read = data['Reading'].values write = data['Writing'].values
# Ploting the scores as scatter plot
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(math, read, write, color='#ef1234') plt.legend()
plt.show()
   m = len(math)
x0 = np.ones(m)
X = np.array([x0, math, read]).T
   # Initial Coefficients
B = np.array([0, 0, 0])
Y = np.array(write)
alpha = 0.0001 defcost_function(X, Y, B):
m = len(Y)
J = np.sum((X.dot(B) - Y) ** 2)/(2 * m) return J
inital_cost = cost_function(X, Y, B) print("Initial Cost")
 print(inital_cost)
defgradient_descent(X, Y, B, alpha, iterations): cost_history = [0] * iterations
m = len(Y)
for iteration in range(iterations):
# Hypothesis Values
h = X.dot(B)
# Difference b/w Hypothesis and Actual Y loss = h - Y
# Gradient Calculation
 
  gradient = X.T.dot(loss) / m
# Changing Values of B using Gradient B = B - alpha * gradient
# New Cost Value
cost = cost_function(X, Y, B) cost_history[iteration] = cost
return B, cost_history
 # 100000 Iterations
newB, cost_history = gradient_descent(X, Y, B, alpha, 100000)

  # Model Evaluation - RMSE
defrmse(Y, Y_pred):
rmse = np.sqrt(sum((Y - Y_pred) ** 2) / len(Y))
return rmse
   # Model Evaluation - R2 Score
def r2_score(Y, Y_pred): mean_y = np.mean(Y)
ss_tot = sum((Y - mean_y) ** 2) ss_res = sum((Y - Y_pred) ** 2)
r2 = 1 - (ss_res / ss_tot)
 
return r2
Y_pred = X.dot(newB)
print("R2 Score") print(r2_score(Y, Y_pred))
