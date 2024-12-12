#importing libraries
import numpy as np
import pandas as pd importmatplotlib.pyplot as plt %matplotlib inline
#Loading dataset
df=pd.read_csv("housing_prices.csv") df.head()

                    
#setting Target and Feature Vectors x=df.iloc[:,:3].values
y=df.iloc[:,3].values
#Splittiing the dataset
fromsklearn.model_selection import train_test_split x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
# Fitting the model
fromsklearn.linear_model import LinearRegression
mlr_model= LinearRegression(fit_intercept=True)
mlr_model.fit(x_train,y_train)
print(mlr_model.intercept_) # (PRICE=(-4481.80028058845)+8.65903854)*AREA
print(mlr_model.coef_)


# Finding R2 score
print(mlr_model.score(x_train,y_train)) print(mlr_model.score(x_test,y_test))
