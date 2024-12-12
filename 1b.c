# Step1:importing all the libraries
import numpy as np
import pandas as pd
importmatplotlib.pyplot as plt
%matplotlib inline
# Step2:load dataset df=pd.read_csv("housing_prices_SLR.csv",delimiter=',') df.head()
Step3: Feature matrix and Target vector
x=df[['AREA']].values#feature Matrix y=df.PRICE.values#Target Matrix x[:5] #slicing
y[:5]
Step4: Split the data into 80-20
#from packagename import function
fromsklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100) #80 20 split,random_state to reproduce the same split everytime
print(x_train.shape) print(x_test.shape) print(x_train.shape) print(x_test.shape)
 
(40, 1)
(10, 1)
(40, 1)
(10, 1)
#step5: Fit the line:Train the SLR Model
fromsklearn.linear_model import LinearRegression
lr_model= LinearRegression()
lr_model.fit(x_train,y_train)
print(lr_model.intercept_) # (PRICE=(-4481.80028058845)+8.65903854)*AREA print(lr_model.coef_)#y=c+mx
b0:-3103.34066448488
b1:[7.75979089] lr_model=LinearRegression(fit_intercept= False)
lr_model.fit(x_train,y_train)
print(lr_model.intercept_) # (PRICE=(-4481.80028058845)+8.65903854)*AREA
print(lr_model.coef_)#y=c+mx
b0:0.0 b1:6.03609138
#step6:predict using the model
fromsklearn.metrics import r2_score
y_train
lr_model.predict(x_train)
# step7:calculating R^2score using tain and test model r2_score(y_train,lr_model.predict(x_train)) R^2_Train_Score:0.820250203127675 r2_score(y_test,lr_model.predict(x_test)) R^2_Test_Score:0.5059420550739799 lr_model.score(x_test,y_test) #2.second way of calculating R2 score R^2_Test_Score:0.5059420550739799
step8:Visualizing the model
plt.scatter(x_train[:,0],y_train,c='red') plt.scatter(x_test[:,0],y_test,c='blue')

plt.plot(x_train[:,0],lr_model.predict(x_train),c='y')
