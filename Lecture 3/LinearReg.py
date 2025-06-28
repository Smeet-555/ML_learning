import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, no_of_iterations=1000):
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations

    def fit(self, X, Y):
        # Get number of samples and features
        self.m, self.n = X.shape

        # Initialize weights and bias
        self.w = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y

        # Run gradient descent
        for i in range(self.no_of_iterations):
            self.update_weights()

    def update_weights(self):
        Y_prediction = self.predict(self.X)

        # Compute gradients
        dw = -(2 / self.m) * (self.X.T @ (self.Y - Y_prediction))
        db = -(2 / self.m) * np.sum(self.Y - Y_prediction)

        # Update weights and bias
        self.w -= self.learning_rate * dw
        self.b -= self.learning_rate * db

    def predict(self, X):
        return X .dot(self.w) + self.b

import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

salary_data = pd.read_csv('salary_data.csv')
# print(salary_data.head())

X = salary_data.iloc[:,:-1].values
Y = salary_data.iloc[:,1]

X_train , X_test , Y_train , Y_test = train_test_split(X,Y , random_state=2 , test_size=0.33)

model = LinearRegression(learning_rate=0.02 ,  no_of_iterations=1000 )

model.fit(X_train , Y_train)

#checking the parameters
# print("weight: " , model.w[0])
# print("bias: " , model.b)

# prediting the model

test_data_prediction = model.predict(X_test)

plt.scatter(X_test,Y_test,color ='r')
plt.plot(X_test,test_data_prediction,color ='b')
plt.xlabel('work experience')
plt.ylabel('salary')
plt.show(  )