import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class SVM_Classifier:
    def __init__(self, learning_rate, no_of_iterations, lambda_parameter):
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations
        self.lambda_parameter = lambda_parameter

    def fit(self, X, Y):
        self.m, self.n = X.shape
        self.w = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y

        for i in range(self.no_of_iterations):
            self.update_weights()

    def update_weights(self):
        y_label = np.where(self.Y <= 0, -1, 1)

        for index, x_i in enumerate(self.X):
            condition = y_label[index] * (np.dot(x_i, self.w) - self.b) >= 1
            if condition:
                dw = 2 * self.lambda_parameter * self.w
                db = 0
            else:
                dw = 2 * self.lambda_parameter * self.w - y_label[index] * x_i
                db = -y_label[index]

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

    def predict(self, X):
        output = np.dot(X, self.w) - self.b
        return np.where(np.sign(output) == -1, 0, 1)


# Load and preprocess data
diabetes_dataset = pd.read_csv('diabetes.csv')
features = diabetes_dataset.drop(columns=['Outcome'], axis=1)
target = diabetes_dataset['Outcome']

scaler = StandardScaler()
features = scaler.fit_transform(features)

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size=0.2, random_state=2)

# Train model
classifier = SVM_Classifier(learning_rate=0.01, no_of_iterations=1000, lambda_parameter=0.01)
classifier.fit(X_train, Y_train)

# Evaluate
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)

# Predict on new input
input_data = [8, 125, 96, 0, 0, 0, 0.232, 54]
input_np = np.asarray(input_data).reshape(1, -1)
input_scaled = scaler.transform(input_np)
prediction = classifier.predict(input_scaled)

if prediction[0] == 1:
    print("The person is non-diabetic")
else:
    print("The person is diabetic")
