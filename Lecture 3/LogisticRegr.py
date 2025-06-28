import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Logistic_Regression:

    def __init__(self, learning_rate, no_of_iterations):
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations

    def fit(self, X, Y):
        self.m, self.n = X.shape
        self.w = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y

        for i in range(self.no_of_iterations):
            self.update_weights()

    def update_weights(self):
        Y_hat = 1 / (1 + np.exp(-(self.X.dot(self.w) + self.b)))

        dw = (1 / self.m) * np.dot(self.X.T, (Y_hat - self.Y))
        db = (1 / self.m) * np.sum(Y_hat - self.Y)

        self.w -= self.learning_rate * dw
        self.b -= self.learning_rate * db

    def predict(self, X):
        Y_pred = 1 / (1 + np.exp(-(X.dot(self.w) + self.b)))
        return np.where(Y_pred > 0.5, 1, 0)


# Load and prepare data
diabetes_dataset = pd.read_csv('diabetes.csv')
features = diabetes_dataset.drop(columns='Outcome', axis=1)
target = diabetes_dataset['Outcome']

# Standardize features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Train/test split
X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size=0.2, random_state=2)

# Train model
classifier = Logistic_Regression(learning_rate=0.01, no_of_iterations=1000)
classifier.fit(X_train, Y_train)

# Accuracy
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print("Training Accuracy Score:", training_data_accuracy)

# Prediction system
input_data = [8, 125, 96, 0, 0, 0, 0.232, 54]
input_np = np.asarray(input_data).reshape(1, -1)
input_scaled = scaler.transform(input_np)

prediction = classifier.predict(input_scaled)

if prediction[0] == 1:
    print("This person is **non-diabetic**.")
else:
    print("This person is **diabetic**.")
