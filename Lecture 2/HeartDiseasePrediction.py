from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
heart_data = pd.read_csv('heart.csv')

# Separate features and labels
X = heart_data.drop(columns=['target'], axis=1)
Y = heart_data['target']

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

# Predictions and accuracy
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
# print(training_data_accuracy)
#
#testdata predictions and accuracy
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
# print(test_data_accuracy)


input_data = (67,0,0,106,223,0,1,142,0,0.3,2,2,2)
input_data_as_array = np.asarray(input_data)
input_data_as_array_reshaped = input_data_as_array.reshape(1,-1)

prediction = model.predict(input_data_as_array_reshaped)
print(prediction)