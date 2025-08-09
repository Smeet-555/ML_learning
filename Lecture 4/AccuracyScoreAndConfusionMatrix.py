import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
heart_data = pd.read_csv('heart.csv')

# Separate features and target
X = heart_data.drop(columns=['target'], axis=1)
Y = heart_data['target']

# Split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

# Evaluate the model

# accuracy score on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
# print("Training Accuracy:", training_data_accuracy)

# accuracy score on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
# print("Testing Accuracy:", test_data_accuracy)

# Confusion matrix
from sklearn.metrics import confusion_matrix
cf_matrix = confusion_matrix(Y_test , X_test_prediction)
# print(cf_matrix)
tn , fp ,fn , tp = cf_matrix.ravel()
# print(tn , fp ,fn ,tp)

import seaborn as sns
sns.heatmap(cf_matrix,annot=True)
plt.show()