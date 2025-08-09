import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
heart_Data = pd.read_csv('heart.csv')

# Processing the dataset
X = heart_Data.drop(columns=['target'], axis=1)
Y = heart_Data['target']

# Splitting the data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)

# Model training
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

# Predictions
Y_pred_train = model.predict(X_train)
Y_pred_test = model.predict(X_test)

# Evaluation on training data
train_accuracy = accuracy_score(Y_train, Y_pred_train)
train_precision = precision_score(Y_train, Y_pred_train)
train_recall = recall_score(Y_train, Y_pred_train)
train_f1 = f1_score(Y_train, Y_pred_train)

# Evaluation on test data
test_accuracy = accuracy_score(Y_test, Y_pred_test)
test_precision = precision_score(Y_test, Y_pred_test)
test_recall = recall_score(Y_test, Y_pred_test)
test_f1 = f1_score(Y_test, Y_pred_test)

# Print results
print("Training Data:")
print(f"Accuracy: {train_accuracy:.3f}")
print(f"Precision: {train_precision:.3f}")
print(f"Recall: {train_recall:.3f}")
print(f"F1 Score: {train_f1:.3f}")

print("\nTest Data:")
print(f"Accuracy: {test_accuracy:.3f}")
print(f"Precision: {test_precision:.3f}")
print(f"Recall: {test_recall:.3f}")
print(f"F1 Score: {test_f1:.3f}")
