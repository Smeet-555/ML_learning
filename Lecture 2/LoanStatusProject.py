import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Load data
loan_dataset = pd.read_csv('loan_data.csv')

# Remove missing values
loan_dataset = loan_dataset.dropna()

# Convert 'Loan_Status' from Y/N to 1/0
loan_dataset.replace({"Loan_Status": {'N': 0, 'Y': 1}}, inplace=True)

# Convert '3+' dependents to 4
loan_dataset['Dependents'].replace('3+', 4, inplace=True)

# Visualization
sns.countplot(x='Education', hue='Loan_Status', data=loan_dataset)
# plt.show()

# Encode other categorical features
loan_dataset.replace({
    'Married': {'No': 0, 'Yes': 1},
    'Gender': {'Male': 1, 'Female': 0},
    'Self_Employed': {'No': 0, 'Yes': 1},
    'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2},
    'Education': {'Graduate': 1, 'NotGraduate': 0}
}, inplace=True)

# Split features and label
X = loan_dataset.drop(columns=['Loan_ID', 'Loan_Status'], axis=1)
Y = loan_dataset['Loan_Status']

# Train-test split
X_train , X_test ,Y_train , Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=2)

# Train SVM model
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train , Y_train)

# Evaluate on training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
# print("Training Accuracy:", training_data_accuracy)

X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
# print("Testing accuracy: ", test_data_accuracy)


# Make a predictive system