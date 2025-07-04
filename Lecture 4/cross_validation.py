from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy as np
import pandas as pd

# Load dataset
heart_data = pd.read_csv('heart.csv')

# Split into features and target
X = heart_data.drop(columns=['target'], axis=1)
Y = heart_data['target']

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=5)

# Initialize models
models = [
    LogisticRegression(max_iter=1000),
    SVC(kernel='linear'),
    KNeighborsClassifier(),
    RandomForestClassifier()
]

# Compare models
def compare_models_train_test():
    for model in models:
        model.fit(X_train, Y_train)
        test_pred = model.predict(X_test)
        accuracy = accuracy_score(Y_test, test_pred)
        print('the accuracy of the' , model ,  '=' , accuracy)

# Run comparison
# compare_models_train_test()


# cross validation for logistic regr
# cv_score_lr = cross_val_score(LogisticRegression(max_iter=1000) , X ,Y , cv=5)
# print(cv_score_lr)

# mean_accuracy_lr = sum(cv_score_lr)/len(cv_score_lr)
# mean_accuracy_lr = mean_accuracy_lr*100
# mean_accuracy_lr = round(mean_accuracy_lr , 2)
# print(mean_accuracy_lr)

# cross validation for SVM
# cv_score_svc = cross_val_score(SVC(kernel='linear') , X ,Y , cv=5)
# print(cv_score_lr)

# mean_accuracy_lr = sum(cv_score_svc)/len(cv_score_svc)
# mean_accuracy_lr = mean_accuracy_lr*100
# mean_accuracy_lr = round(mean_accuracy_lr , 2)
# print(mean_accuracy_lr)


# creating a function to compare accuracy scores
models = [
    LogisticRegression(max_iter=1000),
    SVC(kernel='linear'),
    KNeighborsClassifier(),
    RandomForestClassifier()
]

def compare_models_cv_score():
    for model in models:
        cv_score = cross_val_score(model ,X,Y,cv=5)
        mean_accuracy= sum(cv_score) / len(cv_score)
        mean_accuracy = mean_accuracy * 100
        mean_accuracy = round(mean_accuracy, 2)

        print('cross validation score for the ' , model ,  'is : ',cv_score)
        print('Accuray score in % for the ' , model ,  'is : ',mean_accuracy)

compare_models_cv_score( )




