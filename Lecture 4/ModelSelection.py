import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV , RandomizedSearchCV

# Load dataset
heart_data = pd.read_csv('heart.csv')

# Separate features and labels
X = heart_data.drop(columns=['target'], axis=1).values
Y = heart_data['target'].values

# List of models with names
# models = [
#     ("Logistic Regression", LogisticRegression(max_iter=1000)),
#     ("SVM (Linear)", SVC(kernel='linear')),
#     ("KNN", KNeighborsClassifier()),
#     ("Random Forest", RandomForestClassifier(random_state=0))
# ]
#
# # Function to compare models
# def compare_model_using_cross_validation():
#     for name, model in models:
#         cv_scores = cross_val_score(model, X, Y, cv=5)
#         mean_accuracy = round(cv_scores.mean() * 100, 2)
#
#         print(f"Model: {name}")
#         print(f"Cross-validation scores: {cv_scores}")
#         print(f"Mean Accuracy: {mean_accuracy}%")
#         print("-" * 70)
#
# # Call the function
# compare_model_using_cross_validation()


# for heart.csv the RANDOM FOREST CLASSIFIER has the best accuracy score


# comparing the model with different parameters using grid search cv
# list of models
models_list = [
    ("Logistic Regression", LogisticRegression(max_iter=10000)),
    ("SVM", SVC()),
    ("KNN", KNeighborsClassifier()),
    ("Random Forest", RandomForestClassifier(random_state=0))
]

# Dictionary of hyperparameters
model_hyperparameters = {
    'Logistic Regression': {
        'C': [1, 5, 10, 20]
    },
    'SVM': {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'C': [1, 5, 10, 20]
    },
    'KNN': {
        'n_neighbors': [3, 5, 10]
    },
    'Random Forest': {
        'n_estimators': [10, 20, 50, 100]
    }
}

# Function for GridSearchCV
def ModelSelection(list_of_models, hyperparameter_dictionary):
    result = []
    for name, model in list_of_models:
        print(f"\nModel: {name}")
        params = hyperparameter_dictionary[name]
        print("Parameters being tested:", params)

        classifier = GridSearchCV(model, params, cv=5)
        classifier.fit(X, Y)

        result.append({
            'model used': name,
            'highest score': round(classifier.best_score_ * 100, 2),
            'best hyperparameters': classifier.best_params_
        })

    result_dataframe = pd.DataFrame(result, columns=['model used', 'highest score', 'best hyperparameters'])
    return result_dataframe

# Run the function
results_df = ModelSelection(models_list, model_hyperparameters)
print("\nFinal Comparison:\n")
print(results_df)