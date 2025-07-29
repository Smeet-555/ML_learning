import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# Load dataset
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()
data_frame = pd.DataFrame(breast_cancer_dataset.data, columns=breast_cancer_dataset.feature_names)
data_frame['target'] = breast_cancer_dataset.target

# Split data
X = data_frame.drop(columns='target', axis=1)
Y = data_frame['target']
X = np.asarray(X)
Y = np.asarray(Y)

# Define model and parameter grid
model = SVC()
parameters = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'C': [1, 5, 10, 20]
}

# Grid search
classifier = GridSearchCV(model, parameters, cv=5)
classifier.fit(X, Y)

# Best params and results
# Best params and results
best_parameters = classifier.best_params_
highest_accuracy = classifier.best_score_
results = pd.DataFrame(classifier.cv_results_)

print("Best Parameters:", best_parameters)
print("Highest Accuracy from Grid Search:", highest_accuracy)

# Display selected columns from results
grid_search_result = results[['param_C','param_kernel', 'mean_test_score']]
print(grid_search_result.head())
