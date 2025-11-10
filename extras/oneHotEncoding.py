import numpy as np
import pandas as pd
from pyarrow import int32
from streamlit import columns
from zmq.backend import first

dataset = pd.read_csv('cars.csv')

# print(dataset.head())


# ONE HOT ENCODING USING PANDAS
OHE = pd.get_dummies(dataset , columns=['fuel' , 'owner'])

# print(OHE)

# K-1 ONE HOT ENCODING
OHE_drop = pd.get_dummies(dataset , columns=['fuel' , 'owner'] , drop_first='true')
# print(OHE_drop)


# OHE USING SKLEARN

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
X_train , X_test , Y_train , Y_test = train_test_split(dataset.iloc[:,0:4] , dataset.iloc[:,-1] , test_size=0.2, random_state=2)

# print(X_train.head())

ohe_sklearn = OneHotEncoder(drop='first', sparse_output = False  ,dtype=int )
X_train_new = ohe_sklearn.fit_transform(X_train[['fuel','owner']])
X_test_new = ohe_sklearn.transform(X_test[['fuel','owner']])

# print (X_train_new.shape)

np.hstack(np.hstack((X_train[['brand','km_driven']].values,X_train_new)))

counts = dataset['brand'].value_counts()
dataset['brand'].nunique()
threshold = 100
repl = counts[counts <= threshold].index
result = pd.get_dummies(dataset['brand'].replace(repl, 'uncommon')).sample(5)

print((result))

