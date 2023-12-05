# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the Dataset
dataset = pd.read_csv('Data.csv')
X= dataset.iloc[:,:-1].values
Y= dataset.iloc[:,-1].values

# Taking care of missing data
from sklearn.impute import SimpleImputer
impute = SimpleImputer(missing_values=np.nan, strategy='mean')
impute = impute.fit(X[:,1:3])
X[:,1:3] = impute.transform(X[:,1:3])
 