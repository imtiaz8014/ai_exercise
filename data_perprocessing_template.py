# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the Dataset
dataset = pd.read_csv('Data.csv')
X= dataset.iloc[:,:-1].values
y= dataset.iloc[:,-1].values

# Taking care of missing data
################ if you need it uncommnet it
# from sklearn.impute import SimpleImputer
# impute = SimpleImputer(missing_values=np.nan, strategy='mean')
# impute = impute.fit(X[:,1:3])
# X[:,1:3] = impute.transform(X[:,1:3])

 
# Encoding categorical data
################ if you need it uncommnet it
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# ct = ColumnTransformer(
#     transformers = [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],   # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
#     remainder='passthrough'                                         # Leave the rest of the columns untouched
# )
# X = ct.fit_transform(X)
# labelencoder_y = LabelEncoder()
# y = labelencoder_y.fit_transform(y)

# Splitting the dataset into the Training set and test set
from sklearn.model_selection import train_test_split #sklearn.cross_velidation is depricated
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,shuffle=False)

# Feature Scaling 
########## most of the library preprocess it for you but just incase they don't
# from sklearn.preprocessing import StandardScaler 
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)


