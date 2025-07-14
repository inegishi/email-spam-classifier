import os
import math
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime

#from brokenaxes import brokenaxes
from statsmodels.formula import api
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10,6]

import warnings 
warnings.filterwarnings('ignore')


#Data handling
df = pd.read_csv("emails.csv") #[5172 rows x 3002 colums]
# print(df.info) 


df_features = df.drop(columns=["Email No.", "Prediction"]) #(5172,3000)
df_pred = df["Prediction"] #(5172,)
# print(df_pred.shape)
#print(df_letters.shape) 

features = [i for i in df_features.columns]
# print(len(features)) #(3000,)

#shows Y columns have X unique values
# print(df.nunique().value_counts())

# print(df["Prediction"].value_counts()) #0:3672, 1: 1500

X_train,X_test,y_train,y_test = train_test_split(df_features, df_pred, test_size=0.2, shuffle=True,random_state=42)

model = LogisticRegression()
model.fit(X_train,y_train)

Y_pred = model.predict(X_test)

mse = mean_squared_error(y_test,Y_pred)
r2 = r2_score(y_test,Y_pred)
print(f"mse: {mse}, r2: {r2}")

comparison = pd.DataFrame({
    "actual": y_test,
    "predicted": Y_pred
})
print(comparison.head(50))

model.__sizeof__