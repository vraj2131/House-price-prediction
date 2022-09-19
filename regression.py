# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 19:14:13 2022

@author: Vraj
"""

"""
Regression
"""

from sklearn.datasets import load_boston
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Boston_P = load_boston()

X = Boston_P.data
Y = Boston_P.target

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, train_size=0.75, random_state=76)

from sklearn.preprocessing import MinMaxScaler

Sc = MinMaxScaler(feature_range=(0,1))

X_train = Sc.fit_transform(X_train)

X_test = Sc.fit_transform(X_test)

Y_train = Y_train.reshape(-1, 1)

Y_train = Sc.fit_transform(Y_train)

"""
Multi Linear Regression (MLR)
"""

from sklearn.linear_model import LinearRegression

Linear_R = LinearRegression()

Linear_R.fit(X_train, Y_train)

Predicted_Values_R = Linear_R.predict(X_test)

Predicted_Values_R = Sc.inverse_transform(Predicted_Values_R)

"""
Evaluation metrics
"""

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import math

Mae = mean_absolute_error(Y_test, Predicted_Values_R)

Mse = mean_squared_error(Y_test, Predicted_Values_R)

Rmse = math.sqrt(Mse)

R2 = r2_score(Y_test, Predicted_Values_R)

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred)/y_true))*100

Mape = mean_absolute_percentage_error(Y_test, Predicted_Values_R)

"""
PLR
"""

from sklearn.datasets import load_boston
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Boston_P = load_boston()

X = Boston_P.data[:, 5]

Y = Boston_P.target

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, train_size=0.75, random_state=76)


from sklearn.preprocessing import PolynomialFeatures

Poly_P = PolynomialFeatures(degree=2)

X_train = X_train.reshape(-1, 1)

Poly_X = Poly_P.fit_transform(X_train)

from sklearn.linear_model import LinearRegression

Linear_R = LinearRegression()

Poly_Lr = Linear_R.fit(Poly_X, Y_train)

X_test = X_test.reshape(-1, 1)

Poly_Xt = Poly_P.fit_transform(X_test)

Predicted_Values_Plr = Poly_Lr.predict(Poly_Xt)

from sklearn.metrics import r2_score

R2 = r2_score(Y_test, Predicted_Values_Plr)

"""
Random Forest
"""

from sklearn.ensemble import RandomForestRegressor

Random_F = RandomForestRegressor(n_estimators=500, max_depth=20, random_state=33)

Random_F.fit(X_train, Y_train)

Predicted_Val_Rf = Random_F.predict(X_test)

Predicted_Val_Rf = Predicted_Val_Rf.reshape(-1, 1)

Predicted_Val_Rf = Sc.inverse_transform(Predicted_Val_Rf)

"""
SVR
"""

from sklearn.svm import SVR

Regressor_Svr = SVR(kernel='rbf')

Regressor_Svr.fit(X_train, Y_train)

Predicted_Values_Svr = Regressor_Svr.predict(X_test)

Predicted_Values_Svr = Predicted_Values_Svr.reshape(-1, 1)

Predicted_Values_Svr = Sc.inverse_transform(Predicted_Values_Svr)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import math

Mae = mean_absolute_error(Y_test, Predicted_Values_Svr)

Mse = mean_squared_error(Y_test, Predicted_Values_Svr)

Rmse = math.sqrt(Mse)

R2 = r2_score(Y_test, Predicted_Values_Svr)

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred)/y_true))*100

Mape = mean_absolute_percentage_error(Y_test, Predicted_Values_Svr)









