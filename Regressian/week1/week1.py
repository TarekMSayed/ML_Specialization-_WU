#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 16:30:12 2017

@author: atamishpc
"""
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
 
#%%
train_data = pd.read_csv("kc_house_train_data.csv",
                         dtype= {'bathrooms':float,'waterfront':int,
                                 'sqft_above':int, 'sqft_living15':float,
                                 'grade':int, 'yr_renovated':int, 'price':float,
                                 'bedrooms':float, 'zipcode':str, 'long':float,
                                 'sqft_lot15':float, 'sqft_living':float,
                                 'floors':str, 'condition':int, 'lat':float,
                                 'date':str, 'sqft_basement':int, 'yr_built':int,
                                 'id':str, 'sqft_lot':int, 'view':int})
test_data = pd.read_csv("kc_house_test_data.csv",
                         dtype= {'bathrooms':float,'waterfront':int,
                                 'sqft_above':int, 'sqft_living15':float,
                                 'grade':int, 'yr_renovated':int, 'price':float,
                                 'bedrooms':float, 'zipcode':str, 'long':float,
                                 'sqft_lot15':float, 'sqft_living':float,
                                 'floors':str, 'condition':int, 'lat':float,
                                 'date':str, 'sqft_basement':int, 'yr_built':int,
                                 'id':str, 'sqft_lot':int, 'view':int})
#%%
def simple_linear_regression(input_feature, output):
    X = np.array(train_data[input_feature])
    X = X.reshape(-1,1)
    y = np.array(train_data[output])
    reg = LinearRegression()
    reg.fit(X,y)
    slope = reg.coef_
    intercept = reg.intercept_
    return(intercept, slope)
#%%
input_feature = 'sqft_living'
output = "price"
squarefeet_intercept, squarefeet_slope = simple_linear_regression(input_feature, output)
print("intercept = ", squarefeet_intercept,"\n", "slope = ", squarefeet_slope)
#%%
def get_regression_predictions(input_feature, output, feature_to_pred):
    X = np.array(train_data[input_feature])
    X = X.reshape(-1,1)
    y = np.array(train_data[output])
    reg = LinearRegression()
    reg.fit(X,y)
    predicted_output =reg.predict(np.array([feature_to_pred]).reshape(1,-1))
    return(predicted_output)
#%%
quiz6 = get_regression_predictions(input_feature,output,feature_to_pred=2650)
#%%
def get_residual_sum_of_squares(input_feature, output):
    X = np.array(train_data[input_feature])
    X = X.reshape(-1,1)
    y = np.array(train_data[output])
    reg = LinearRegression()
    reg.fit(X,y)
    pred =reg.predict(X)
    RSS = ((y - pred) ** 2).sum()
    return(RSS)
#%%
quiz7 = get_residual_sum_of_squares(input_feature, output)
#%%
def inverse_regression_predictions(output, intercept, slope):
    estimated_input = (output-intercept)/slope
    return(estimated_input)
#%%
quiz8 = inverse_regression_predictions(800000, squarefeet_intercept, squarefeet_slope)
#%%
def get_residual_sum_of_squares(input_feature, output):
    X = np.array(train_data[input_feature])
    X = X.reshape(-1,1)
    y = np.array(train_data[output])
    X_test = np.array(test_data[input_feature])
    X_test = X_test.reshape(-1,1)
    y_test = np.array(test_data[output])
    reg = LinearRegression()
    reg.fit(X,y)
    pred =reg.predict(X_test)
    RSS = ((y_test - pred) ** 2).sum()
    return(RSS)
#%%
quiz10_1 = get_residual_sum_of_squares(input_feature,output)
input_feature="bedrooms"
quiz10_2 = get_residual_sum_of_squares(input_feature,output)