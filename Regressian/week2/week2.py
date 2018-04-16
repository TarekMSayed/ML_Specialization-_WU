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
# add new features
#    ‘’ = ‘bedrooms’*‘bedrooms’
#    ‘bed_bath_rooms’ = ‘bedrooms’*‘bathrooms’
#    ‘log_sqft_living’ = log(‘sqft_living’)
#    ‘lat_plus_long’ = ‘lat’ + ‘long’
train_data['bedrooms_squared'] = train_data['bedrooms']**2
train_data['bed_bath_rooms'] = train_data['bedrooms']*train_data['bathrooms']
train_data['log_sqft_living'] = np.log(train_data['sqft_living'])
train_data['lat_plus_long'] = train_data['lat']+train_data['long']
test_data['bedrooms_squared'] = test_data['bedrooms']**2
test_data['bed_bath_rooms'] = test_data['bedrooms']*test_data['bathrooms']
test_data['log_sqft_living'] = np.log(test_data['sqft_living'])
test_data['lat_plus_long'] = test_data['lat']+test_data['long']
#%%
#  Quiz1
bedrooms_squared_mean = np.mean(test_data.bedrooms_squared)
bed_bath_rooms_mean = np.mean(test_data.bed_bath_rooms)
log_sqft_living_mean = np.mean(test_data.log_sqft_living)
lat_plus_long_mean = np.mean(test_data.lat_plus_long)

#%%
def simple_linear_regression(input_feature, output):
    X = train_data.as_matrix(input_feature)
#    X = X.reshape(-1,1)
    y = np.array(train_data[output])
    reg = LinearRegression()
    reg.fit(X,y)
    slope = reg.coef_
    intercept = reg.intercept_
    return(intercept, slope)
#%%
# sqft_living’, ‘bedrooms’, ‘bathrooms’, ‘lat’, and ‘long’
model1_input_feature = ['sqft_living','bedrooms','bathrooms','lat','long']
output = "price"
model1_intercept, model1_slope = simple_linear_regression(model1_input_feature, output)
#print("intercept = ", squarefeet_intercept,"\n", "slope = ", squarefeet_slope)
#%%
# Model 2: ‘sqft_living’, ‘bedrooms’, ‘bathrooms’, ‘lat’,‘long’,
# and ‘bed_bath_rooms’
model2_input_feature = ['sqft_living','bedrooms','bathrooms','lat','long',
                        'bed_bath_rooms']
output = "price"
model2_intercept, model2_slope = simple_linear_regression(model2_input_feature, output)
#print("intercept = ", squarefeet_intercept,"\n", "slope = ", squarefeet_slope)
#%%
# Model 3: ‘sqft_living’, ‘bedrooms’, ‘bathrooms’, ‘lat’,‘long’,
# ‘bed_bath_rooms’, ‘bedrooms_squared’, ‘log_sqft_living’, and ‘lat_plus_long’
model3_input_feature = ['sqft_living','bedrooms','bathrooms','lat','long',
                        'bed_bath_rooms','bedrooms_squared',
                        'log_sqft_living','lat_plus_long']
output = "price"
model3_intercept, model3_slope = simple_linear_regression(model3_input_feature, output)
#print("intercept = ", squarefeet_intercept,"\n", "slope = ", squarefeet_slope)

#%%
def get_residual_sum_of_squares_train(input_feature, output):
    X = train_data.as_matrix(input_feature)
    y = np.array(train_data[output])
    reg = LinearRegression()
    reg.fit(X,y)
    pred =reg.predict(X)
    RSS = ((y - pred) ** 2).sum()
    return(RSS)
#%%
Rss_train1 = get_residual_sum_of_squares_train(model1_input_feature,output)
Rss_train2 = get_residual_sum_of_squares_train(model2_input_feature,output)
Rss_train3 = get_residual_sum_of_squares_train(model3_input_feature,output)

#%%
def get_residual_sum_of_squares_test(input_feature, output):
    X = train_data.as_matrix(input_feature)
    y = np.array(train_data[output])
    X_test = test_data.as_matrix(input_feature)
    y_test = np.array(test_data[output])
    reg = LinearRegression()
    reg.fit(X,y)
    pred =reg.predict(X_test)
    RSS = ((y_test - pred) ** 2).sum()
    return(RSS)
#%%
Rss_test1 = get_residual_sum_of_squares_test(model1_input_feature,output)
Rss_test2 = get_residual_sum_of_squares_test(model2_input_feature,output)
Rss_test3 = get_residual_sum_of_squares_test(model3_input_feature,output)