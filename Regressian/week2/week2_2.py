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
from math import sqrt
#%%
sales = pd.read_csv("kc_house_data.csv",
                         dtype= {'bathrooms':float,'waterfront':int,
                                 'sqft_above':int, 'sqft_living15':float,
                                 'grade':int, 'yr_renovated':int, 'price':float,
                                 'bedrooms':float, 'zipcode':str, 'long':float,
                                 'sqft_lot15':float, 'sqft_living':float,
                                 'floors':str, 'condition':int, 'lat':float,
                                 'date':str, 'sqft_basement':int, 'yr_built':int,
                                 'id':str, 'sqft_lot':int, 'view':int})
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
def get_numpy_data(data_sframe, features, output):
    data_sframe['constant'] = 1 # add a constant column to an SFrame
    # prepend variable 'constant' to the features list
    features = ['constant'] + features
    # select the columns of data_SFrame given by the ‘features’ list into the SFrame ‘features_sframe’

    # this will convert the features_sframe into a numpy matrix with GraphLab Create >= 1.7!!
    features_matrix = data_sframe.as_matrix(features)
    # assign the column of data_sframe associated with the target to the variable ‘output_sarray’

    # this will convert the SArray into a numpy array:
    output_array = np.array(data_sframe[output]) # GraphLab Create>= 1.7!!
    return(features_matrix, output_array)

#%%
def predict_outcome(feature_matrix, weights):
    predictions = np.dot(feature_matrix,weights)
    return(predictions)
#%%
def feature_derivative(errors, feature):
    derivative = np.dot(feature,errors) *2
    return(derivative)
#%%Although
def regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance):
    converged = False
    weights = np.array(initial_weights)
    while not converged:
        # compute the predictions based on feature_matrix and weights:
        # compute the errors as predictions - output:
        pred = predict_outcome(feature_matrix,weights)
        errors = pred - output
        
        gradient_sum_squares = 0 # initialize the gradient
        # while not converged, update each weight individually:
        for i in range(len(weights)):
            # Recall that feature_matrix[:, i] is the feature column associated with weights[i]
            # compute the derivative for weight[i]:
            derivative = feature_derivative(errors,feature_matrix[:,i])
            # add the squared derivative to the gradient magnitude
            gradient_sum_squares += derivative**2
            # update the weight based on step size and derivative:
            weights[i] -= step_size*derivative
        gradient_magnitude = sqrt(gradient_sum_squares)
        if gradient_magnitude < tolerance:
            converged = True
    return(weights)
#%%
simple_features = ['sqft_living']
my_output= 'price'
(simple_feature_matrix, output) = get_numpy_data(train_data, simple_features, my_output)
initial_weights = np.array([-47000., 1.])
step_size = 7e-12
tolerance = 2.5e7
#%%
simple_weights = regression_gradient_descent(simple_feature_matrix, output,
                                             initial_weights, step_size,tolerance)
#%%
(test_simple_feature_matrix, test_output) = get_numpy_data(test_data, simple_features,my_output)
pred_test_simple = predict_outcome(test_simple_feature_matrix,simple_weights)
#%%
Rss_test= np.sum((test_output-pred_test_simple)**2)
#%%
model_features = ['sqft_living', 'sqft_living15']
my_output = 'price'
(feature_matrix, output) = get_numpy_data(train_data, model_features,my_output)
initial_weights = np.array([-100000., 1., 1.])
step_size = 4e-12
tolerance = 1e9
#%%
regression_weights = simple_weights = regression_gradient_descent(feature_matrix, output,
                                             initial_weights, step_size,tolerance)
#%%
(test_feature_matrix, test_output) = get_numpy_data(test_data, model_features,my_output)
pred_test = predict_outcome(test_feature_matrix,regression_weights)
#%%
Rss_test2= np.sum((test_output-pred_test)**2)
#%%

















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
def get_residual_sum_of_squares_train(input_feature, output):
    X = train_data.as_matrix(input_feature)
    y = np.array(train_data[output])
    reg = LinearRegression()
    reg.fit(X,y)
    pred =reg.predict(X)
    RSS = ((y - pred) ** 2).sum()
    return(RSS)

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
