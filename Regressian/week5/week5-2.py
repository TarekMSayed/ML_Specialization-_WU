#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on DEC 16 16:30:12 2017

@author: atamishpc
"""
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%
dtype_dict= {'bathrooms':float,'waterfront':int,
                                 'sqft_above':int, 'sqft_living15':float,
                                 'grade':int, 'yr_renovated':int, 'price':float,
                                 'bedrooms':float, 'zipcode':str, 'long':float,
                                 'sqft_lot15':float, 'sqft_living':float,
                                 'floors':float, 'condition':int, 'lat':float,
                                 'date':str, 'sqft_basement':int, 'yr_built':int,
                                 'id':str, 'sqft_lot':int, 'view':int}
#%%
train_data = pd.read_csv("kc_house_train_data.csv",dtype= dtype_dict)
test_data = pd.read_csv("kc_house_test_data.csv",dtype= dtype_dict)
sales = pd.read_csv("kc_house_data.csv",dtype= dtype_dict)
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
def predict_output(feature_matrix, weights):
    predictions = np.dot(feature_matrix,weights)
    return(predictions)
#%%
def normalize_features(features):
    norms = np.linalg.norm(features, axis=0)
    normalized_features = features / norms
    return (normalized_features, norms)
#%%
def lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty):
    # compute prediction
    prediction =predict_output(feature_matrix,weights)
    # compute ro[i] = SUM[ [feature_i]*(output - prediction + weight[i]*[feature_i]) ]
    ro_i = (feature_matrix[:,i]*(output - prediction +weights[i]*feature_matrix[:,i])).sum()
    
    if i == 0: # intercept -- do not regularize
        new_weight_i = ro_i
    elif ro_i < -l1_penalty/2.:
        new_weight_i = ro_i + l1_penalty/2
    elif ro_i > l1_penalty/2.:
        new_weight_i =  ro_i - l1_penalty/2
    else:
        new_weight_i = 0.
    
    return new_weight_i
#%% test the function should print 0.425558846691
import math
print (lasso_coordinate_descent_step(1, np.array([[3./math.sqrt(13),1./math.sqrt(10)],                                              [2./math.sqrt(13),3./math.sqrt(10)]]),np.array([1., 1.]), np.array([1., 4.]), 0.1))
#%%
def lasso_cyclical_coordinate_descent(feature_matrix, output, initial_weights, l1_penalty, tolerance):
    weights = np.array(initial_weights) # make sure it's a numpy array
    counter = 0
    while counter < len(weights):
      counter = 0
      for i in range(len(weights)): # loop over each weight
          new_weight =  lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty)
          if weights[i]-new_weight <= tolerance:
            counter += 1
          weights[i] = new_weight
    return weights
#%%
simple_features = ['sqft_living','bedrooms']
my_output = 'price'
(simple_feature_matrix, simple_output) = get_numpy_data(sales, simple_features, my_output)
#(simple_test_feature_matrix, test_output) = get_numpy_data(test_data, simple_features, my_output)
simple_feature_matrix_normlized, simple_norms = normalize_features(simple_feature_matrix)
initial_weights = [1, 4, 1]
predictions = predict_output(simple_feature_matrix_normlized,initial_weights)
ro = np.zeros_like(initial_weights)
for i in range(len(initial_weights)):
  #ro[i] = SUM[ [feature_i]*(output - prediction + w[i]*[feature_i]) ]
  ro[i] = (simple_feature_matrix_normlized[:,i]*(simple_output - predictions + initial_weights[i]*simple_feature_matrix_normlized[:,i])).sum()

print('{:0.2E}'.format((ro*2)[1]))
print('{:0.2E}'.format((ro*2)[2]))
#simple_weights_0_penalty = ridge_regression_gradient_descent(simple_feature_matrix, output, initial_weights, step_size, 0, max_iterations)
#%%
initial_weights = np.zeros(3)
l1_penalty = 1e7
tolerance = 1.0
simple_weights = lasso_cyclical_coordinate_descent(simple_feature_matrix_normlized, simple_output, initial_weights, l1_penalty, tolerance)
simple_predictions = predict_output(simple_feature_matrix_normlized,simple_weights)
RSS_simple = ((simple_output - simple_predictions) ** 2).sum()
#%%
print('{:.0E}'.format(RSS_simple)) # 2e15
#%%
more_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
                 'waterfront', 'view', 'condition', 'grade', 'sqft_above',
                 'sqft_basement', 'yr_built', 'yr_renovated']
(more_feature_matrix, more_output) = get_numpy_data(train_data, more_features, my_output)
(more_test_feature_matrix, test_output) = get_numpy_data(test_data, more_features, my_output)
more_feature_matrix_normlized, more_norms = normalize_features(more_feature_matrix)
#%%
l1_penalty = 1e7
tolerance = 1.0
initial_weights = np.zeros(len(more_features)+1)
weights1e7 = lasso_cyclical_coordinate_descent(more_feature_matrix_normlized, more_output, initial_weights, l1_penalty, tolerance)
print(list(zip(["constant"]+more_features, weights1e7)))
#%%
l1_penalty = 1e8
weights1e8 = lasso_cyclical_coordinate_descent(more_feature_matrix_normlized, more_output, initial_weights, l1_penalty, tolerance)
#%%
l1_penalty = 1e4
tolerance = 5e5
weights1e4 = lasso_cyclical_coordinate_descent(more_feature_matrix_normlized, more_output, initial_weights, l1_penalty, tolerance)
#%%
(feature_matrix, all_output) = get_numpy_data(sales, more_features, my_output)
feature_matrix_normlized, norms = normalize_features(feature_matrix)
test_feature_matrix_normlized, test_norms = normalize_features(more_test_feature_matrix)
weights_normalized1e7 = weights1e7 / more_norms
#%%
test_predictions1e7 = predict_output(more_test_feature_matrix,weights_normalized1e7)
RSS_1e7 = ((test_output - test_predictions1e7) ** 2).sum()
print('{:.1E}'.format(RSS_1e7)) 
#%%
weights_normalized1e8 = weights1e8 / more_norms
test_predictions1e8 = predict_output(more_test_feature_matrix,weights_normalized1e8)
RSS_1e8 = ((test_output - test_predictions1e8) ** 2).sum()
print('{:.1E}'.format(RSS_1e8)) 
#%%
weights_normalized1e4 = weights1e4 / more_norms
test_predictions1e4 = predict_output(more_test_feature_matrix,weights_normalized1e4)
RSS_1e4 = ((test_output - test_predictions1e4) ** 2).sum()
print('{:.1E}'.format(RSS_1e4)) 

