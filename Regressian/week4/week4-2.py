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
from sklearn.linear_model import Ridge

#%%
dtype_dict= {'bathrooms':float,'waterfront':int,
                                 'sqft_above':int, 'sqft_living15':float,
                                 'grade':int, 'yr_renovated':int, 'price':float,
                                 'bedrooms':float, 'zipcode':str, 'long':float,
                                 'sqft_lot15':float, 'sqft_living':float,
                                 'floors':str, 'condition':int, 'lat':float,
                                 'date':str, 'sqft_basement':int, 'yr_built':int,
                                 'id':str, 'sqft_lot':int, 'view':int}
#%%
train_data = pd.read_csv("kc_house_train_data.csv",dtype= dtype_dict)
test_data = pd.read_csv("kc_house_test_data.csv",dtype= dtype_dict)
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

#Cost(w)= SUM[ (prediction - output)^2 ]+ l2_penalty*(w[0]^2 + w[1]^2 + ... + w[k]^2).
#%%
def feature_derivative_ridge(errors, feature, weight, l2_penalty, feature_is_constant):
    sum_err = np.dot(feature,errors)*2
    if feature_is_constant:
      derivative = sum_err
    else:
      derivative = sum_err + (2*l2_penalty*weight) #axis = 1
    return derivative
#%%
sales = pd.read_csv("kc_house_data.csv",dtype= dtype_dict)
(example_features, example_output) = get_numpy_data(sales, ['sqft_living'], 'price')
my_weights = np.array([1., 10.])
test_predictions = predict_outcome(example_features, my_weights)
errors = test_predictions - example_output # prediction errors

# next two lines should print the same values
print (feature_derivative_ridge(errors, example_features[:,1], my_weights[1], 1, False))
print (np.sum(errors*example_features[:,1])*2+20.)
print ('')

# next two lines should print the same values
print (feature_derivative_ridge(errors, example_features[:,0], my_weights[0], 1, True))
print (np.sum(errors)*2.)
#%%
def ridge_regression_gradient_descent(feature_matrix, output, initial_weights, step_size, l2_penalty, max_iterations=100):
    weights = np.array(initial_weights) # make sure it's a numpy array
        #while not reached maximum number of iterations:
    iteration = 1
    while iteration <= max_iterations:
        # compute the predictions using your predict_output() function
        predictions = predict_outcome(feature_matrix,weights)
        # compute the errors as predictions - output
        errors = predictions - output
        for i in range(len(weights)): # loop over each weight
            # Recall that feature_matrix[:,i] is the feature column associated with weights[i]
            # compute the derivative for weight[i].
            #(Remember: when i=0, you are computing the derivative of the constant!)
            if i == 0:
              derivative = feature_derivative_ridge(errors, feature_matrix[:,i], weights[0], l2_penalty, True)
            else:
              derivative = feature_derivative_ridge(errors, feature_matrix[:,i], weights[i], l2_penalty, False)
            # subtract the step size times the derivative from the current weight
            weights[i] -= step_size*derivative
        iteration += 1
    return weights
#%%
features = ['sqft_living']
output = "price"
#%%
simple_features = ['sqft_living']
my_output = 'price'
(simple_feature_matrix, output) = get_numpy_data(train_data, simple_features, my_output)
(simple_test_feature_matrix, test_output) = get_numpy_data(test_data, simple_features, my_output)
step_size = 1e-12
max_iterations = 1000
initial_weights = np.zeros(2)
simple_weights_0_penalty = ridge_regression_gradient_descent(simple_feature_matrix, output, initial_weights, step_size, 0, max_iterations)
#%%
simple_weights_high_penalty = ridge_regression_gradient_descent(simple_feature_matrix, output, initial_weights, step_size, 1e11, max_iterations)
#%%
plt.plot(simple_feature_matrix,output,'k.',
        simple_feature_matrix,predict_outcome(simple_feature_matrix, simple_weights_0_penalty),'b-',
        simple_feature_matrix,predict_outcome(simple_feature_matrix, simple_weights_high_penalty),'r-')
#%%
pred_test_low = predict_outcome(simple_test_feature_matrix,simple_weights_0_penalty)
pred_test_high = predict_outcome(simple_test_feature_matrix,simple_weights_high_penalty)
Rss_test_low= np.sum((test_output-pred_test_low)**2)
Rss_test_high= np.sum((test_output-pred_test_high)**2)
#%%
model_features = ['sqft_living', 'sqft_living15']
my_output = 'price'
(feature_matrix, output) = get_numpy_data(train_data, model_features, my_output)
(test_feature_matrix, test_output) = get_numpy_data(test_data, model_features, my_output)
step_size = 1e-12
max_iterations = 1000
initial_weights = np.zeros(3)
multiple_weights_0_penalty = ridge_regression_gradient_descent(feature_matrix, output, initial_weights, step_size, 0, max_iterations)
multiple_weights_high_penalty = ridge_regression_gradient_descent(feature_matrix, output, initial_weights, step_size, 1e11, max_iterations)
#%%
multiple_pred_test_low = predict_outcome(test_feature_matrix,multiple_weights_0_penalty)
multiple_pred_test_high = predict_outcome(test_feature_matrix,multiple_weights_high_penalty)
Rss_multiple_test_low= np.sum((test_output-multiple_pred_test_low)**2)
Rss_multiple_test_high= np.sum((test_output-multiple_pred_test_high)**2)
#%%















