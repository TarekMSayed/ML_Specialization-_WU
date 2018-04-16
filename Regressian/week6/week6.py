#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 16:30:12 2017

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
train_data = pd.read_csv("kc_house_data_small_train.csv",dtype= dtype_dict)
test_data = pd.read_csv("kc_house_data_small_test.csv",dtype= dtype_dict)
sales = pd.read_csv("kc_house_data_small.csv",dtype= dtype_dict)
validation_data = pd.read_csv("kc_house_data_validation.csv",dtype= dtype_dict)

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
def normalize_features(features):
    norms = np.linalg.norm(features, axis=0)
    normalized_features = features / norms
    return (normalized_features, norms)
#%%
simple_features = ['sqft_living','bedrooms']
features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
            'waterfront', 'view', 'condition', 'grade', 'sqft_above',
            'sqft_basement', 'yr_built', 'yr_renovated', 'lat', 'long',
            'sqft_living15', 'sqft_lot15']
my_output = 'price'
(features_train, train_output) = get_numpy_data(train_data, features, my_output)
(features_test, test_output) = get_numpy_data(test_data, features, my_output)
(features_valid, valid_output) = get_numpy_data(validation_data, features, my_output)

features_train, norms = normalize_features(features_train)
features_test = features_test / norms
features_valid = features_valid / norms
#%%
print (features_test[0])
print (features_train[9])
#%%
x1 = features_test[0]
x2 = features_train[9]
quiz1  =np.sqrt(np.sum((x1-x2)**2))
print('{:0.3f}'.format(quiz1))
#%%
first10 = features_train[0:10]
query = features_test[0]
first10_distance = np.sqrt(np.sum((first10-query)**2,axis = 1))

print(np.where(first10_distance == np.min(first10_distance)))
#%%
def compute_distances(features_instances, features_query):
    distances = np.sqrt(np.sum((features_instances - features_query)**2,axis = 1))
    return distances
#%%
distances1 = compute_distances(features_train, query)
print(distances1[100])
#should contain 0.0237082324496
#%%
query2 = features_test[2]
distances2 = compute_distances(features_train, query2)
print(np.where(distances2 == np.min(distances2)))
print(train_output[382])
#%%
def k_nearest_neighbors(k, feature_train, features_query):
    distances = compute_distances(features_train, features_query)
    neighbors = np.argsort(distances)[:k]
    return neighbors
#%%
print(k_nearest_neighbors(4,features_train,query2))
#%%
def predict_output_of_query(k, features_train, output_train, features_query):
    distances = np.sqrt(np.sum((features_train - features_query)**2,axis = 1))
    neighbors = np.argsort(distances)[:k]
    prediction = np.average(output_train[neighbors])
    return prediction
#%%
print(predict_output_of_query(4,features_train,train_output,query2))
#%%
def predict_output(k, features_train, output_train, features_query):
    predictions = np.zeros(features_query.shape[0])
    for query in range(features_query.shape[0]):
      predictions[query] = predict_output_of_query(k, features_train, output_train, features_query[query])
    return predictions
#%%
first10_query_prediction = predict_output(10,features_train,train_output,features_test[:10])
first10_query_prediction.sort() 
#%%
RSS = []
for k in range(1,16):
  valid_predections = predict_output(k,features_train,train_output,features_valid)
  rss = ((valid_output - valid_predections) ** 2).sum()
  RSS.append((rss,k))
RSS.sort()
print(RSS[0])
#%%
test_predections = predict_output(8,features_train,train_output,features_test)
rss_test = ((test_output - test_predections) ** 2).sum()
#%%
print('{:.2E}'.format(rss_test)) 
#%%

