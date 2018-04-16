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
#from sklearn.linear_model import LinearRegression
#from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
#%%
dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float,
              'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float,
              'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float,
              'floors':float, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int,
              'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}
#%%
testing = pd.read_csv('wk3_kc_house_test_data.csv', dtype=dtype_dict)
training = pd.read_csv('wk3_kc_house_train_data.csv', dtype=dtype_dict)
validation = pd.read_csv('wk3_kc_house_valid_data.csv', dtype=dtype_dict)
#%%
#%%
sales = pd.read_csv('kc_house_data.csv', dtype=dtype_dict)
from math import log, sqrt
sales['sqft_living_sqrt'] = sales['sqft_living'].apply(sqrt)
sales['sqft_lot_sqrt'] = sales['sqft_lot'].apply(sqrt)
sales['bedrooms_square'] = sales['bedrooms']*sales['bedrooms']
sales['floors_square'] = sales['floors']**2
#%%
all_features = ['bedrooms', 'bedrooms_square', 'bathrooms', 'sqft_living', 'sqft_living_sqrt',
            'sqft_lot', 'sqft_lot_sqrt', 'floors', 'floors_square',
            'waterfront', 'view', 'condition', 'grade',
            'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated']

model_all = Lasso(alpha=5e2, normalize=True) # set parameters
model_all.fit(sales[all_features], sales['price']) # learn weights
#%%
model_all_waighs = model_all.coef_ #quiz1 = 3,10,12
#%%
testing['sqft_living_sqrt'] = testing['sqft_living'].apply(sqrt)
testing['sqft_lot_sqrt'] = testing['sqft_lot'].apply(sqrt)
testing['bedrooms_square'] = testing['bedrooms']*testing['bedrooms']
testing['floors_square'] = testing['floors']*testing['floors']

training['sqft_living_sqrt'] = training['sqft_living'].apply(sqrt)
training['sqft_lot_sqrt'] = training['sqft_lot'].apply(sqrt)
training['bedrooms_square'] = training['bedrooms']*training['bedrooms']
training['floors_square'] = training['floors']*training['floors']

validation['sqft_living_sqrt'] = validation['sqft_living'].apply(sqrt)
validation['sqft_lot_sqrt'] = validation['sqft_lot'].apply(sqrt)
validation['bedrooms_square'] = validation['bedrooms']*validation['bedrooms']
validation['floors_square'] = validation['floors']*validation['floors']
#%%

#%%
def get_residual_sum_of_squares_valid(data_set,valid,l1_penalty,input_feature, output):
    X = data_set.as_matrix(input_feature)
    #X = X.reshape(-1,1)
    y = np.array(data_set[output])
    X_valid = valid.as_matrix(input_feature)
    y_valid = np.array(valid[output])
    reg = Lasso(alpha=l1_penalty, normalize=True)
    reg.fit(X,y)
    pred =reg.predict(X_valid)
    RSS = ((y_valid - pred) ** 2).sum()
    #RSS = 0
    return RSS
#%%
l1_penalty_arr = np.logspace(1, 7, num=13)
output = "price"
err = []
for i in l1_penalty_arr:
   err.append((get_residual_sum_of_squares_valid(training,validation,i,all_features, output),i))
err.sort() #quiz2 = 1^10
#%%
model = Lasso(alpha=10, normalize=True) # set parameters
model.fit(training[all_features], training['price']) # learn weights
quiz3 = np.count_nonzero(model.coef_) + np.count_nonzero(model.intercept_) #quiz3 = 15
#%%
#%%
def get_nonzero(data_set,l1_penalty,input_feature, output):
    X = data_set.as_matrix(input_feature)
    y = np.array(data_set[output])
    reg = Lasso(alpha=l1_penalty, normalize=True)
    reg.fit(X,y)
    return np.count_nonzero(reg.coef_) + np.count_nonzero(reg.intercept_)
#%%
max_nonzeros = 7
l1_penalty_arr = np.logspace(1, 4, num=20)
output = "price"
max_nonzeros_arr = []
for i in l1_penalty_arr:
  non_zero = get_nonzero(training,i,all_features, output)
  max_nonzeros_arr.append((non_zero, i))
max_nonzeros_arr.sort() #quiz2 = 1^10
#%%
l1_penalty_min = 784.75997035146065
l1_penalty_max = 127.42749857031335
#%%
err2 = []
for i in np.linspace(l1_penalty_min,l1_penalty_max,20):
  non_zero = get_nonzero(training,i,all_features, output)
  if non_zero == max_nonzeros:
    err2.append((get_residual_sum_of_squares_valid(training,validation,i,all_features, output),i,non_zero))
err2.sort() # 156.10909673930752
#%%
max_nonzero_model = Lasso(alpha=err2[0][1], normalize=True) # set parameters
max_nonzero_model.fit(training[all_features], training['price']) # learn weights
quiz = max_nonzero_model.coef_ # 0,2,3,9,10,12,15