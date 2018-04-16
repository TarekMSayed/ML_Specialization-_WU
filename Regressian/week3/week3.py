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
dtype_dict= {'bathrooms':float,'waterfront':int,
                                 'sqft_above':int, 'sqft_living15':float,
                                 'grade':int, 'yr_renovated':int, 'price':float,
                                 'bedrooms':float, 'zipcode':str, 'long':float,
                                 'sqft_lot15':float, 'sqft_living':float,
                                 'floors':str, 'condition':int, 'lat':float,
                                 'date':str, 'sqft_basement':int, 'yr_built':int,
                                 'id':str, 'sqft_lot':int, 'view':int}
#%%
train_data = pd.read_csv("wk3_kc_house_train_data.csv",dtype= dtype_dict)
test_data = pd.read_csv("wk3_kc_house_test_data.csv",dtype= dtype_dict)
validation_data = pd.read_csv("wk3_kc_house_valid_data.csv",dtype= dtype_dict)
dataset1 = pd.read_csv("wk3_kc_house_set_1_data.csv",dtype= dtype_dict)
dataset2 = pd.read_csv("wk3_kc_house_set_2_data.csv",dtype= dtype_dict)
dataset3 = pd.read_csv("wk3_kc_house_set_3_data.csv",dtype= dtype_dict)
dataset4 = pd.read_csv("wk3_kc_house_set_4_data.csv",dtype= dtype_dict)
#%%
def polynomial_dataframe(data_set, feature, degree): # feature is pandas.Series type
    # assume that degree >= 1
    # initialize the dataframe:
    poly_dataframe = pd.DataFrame()
    # and set poly_dataframe['power_1'] equal to the passed feature
    poly_dataframe['power_1'] = data_set[feature]
    # first check if degree > 1
    if degree > 1:
        # then loop over the remaining degrees:
        for power in range(2, degree+1):
            # first we'll give the column a name:
            name = 'power_' + str(power)
            # assign poly_dataframe[name] to be feature^power; use apply(*)
            poly_dataframe[name] = data_set[feature]**power
            ...
    poly_dataframe['price'] = data_set['price']
    return poly_dataframe
#%%
poly_1_15 = polynomial_dataframe(dataset1,'sqft_living',15)
poly_2_15 = polynomial_dataframe(dataset2,'sqft_living',15)
poly_3_15 = polynomial_dataframe(dataset3,'sqft_living',15)
poly_4_15 = polynomial_dataframe(dataset4,'sqft_living',15)
#%%    
def simple_linear_regression(data_set,input_feature, output):
    X = data_set.as_matrix(input_feature)
#    X = X.reshape(-1,1)
    y = np.array(data_set[output])
    reg = LinearRegression()
    reg.fit(X,y)
    slope = reg.coef_
    intercept = reg.intercept_
    plt.figure()
    plt.plot(data_set['power_1'],data_set['price'],'.',
             data_set['power_1'], reg.predict(X),'-')
    return(intercept, slope)
#%%
features = ['power_1','power_2','power_3','power_4','power_5','power_6',
            'power_7','power_8','power_9','power_10','power_11','power_12',
            'power_13','power_14','power_15']
model1_intercept, model1_slope = simple_linear_regression(poly_1_15,features, "price")
model2_intercept, model2_slope = simple_linear_regression(poly_2_15,features, "price")
model3_intercept, model3_slope = simple_linear_regression(poly_3_15,features, "price")
model4_intercept, model4_slope = simple_linear_regression(poly_4_15,features, "price")


#%%
def get_residual_sum_of_squares_valid(data_set,input_feature, output):
    X = data_set.as_matrix(input_feature)
    #X = X.reshape(-1,1)
    y = np.array(data_set[output])
    X_valid = valid.as_matrix(input_feature)
    y_valid = np.array(validation_data[output])
    reg = LinearRegression()
    reg.fit(X,y)
    pred =reg.predict(X_valid)
    RSS = ((y_valid - pred) ** 2).sum()
    #RSS = 0
    return RSS
#%%
poly1 = polynomial_dataframe(train_data,'sqft_living',1)
valid1 =  polynomial_dataframe(validation_data,'sqft_living',1)
RSS1=get_residual_sum_of_squares_valid(poly1,['power_1'],"price")
#%%
RSS = []
for power in range(2,16):
  poly = polynomial_dataframe(train_data,'sqft_living',power)
  print(poly.head())
  valid =  polynomial_dataframe(validation_data,'sqft_living',power)
  RSS.append(get_residual_sum_of_squares_valid(poly,features[:power-1],"price"))
#%%
RSS.insert(0,RSS1)
#%%
def get_residual_sum_of_squares_test(data_set,input_feature, output):
    X = data_set.as_matrix(input_feature)
    y = np.array(data_set[output])
    X_test = test6.as_matrix(input_feature)
    y_test = np.array(test_data[output])
    reg = LinearRegression()
    reg.fit(X,y)
    pred =reg.predict(X_test)
    RSS = ((y_test - pred) ** 2).sum()
    return(RSS)
#%%
poly6 = polynomial_dataframe(train_data,'sqft_living',6)
test6 =  polynomial_dataframe(test_data,'sqft_living',6)
RSS_test6 = get_residual_sum_of_squares_test(poly6,features[:5],"price")