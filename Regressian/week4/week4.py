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
train_data = pd.read_csv("wk3_kc_house_train_data.csv",dtype= dtype_dict)
train_valid_shuffled = pd.read_csv("wk3_kc_house_train_valid_shuffled.csv",dtype= dtype_dict)
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
sales = pd.read_csv('kc_house_data.csv', dtype=dtype_dict)
sales = sales.sort_values(['sqft_living','price'])
#sales = sales['sqft_living','price']
l2_small_penalty = 1.5e-5
#%%
poly15_data = polynomial_dataframe(sales,['sqft_living'], 15) # use equivalent of `polynomial_sframe`
model = Ridge(alpha=l2_small_penalty, normalize=True)
model.fit(poly15_data, sales['price'])
model_coef = model.coef_
model_intercept = model.intercept_
#%%
poly_1_15 = polynomial_dataframe(dataset1,'sqft_living',15)
poly_2_15 = polynomial_dataframe(dataset2,'sqft_living',15)
poly_3_15 = polynomial_dataframe(dataset3,'sqft_living',15)
poly_4_15 = polynomial_dataframe(dataset4,'sqft_living',15)
#%%
l2_small_penalty_2=1e-9
#%%
l2_large_penalty=1.23e2
#%%    
def ridge_linear_regression(data_set,input_feature, output):
    X = data_set.as_matrix(input_feature)
#    X = X.reshape(-1,1)
    y = np.array(data_set[output])
    reg = Ridge(alpha=l2_large_penalty, normalize=True)
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
model1_intercept, model1_slope = ridge_linear_regression(poly_1_15,features, "price")
model2_intercept, model2_slope = ridge_linear_regression(poly_2_15,features, "price")
model3_intercept, model3_slope = ridge_linear_regression(poly_3_15,features, "price")
model4_intercept, model4_slope = ridge_linear_regression(poly_4_15,features, "price")
#%%


#%%
def get_residual_sum_of_squares_valid(data_set,valid,l2_penalty,input_feature, output):
    X = data_set.as_matrix(input_feature)
    #X = X.reshape(-1,1)
    y = np.array(data_set[output])
    X_valid = valid.as_matrix(input_feature)
    y_valid = np.array(valid[output])
    reg = Ridge(alpha=l2_penalty, normalize=True)
    reg.fit(X,y)
    pred =reg.predict(X_valid)
    RSS = ((y_valid - pred) ** 2).sum()
    #RSS = 0
    return RSS
#%%
def k_fold_cross_validation(k, l2_penalty, data, output):
    n = len(data)
    #k = 10 # 10-fold cross-validation
    RSS =[]
    for i in range(k):
        start = (n*i)//k
        end = (n*(i+1))//k-1
        print (i, (start, end))
        poly_data = polynomial_dataframe(data,['sqft_living'], 15)
        valid_15 = poly_data[start:end+1]
        poly15 = poly_data[0:start].append(poly_data[end+1:n])
        #print(train.head(),"\n",valid.head())
        #poly15 = polynomial_dataframe(train,['sqft_living'], 15)
        #valid_15 = polynomial_dataframe(valid,['sqft_living'], 15)
        
        #print((poly15==np.NAN).shape)
        #print(poly15.shape,"\n",valid_15.shape)
        #print(poly15.head(),"\n",valid_15.head())
        RSS.append(get_residual_sum_of_squares_valid(poly15,valid_15,l2_penalty,features,output))
    return sum(RSS)/10.0
#%%
l2_penalty_arr = np.logspace(3, 9, num=13)
k = 10
output = "price"
err = []
for i in l2_penalty_arr:
   err.append((k_fold_cross_validation(k, i, train_valid_shuffled,output),i))
err.sort()
#%%
poly_train = polynomial_dataframe(train_data,['sqft_living'], 15)
poly_test = polynomial_dataframe(test_data,['sqft_living'], 15)
test_err = get_residual_sum_of_squares_valid(poly_train,poly_test,1000,features,output)
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
