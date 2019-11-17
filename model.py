import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from fancyimpute import KNN
import warnings
from datetime import datetime
import calendar
from math import sin, cos, sqrt, atan2, radians,asin
import folium
from folium import FeatureGroup, LayerControl, Map, Marker
from folium.plugins import HeatMap
from folium.plugins import TimestampedGeoJson
from folium.plugins import MarkerCluster
import matplotlib.dates as mdates
import matplotlib as mpl
from datetime import timedelta
import datetime as dt
pd.set_option('display.max_colwidth', -1)
plt.style.use('fivethirtyeight')
from sklearn import preprocessing
warnings.filterwarnings('ignore')
from geopy.distance import geodesic
from geopy.distance import great_circle
from scipy.stats import chi2_contingency
import statsmodels.api as sm
from statsmodels.formula.api import ols
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.externals import joblib
import pickle
from ggplot import *

## loadind the train and test data
train=pd.read_csv("train_cab.csv")
test=pd.read_csv("test.csv")

#droping that entire row of index no 1327 having non date time value
train=train.drop(train.index[1327])

#Changing the required data types of fare amount variable
train['fare_amount'] = pd.to_numeric(train['fare_amount'], errors='coerce')

# droping the variables which have NA as we do not want random imputation of NA values in target variable
# droping the variables which are below 1 
train = train.drop(train[train['fare_amount']<1].index, axis=0)
train=train.dropna(subset=['fare_amount'])

#converting required data type of date time variable of train and test data
train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'])
test['pickup_datetime'] = pd.to_datetime(test['pickup_datetime'])

# Driving new features from the time stamp variable like year, Month, date, day of week and Hour varaiables
# driving new features 
data = [train,test]
for i in data:
    i['Year'] = i['pickup_datetime'].dt.year
    i['Month'] = i['pickup_datetime'].dt.month
#    i['Date'] = i['pickup_datetime'].dt.day
    i['Day of Week'] = i['pickup_datetime'].dt.dayofweek
    i['Hour'] = i['pickup_datetime'].dt.hour
    
    
#writing a function to create a new features in a day like morning, afternoon, evening, night_pm, night_am
#from hour variable
def f(x):
    ''' for sessions in a day using hour column '''
    if (x >=5) and (x <= 11):
        return 'morning'
    elif (x >=12) and (x <=16 ):
        return 'afternoon'
    elif (x >= 17) and (x <= 20):
        return'evening'
    elif (x >=21) and (x <= 23) :
        return 'night_PM'
    elif (x >=0) and (x <=4):
        return'night_AM'
        
#writing a function to drive a new features in a year like spring, summer, fall and winter seasons from month varibale
def g(x):
    ''' for seasons in a year using month column'''
    if (x >=3) and (x <= 5):
        return 'spring'
    elif (x >=6) and (x <=8 ):
        return 'summer'
    elif (x >= 9) and (x <= 11):
        return'fall'
    elif (x >=12)|(x <= 2) :
        return 'winter'
    
#function for creating new features in a week like weekday and weekend from day of week variabale
def h(x):
    ''' for week:weekday/weekend in a day_of_week column '''
    if (x >=0) and (x <= 4):
        return 'weekday'
    elif (x >=5) and (x <=6 ):
        return 'weekend'
    
# applying the f function on Hour variable in both test and train data
train['session'] = train['Hour'].apply(f)
test['session'] = test['Hour'].apply(f)

#applying the g function on Month variable in both test and train data
train['seasons'] = train['Month'].apply(g)
test['seasons'] = test['Month'].apply(g)

#applying the h function on Day of Week variable in both test and train data
train['week'] = train['Day of Week'].apply(h)
test['week'] = test['Day of Week'].apply(h)

#Droping the date time variables and new varibales from which new features have been derived
train=train.drop(['pickup_datetime','Month','Day of Week','Hour'],axis=1)
test=test.drop(['pickup_datetime','Month','Day of Week','Hour'],axis=1)

#removing lat long values which are zero and the values which are not in the range with basic understanding and 
#from the above plots
cnames=['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']
train = train.drop(train[train['pickup_latitude']>90].index, axis=0)
for i in cnames:
    train = train.drop(train[train[i]==0].index, axis=0)
    
# Calculating distance travelled by the cab from pickup and dropoff location using great_circle from geopy library
data = [train, test]
for i in data:
    i['great_circle_dist']=i.apply(lambda x: great_circle((x['pickup_latitude'],x['pickup_longitude']), 
                                                     (x['dropoff_latitude'],   x['dropoff_longitude'])).miles, axis=1)
    i['geodesic_dist']=i.apply(lambda x: geodesic((x['pickup_latitude'],x['pickup_longitude']), 
                                             (x['dropoff_latitude'],   x['dropoff_longitude'])).miles, axis=1)
    
#removing the lat long variables which are used for dist calculation and great_circle distance variable as geopy 
train=train.drop(['pickup_longitude', 'pickup_latitude','dropoff_longitude','dropoff_latitude','great_circle_dist' ],axis=1)
test=test.drop(['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','great_circle_dist' ],axis=1)

#be lessthan one and maximumm would not greater than 6 
train = train.drop(train[train['passenger_count']>6].index, axis=0)
train = train.drop(train[train['passenger_count']<1].index, axis=0)

#Assigning levels to the categories of categorical variable in train data
lis=[]
for i in range(0, train.shape[1]):
    #print(i)
    if(train.iloc[:,i].dtypes=='object'):
        train.iloc[:,i]=pd.Categorical(train.iloc[:,i])
        #print(marketing[[i]])
        train.iloc[:,i]=train.iloc[:,i].cat.codes
        train.iloc[:,i]=train.iloc[:,i].astype('object')
        
        lis.append(train.columns[i])
        
#Assigning levels to the categories of a categorical variable in test data
lis=[]
for i in range(0, test.shape[1]):
    #print(i)
    if(test.iloc[:,i].dtypes=='object'):
        test.iloc[:,i]=pd.Categorical(test.iloc[:,i])
        #print(marketing[[i]])
        test.iloc[:,i]=test.iloc[:,i].cat.codes
        test.iloc[:,i]=test.iloc[:,i].astype('object')
        
        lis.append(test.columns[i])
        
# imputatind all the missing values in passenger count variable with knn imputation with k=15
train = pd.DataFrame(KNN(k = 15).fit_transform(train), columns = train.columns, index=train.index)

#defining the treatment for outliers in data
def outlier_treatment(col):
    ''' calculating outlier indices and replacing them with NA  '''
    #Extract quartiles
    q75, q25 = np.percentile(train[col], [75 ,25])
    print(q75,q25)
    #Calculate IQR
    iqr = q75 - q25
    #Calculate inner and outer fence
    minimum = q25 - (iqr*1.5)
    maximum = q75 + (iqr*1.5)
    print(minimum,maximum)
    #Replace with NA
    train.loc[train[col] < minimum,col] = np.nan
    train.loc[train[col] > maximum,col] = np.nan
    
# Giving the outlier treatment to the fare amount and distance values
outlier_treatment('fare_amount')
outlier_treatment('geodesic_dist')

#Imputing all the nan values in the varibales with KNN imputation
train = pd.DataFrame(KNN(k = 3).fit_transform(train), columns = train.columns, index=train.index)

#converting to proper datatypes of all the categorical variables
train['passenger_count']=train['passenger_count'].astype('int').round().astype('object').astype('category')
train['Year']=train['Year'].astype('int').round().astype('object').astype('category')
train['session']=train['session'].astype('int').round().astype('object').astype('category')
train['seasons']=train['seasons'].astype('int').round().astype('object').astype('category')
train['week']=train['week'].astype('int').round().astype('object').astype('category')

#Statistically correlated features move together directionally. Linear models assume feature independence.
#And if features are correlated that could introduce bias into our models.
cat_var=['passenger_count', 'Year', 'session', 'seasons', 'week']

num_var=['fare_amount', 'geodesic_dist']
train[cat_var]=train[cat_var].apply(lambda x: x.astype('category') )

## correlation anlysis
#Correlation plot
df_corr=train.loc[:,num_var]
#Plotting Heatmap of correlation
#set the width and height of the plot
f, ax=plt.subplots(figsize=(7,5))

#generate correlation matrix
corr=df_corr.corr()

#plot using seaborn library
sns.heatmap(corr, mask=np.zeros_like(corr,dtype=np.bool),cmap=sns.diverging_palette(220,10, as_cmap=True),
            square=True, ax=ax)

#chi square testing on categorical  variables
for i in cat_var:
    for j in cat_var:
        if(i != j):
            chi2, p, dof, ex = chi2_contingency(pd.crosstab(train[i], train[j]))
            if(p < 0.05):
                print(i,"and",j,"are dependent on each other with",p,'----Remove')
            else:
                print(i,"and",j,"are independent on each other with",p,'----Keep')
                
#checking for variability in the categories of categorical varibales with ols function from stats model library 
model = ols('fare_amount ~(passenger_count)+(Year)+(seasons)+(week)+(session)',data=train).fit()
aov_table = sm.stats.anova_lm(model)
aov_table

outcome, predictors = dmatrices('fare_amount ~ (geodesic_dist+passenger_count+seasons+week+session+Year)',train, return_type='dataframe')
# calculating VIF for each individual Predictors
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(predictors.values, i) for i in range(predictors.shape[1])]
vif["features"] = predictors.columns
vif

#Creating dummies for each category in passenger_count and time stamp variables, 
#merging dummies dataframe to both train and test dataframe
temp = pd.get_dummies(train['passenger_count'], prefix = 'passenger_count')
train = train.join(temp)
temp = pd.get_dummies(test['passenger_count'], prefix = 'passenger_count')
test = test.join(temp)
temp = pd.get_dummies(train['seasons'], prefix = 'season')
train = train.join(temp)
temp = pd.get_dummies(test['seasons'], prefix = 'season')
test = test.join(temp)
temp = pd.get_dummies(train['week'], prefix = 'week')
train = train.join(temp)
temp = pd.get_dummies(test['week'], prefix = 'week')
test = test.join(temp)
temp = pd.get_dummies(train['session'], prefix = 'session')
train = train.join(temp)
temp = pd.get_dummies(test['session'], prefix = 'session')
test = test.join(temp)
temp = pd.get_dummies(train['Year'], prefix = 'year')
train = train.join(temp)
temp = pd.get_dummies(test['Year'], prefix = 'year')
test = test.join(temp)
#temp = pd.get_dummies(train['Date'], prefix = 'day')
#train = train.join(temp)
#temp = pd.get_dummies(test['Date'], prefix = 'day')
#test = test.join(temp)

# droping the varibales which are used for creating dummies along with passenger_count_1 dummy variables as the fare 
#amount is same as when passngers count is 2 or more
train=train.drop(['passenger_count','Year','session', 'seasons', 'week','passenger_count_1'],axis=1)
test=test.drop(['passenger_count','Year','session', 'seasons', 'week','passenger_count_1'],axis=1)

#divide data into test and train1 from training dat for model training and performence evaluation
train1, test1=train_test_split(train, test_size=0.3)

# definind the rmse error mertric definition
def rmse(y, y_pred):
    return np.sqrt(np.mean(np.square(y - y_pred)))

#defining the R squered formula
def Rsquar (y, y_pred):
    return metrics.r2_score(y, y_pred)

# defining the adj R squared formula
def adjRsquare(y,y_pred):
    return (1-(1-metrics.r2_score(y, y_pred))*(len(train1)-1)/(len(train)-(len(test1))-1))

# training the model using the training dataset
model=sm.OLS(train1.iloc[:,0].astype(float), train1.iloc[:,1:25].astype(float)).fit()


# check the summary of MLR model
model.summary()

#make the predictions on the test data using trained model
predictions_LR=model.predict(test1.iloc[:,1:25])

# training the decision tree regressor model on training data
DT_Model=DecisionTreeRegressor(max_depth=2,).fit(train1.iloc[:,1:25],train1.iloc[:,0])

# predicting the test data targets with trained decision tree regressor 
predictions_DT=DT_Model.predict(test1.iloc[:,1:25])

#training the Random forest regressor model on train data
RF_Model=RandomForestRegressor(n_estimators=500).fit(train1.iloc[:,1:25],train1.iloc[:,0])

#predicting the test cases with trained model on train data
predictions_RF=RF_Model.predict(test1.iloc[:,1:25])

#Random forest has been choosed for the prediction of test data beacause best performence over other models
#training the model on whole train data
RF_Model=RandomForestRegressor(n_estimators=200).fit(train.iloc[:,1:25],train.iloc[:,0])

# prediction of the test data cases fare amount values using the trained model 
predictions_RF_test=RF_Model.predict(test.iloc[:,0:24])

#making the data frame with random forest predictions
pred=pd.DataFrame(predictions_RF_test)

# loading the test data into python environment
test_pred=pd.read_csv("test.csv")

# concatinating the both predictions and test data with pd.concat function
test_data_pred_with_RF = pd.concat([test_pred, pred], axis=1)

# renaming the predicted fareamount column name  as fare_amount_pred
test_data_pred_with_RF=test_data_pred_with_RF.rename(columns = { 0: 'fare_amount_pred'})

#saving the pradicted values in test dat into disc 
test_data_pred_with_RF.to_csv("predictions_RF.csv",index=False)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import requests
import json
from flask import Flask, request, jsonify, render_template
import os

pickle.dump(RF_Model, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))




