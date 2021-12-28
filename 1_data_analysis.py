#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 14:54:44 2021

@author: abdul
"""

# to handle datasets
import pandas as pd
import numpy as np

# for plotting
import matplotlib.pyplot as plt
import seaborn as sns

# for the yeo-johnson transformation
import scipy.stats as stats

# to display all the columns of the dataframe in the notebook
pd.pandas.set_option('display.max_columns', None)

# load dataset
data = pd.read_csv('train.csv')

# rows and columns of the data
print(data.shape)

# visualise the dataset
data.head()

# drop id, it is just a number given to identify each house
data.drop('Id', axis=1, inplace=True)

data.shape

# histogran to evaluate target distribution
data['SalePrice'].hist(bins=50, density=True)
plt.ylabel('Number of houses')
plt.xlabel('Sale Price')
plt.show()

# let's transform the target using the logarithm
np.log(data['SalePrice']).hist(bins=50, density=True)
plt.ylabel('Number of houses')
plt.xlabel('Log of Sale Price')
plt.show()


# let's identify the categorical variables
# we will capture those of type *object*

cat_vars = [var for var in data.columns if data[var].dtype == 'O']

# lets add MSSubClass to the list of categorical variables
cat_vars = cat_vars + ['MSSubClass']

len(cat_vars)

# cast all variables as categorical
data[cat_vars] = data[cat_vars].astype('O')


# now let's identify the numerical variables
num_vars = [var for var in data.columns if var not in cat_vars and var != 'SalePrice']

# number of numerical variables
len(num_vars)

len(num_vars)+len(cat_vars)



#---------------------------------------MISSING VALUE------------------------------

# make a list of the variables that contain missing values
vars_with_na = [var for var in data.columns if data[var].isnull().sum() > 0]

# determine percentage of missing values (expressed as decimals)
# and display the result ordered by % of missin data

data[vars_with_na].isnull().mean().sort_values(ascending=False)

# plot

data[vars_with_na].isnull().mean().sort_values(
    ascending=False).plot.bar(figsize=(10, 4))
plt.ylabel('Percentage of missing data')
plt.axhline(y=0.90, color='r', linestyle='-')
plt.axhline(y=0.50, color='g', linestyle='-')

plt.show()


# now we can determine which variables, from those with missing data,
# are numerical and which are categorical

cat_na = [var for var in cat_vars if var in vars_with_na]
num_na = [var for var in num_vars if var in vars_with_na]

print('Number of categorical variables with na: ', len(cat_na))
print('Number of numerical variables with na: ', len(num_na))


def analyse_na_value(df, var):

    # copy of the dataframe, so that we do not override the original data
    df = df.copy()
    # let's make an interim variable that indicates 1 if the
    # observation was missing or 0 otherwise
    df[var] = np.where(df[var].isnull(), 1, 0)

    # let's compare the median SalePrice in the observations where data is missing
    # vs the observations where data is available

    # determine the median price in the groups 1 and 0,
    # and the standard deviation of the sale price,
    # and we capture the results in a temporary dataset
    tmp = df.groupby(var)['SalePrice'].agg(['mean', 'std'])

    # plot into a bar graph
    tmp.plot(kind="barh", y="mean", legend=False,
             xerr="std", title="Sale Price", color='green')

    plt.show()


for var in vars_with_na:
    analyse_na_value(data, var)


#NUMERICAL VARIABLES IN DATASET    
print('Number of numerical variables: ', len(num_vars))

#  YEAR   visualise the numerical variables
data[num_vars].head()
year_vars = [var for var in num_vars if 'Yr' in var or 'Year' in var]
year_vars

# plot median sale price vs year in which it was sold
plt.figure(figsize=(15,5))
data.groupby('YrSold')['SalePrice'].median().plot()
plt.ylabel('Median House Price')

# plot median sale price vs year in which it was built
plt.figure(figsize=(15,5))
data.groupby('YearBuilt')['SalePrice'].median().plot()
plt.ylabel('Median House Price')


#DISCRETE VARIABLES
#  let's male a list of discrete variables
discrete_vars = [var for var in num_vars if len(
    data[var].unique()) < 20 and var not in year_vars]


print('Number of discrete variables: ', len(discrete_vars))

data[discrete_vars].head()

for var in discrete_vars:
    # make boxplot with Catplot
    sns.catplot(x=var, y='SalePrice', data=data, kind="box", height=4, aspect=1.5)
    # add data points to boxplot with stripplot
    sns.stripplot(x=var, y='SalePrice', data=data, jitter=0.1, alpha=0.3, color='k')
    plt.show()


#CONTINUOUS VARIABLES

# make list of continuous variables
cont_vars = [var for var in num_vars if var not in discrete_vars+year_vars]

print('Number of continuous variables: ', len(cont_vars))

data[cont_vars].head()

data[cont_vars].hist(bins=30, figsize=(15,15))
plt.show()

'''
The variables are not normally distributed. 
And there are a particular few that are extremely skewed like 3SsnPorch, ScreenPorch and MiscVal.

Sometimes, transforming the variables to improve the value spread, improves the model performance. 
But it is unlikely that a transformation will help change the distribution of the 
super skewed variables dramatically.
'''


super_skewed = ['BsmtFinSF2', 'LowQualFinSF', 'EnclosedPorch',
    '3SsnPorch', 'ScreenPorch', 'MiscVal']

cont_vars1=[x for x in cont_vars if x not in super_skewed]


# Let's go ahead and analyse the distributions of the variables
# after applying a yeo-johnson transformation

# temporary copy of the data
tmp = data.copy()

for var in cont_vars:
    # transform the variable - yeo-johsnon
    tmp[var], param = stats.yeojohnson(data[var])

    
# plot the histograms of the transformed variables
tmp[cont_vars1].hist(bins=30, figsize=(15,15))
plt.show()


#ORIGINAL VS TRANSFORMED

for var in cont_vars1:
    
    plt.figure(figsize=(12,4))
    
    # plot the original variable vs sale price    
    plt.subplot(1, 2, 1)
    plt.scatter(data[var], np.log(data['SalePrice']))
    plt.ylabel('Sale Price')
    plt.xlabel('Original ' + var)

    # plot transformed variable vs sale price
    plt.subplot(1, 2, 2)
    plt.scatter(tmp[var], np.log(tmp['SalePrice']))
    plt.ylabel('Sale Price')
    plt.xlabel('Transformed ' + var)
                
    plt.show()



'By eye, the transformations seems to improve the relationship only for LotArea.'


'''
Let's transform SUPER SKEWED into binary variables and see how predictive they are:
'''


for var in super_skewed:    
    tmp = data.copy()    
    # map the variable values into 0 and 1
    tmp[var] = np.where(data[var]==0, 0, 1)
    
    # determine mean sale price in the mapped values
    tmp = tmp.groupby(var)['SalePrice'].agg(['mean', 'std'])

    # plot into a bar graph
    tmp.plot(kind="barh", y="mean", legend=False,
             xerr="std", title="Sale Price", color='green')

    plt.show()


#Categorical Variable
    
print('Number of categorical variables: ', len(cat_vars))
data[cat_vars].head()

data[cat_vars].nunique().sort_values(ascending=False).plot.bar(figsize=(15,5))


# re-map strings to numbers, which determine quality

qual_mappings = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5, 'Missing': 0, 'NA': 0}

qual_vars = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
             'HeatingQC', 'KitchenQual', 'FireplaceQu',
             'GarageQual', 'GarageCond']

for var in qual_vars:
    data[var] = data[var].map(qual_mappings)

exposure_mappings = {'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4, 'Missing': 0, 'NA': 0}

var = 'BsmtExposure'

data[var] = data[var].map(exposure_mappings)


finish_mappings = {'Missing': 0, 'NA': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}

finish_vars = ['BsmtFinType1', 'BsmtFinType2']

for var in finish_vars:
    data[var] = data[var].map(finish_mappings)
    
garage_mappings = {'Missing': 0, 'NA': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}

var = 'GarageFinish'

data[var] = data[var].map(garage_mappings)

fence_mappings = {'Missing': 0, 'NA': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4}

var = 'Fence'

data[var] = data[var].map(fence_mappings)

qual_vars  = qual_vars + finish_vars + ['BsmtExposure','GarageFinish','Fence']




   
    












