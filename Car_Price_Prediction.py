# for warnings
import warnings

warnings.simplefilter(action='ignore', category=Warning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.max_rows', None)

df = pd.read_csv("data\CarPrice_Assignment.csv")


def check_df(dataframe, head=5):
    '''
    It is the function that looks at the overall picture with the given dataframe

    Parameters
    ----------
    dataframe: dataframe

    head : function of dataframe

    Returns
    -------
    '''

    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)

# categorical columns
cat_cols = [col for col in df.columns if str(df[col].dtypes) in ['bool', 'object', 'category']]

# fake numerical, actually categorical columns
num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and str(df[col].dtypes) in ['int', 'float']]

# cardinal categorical columns
cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ['object', 'category']]

# all cat_cols
cat_cols = cat_cols + num_but_cat
cat_cols = [col for col in cat_cols if col not in cat_but_car]


######################################
# Analysis of Categorical Variables

def cat_summary(dataframe, col_name, plot=False):
    """
    Function give that counts of unique values and their ratios

    Parameters
    ----------
    dataframe : dataframe
        The dataframe from which variable names are to be retrieved

    col_name : object,bool,int,float,categorical
        Dataframe's columns

    plot : seaborn lib.
        It plots the given columns with 'countplot'. Its initial value is False.

    Returns
    -------

    """

    if dataframe[col_name].dtype == 'bool':
        dataframe[col_name] = dataframe[col_name].astype(int)

        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
        print("##########################################")

        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)

    else:
        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
        print("##########################################")

        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)


# Browse through the categorical columns.
for col in cat_cols:
    cat_summary(df, col, plot=True)

######################################

# Analysis of Numerical Variables

num_cols = [col for col in df.columns if str(df[col].dtypes == ['float', 'int'])]
num_cols = [col for col in num_cols if col not in cat_cols]


def num_summary(dataframe, numerical_col, plot=False):
    """
      Function give that calculating some statistical data like percentile, mean and std of the numerical values

    Parameters
    ----------
    dataframe : dataframe
        The dataframe from which variable names are to be retrieved

    numerical_col : object,bool,int,float,categorical
        dataframe's columns

    plot : seaborn lib.

    Returns
    -------

    """

    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]

    print(dataframe[numerical_col].describe(quantiles).T)
    print("#######################################")

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)
        print("#######################################")


# Just numerical columns
for col in num_cols:
    num_summary(df, col)

######################################

# Data Preprocessing

manufacturer = df['CarName'].apply(lambda x: x.split(' ')[0])
manufacturer_df = pd.DataFrame({'Full Names': df['CarName'],
                                'Brand Names': df['CarName'].apply(lambda x: x.split(' ')[0])})

# Backup
data = df.copy()

# useless columns : CarName
data.drop(columns=['CarName'], axis=1, inplace=True)

# insert Brand_Names
data.insert(3, 'Brand_Names', df['CarName'].apply(lambda x: x.split(' ')[0]))

data.head()

print(data.Brand_Names)

print(data.Brand_Names.unique())

# for nissan
data.Brand_Names = data.Brand_Names.str.lower()

# for other wrong written
data.replace({'maxda': 'mazda',
              'porcshce': 'porsche',
              'toyouta': 'toyota',
              'volkswagen': 'vw',
              'vokswagen': 'vw'}, inplace=True)

# check
data.Brand_Names.unique()

# opinion : effective columns
effective_cols_for_price = ['wheelbase', 'carlength', 'carwidth', 'carheight',
                            'curbweight', 'enginesize', 'boreratio', 'stroke',
                            'compressionratio', 'horsepower', 'peakrpm', 'citympg', 'highwaympg']

# let's examine the effective columns again.

for i, col in enumerate(effective_cols_for_price):
    plt.subplot(5, 3, i + 1)
    plt.title(effective_cols_for_price[i] + '- Price')
    sns.regplot(x=data[col], y=data['price'], data=data)

# for just one table that have all graph.
plt.show()
plt.tight_layout()

# opinion : useless columns for price
useless_columns = ['stroke', 'highwaympg',
                   'car_ID', 'doornumber', 'carheight',
                   'compressionratio', 'symboling', 'citympg',
                   'fuelsystem', 'peakrpm', 'Brand_Names']

# the other columns except useless_columns
usefull_columns = [col for col in data.columns if col not in useless_columns]

# check point
data_new = data[usefull_columns]

# correlation
data_new_corr = data_new.corr()
sns.heatmap(data_new_corr, annot=True)

##################################################

# Feature Engineering

# 1 Horsepower = Torque x R.P.M. / 5252
# Because of this math, lb-ft of torque and horsepower will always be the same at 5,252 RPM.
# https://powertestdyno.com/how-to-calculate-horsepower/

Torque = df['horsepower'] * 5252 / df['peakrpm']
data.insert(10, 'torque', Torque)

# check
data.head()

plt.title('Torque - Price')
sns.regplot(x=data.torque, y=data.price)
plt.show()

plt.title('Torque', fontsize=18)
sns.distplot(data.torque)
plt.show()

######################################################

# pattern identification

automobiles = data_new.copy()

dummies_list = [col for col in cat_cols if col not in useless_columns]
print(dummies_list)

for i in dummies_list:
    temp_df = pd.get_dummies(eval('automobiles' + '.' + i), drop_first=True)

    automobiles = pd.concat([automobiles, temp_df], axis=1)

    automobiles.drop([i], axis=1, inplace=True)

automobiles.head()

# check any base variables
data.aspiration.unique()
# Base variables due to get_dummies exist
# EX.// threw out the std, just created a column called turbo

######################################
# Train-Test Split

from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(automobiles, train_size=0.7, random_state=30)

######################################
# Scaling

# minimization of variable dimensions for multiple linear regression

from sklearn.preprocessing import MinMaxScaler

# minScaled = 0 and max_Scaled = 1
scaler = MinMaxScaler()

scale_cols = [col for col in num_cols if col in data.columns]

train_data[scale_cols] = scaler.fit_transform(train_data[scale_cols])
train_data.head()

y_train = train_data.pop('price')
y_train.head()

X_train = train_data
train_data.head()

###############################################
# Multiple Linear Regression

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

lr = LinearRegression()

lr.fit(X_train, y_train)
# list(zip(X_train,y_train))
##############################

# RFE (Recursive Feature Elemination)

# Backward elimination

rfe = RFE(lr, n_features_to_select=10)
rfe = rfe.fit(X_train, y_train)

# variable selected?
rfe.support_

# Order of selection
rfe.ranking_

# look at the general view of rfe
list(zip(X_train.columns, rfe.support_, rfe.ranking_))

# only selected columns (rfe.support_ = True)
X_train_rfe = X_train[X_train.columns[rfe.support_]]

###############################################

# Analysis of OLS

X_train_rfemodel = X_train_rfe.copy()

# column of 1st for β0
X_train_rfemodel = sm.add_constant(X_train_rfemodel)

lr = sm.OLS(y_train, X_train_rfemodel).fit()

print(lr.summary())


def train_ols(X, y):
    X = sm.add_constant(X)
    lr = sm.OLS(y, X).fit()
    print(lr.summary())


# deleting columns because of p-value and run an OLS again.

X_train_rfemodel = X_train_rfemodel.drop(['two'], axis=1)
train_ols(X_train_rfemodel, y_train)

X_train_rfemodel = X_train_rfemodel.drop(['dohcv'], axis=1)
train_ols(X_train_rfemodel, y_train)

X_train_rfemodel = X_train_rfemodel.drop(['five'], axis=1)
train_ols(X_train_rfemodel, y_train)

##########################################################

# Order of Importance of Coefficients

X_train_final = X_train[['curbweight', 'enginesize', 'horsepower', 'rear', 'four',
                         'six', 'twelve']]

lr_final = LinearRegression()
lr_final.fit(X_train_final, y_train)

lr_final.coef_

coefficients_ = pd.DataFrame(lr_final.coef_, index=['curbweight', 'enginesize', 'horsepower', 'rear', 'four',
                                                    'six', 'twelve'], columns=['Coeff'])

# Final Result
coefficients_.sort_values(by=['Coeff'], ascending=False)

#########################
