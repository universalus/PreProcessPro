
import numpy as np
import pandas as pd

from scipy import stats

from loguru import logger
from timeit import default_timer as timer

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from sklearn.preprocessing import MinMaxScaler, StandardScaler



def handle_duplicates(data):
    logger.info('Starting handling duplicates...')
    start = timer()
    try:
        original = data.shape
        data.drop_duplicates(inplace=True, ignore_index=False)
        data = data.reset_index(drop=True)
        new = data.shape
        count = original[0] - new[0]
        if count != 0:
            logger.debug('Deleting {} duplicate(s) succeeded', count)
        else:
            logger.debug('No missing values found')
        end = timer()
        logger.info('Completed handling of duplicates in {} seconds', round(end-start, 6))
    except:
        logger.warning('Handling of duplicates failed')
        
    return data


def remove_outliers_iqr(data, columns=None, iqr_factor=1.5):
    # Remove outliers using the Interquartile Range (IQR) method
    if columns is None:
        columns = data.select_dtypes(include=np.number).columns.tolist()

    for column in columns:
        q1 = np.percentile(data[column], 25)
        q3 = np.percentile(data[column], 75)
        iqr = q3 - q1
        lower_bound = q1 - iqr_factor * iqr
        upper_bound = q3 + iqr_factor * iqr
        data = data.loc[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

    return data


def encode_categorical_features(dataset):
    encoder = LabelEncoder()
    encoded_dataset = dataset.copy()
    for column in encoded_dataset.columns:
        if encoded_dataset[column].dtype == 'object':
            encoded_dataset[column] = encoder.fit_transform(encoded_dataset[column])
    return encoded_dataset


def decode_categorical_features(encoded_dataset, original_dataset):
    encoder = LabelEncoder()
    decoded_dataset = encoded_dataset.copy()
    for column in original_dataset.columns:
        if original_dataset[column].dtype == 'object':
            decoded_dataset[column] = encoder.inverse_transform(original_dataset[column])
    return decoded_dataset


def fill_missing_values_mean(data, column):
    # Replace missing values with mean
    filtered_col = remove_outliers_iqr(data[column].copy(), column)
    
    data = data[column].fillna( filtered_col[column].mean() , inplace=True)
    return data


def fill_missing_values_median(data, column):
    # Replace missing values with median
    data = data[column].fillna( data[column].median() , inplace=True)
    return data


def fill_missing_values_mode(data, column):
    # Replace missing values mode
    data = data[column].fillna( data[column].mode() , inplace=True)
    return data


def fill_correlated_missing_values_regression(data, col, correlated_data, polynomial_degree=2):
    # Indices of missing values in the column
    missing_indices = data[col].loc[data[col].isnull()].index
        
    # Missing column as the target variable
    y = correlated_data[col].values
    correlated_cols=correlated_cols.drop(col)
    # Correlated columns as input features
    X = correlated_data[correlated_cols].values

    # Create polynomial features
    poly_features = PolynomialFeatures(degree=polynomial_degree)
    X_poly = poly_features.fit_transform(X)
    # Create a linear regression model
    model = LinearRegression()
    # Fit the model to the data
    model.fit(X_poly, y)

    for index in missing_indices:
        correlated_vals = data.loc[index, correlated_cols]
        X_pred = poly_features.transform(correlated_vals.values.reshape(1, -1))
        # Predict the missing value using the polynomial regression model
        fill_value = model.predict(X_pred)
        # Fill in the missing value
        data.loc[index, col] = fill_value
        
    return data


def fill_correlated_missing_values_knn(data, col, correlated_data, polynomial_degree=2, k=5):
    # Indices of missing values in the column
    missing_indices = data[col].loc[data[col].isnull()].index

    # Missing column as the target variable
    y = correlated_data[col].values
    correlated_cols = correlated_cols.drop(col)
    # Correlated columns as input features
    X = correlated_data[correlated_cols].values

    # Create polynomial features
    poly_features = PolynomialFeatures(degree=polynomial_degree)
    X_poly = poly_features.fit_transform(X)
    # Create a k-NN regression model
    model = KNeighborsRegressor(n_neighbors=k)
    # Fit the model to the data
    model.fit(X_poly, y)

    for index in missing_indices:
        correlated_vals = data.loc[index, correlated_cols]
        X_pred = correlated_vals.values.reshape(1, -1)
        # Predict the missing value using the k-NN regression model
        fill_value = model.predict(X_pred)[0]
        # Fill in the missing value
        data.loc[index, col] = fill_value
    
    return data


def find_correlated_cols(data, column, correlation_coefficient):
    # Correlation coefficients between the missing column and other columns
    correlated_cols = data.corr().abs().loc[column]
    # Filter correlated columns with correlation > 0.80
    correlated_cols = correlated_cols[correlated_cols > correlation_coefficient].index
    # Data containing both missing column and correlated columns without missing values
    correlated_data = data[correlated_cols].dropna()
    return correlated_data


def fill_missing_values(data, original_data, correlation_coefficient=0.8):
    # Identify columns with missing values
    missing_cols = data.columns[data.isnull().any()]
    
    for col in missing_cols:
        correlation_data=find_correlated_cols(data, col, correlation_coefficient)
        if(correlation_data is None):
            if(original_data[col]=='int' or original_data[col]=='float'):
                data = fill_missing_values_mean(data, col)
                # data = fill_missing_values_median(data, col)
            else:
                data = fill_missing_values_mode(data, col)
        else:
            if(original_data[col]=='int' or original_data[col]=='float'):
                data = fill_correlated_missing_values_regression(data, col, correlation_data)
            else:
                data = fill_correlated_missing_values_knn(data, col, correlation_data)
    return data


def handle_missing_values(data):
    logger.info('Starting handling missing values...')
    start = timer()
    try:
        encoded_data = encode_categorical_features(data)
        encoded_data = fill_missing_values(encoded_data, data)
        data = decode_categorical_features(encoded_data, data)
        end = timer()
        logger.info('Completed handling of missing values in {} seconds', round(end-start, 6))
    except:
        logger.warning('Handling of missing values failed')
    return data


def replace_outliers_iqr(data, columns=None, iqr_factor=1.5):
    # Remove outliers using the Interquartile Range (IQR) method
    if columns is None:
        columns = data.select_dtypes(include=np.number).columns.tolist()

    for column in columns:
        q1 = np.percentile(data[column], 25)
        q3 = np.percentile(data[column], 75)
        iqr = q3 - q1
        lower_bound = q1 - iqr_factor * iqr
        upper_bound = q3 + iqr_factor * iqr
        data.loc[(data['value'] < lower_bound) | (data['value'] > upper_bound), 'value'] = np.nan

    return data


def handle_outliers_replacing(data):
    logger.info('Starting handling outliers...')
    start = timer()
    try:
        data = replace_outliers_iqr(data)
        data = handle_missing_values(data)
        end = timer()
        logger.info('Completed handling of outliers in {} seconds', round(end-start, 6))
    except:
        logger.warning('Handling of outliers failed')
    return data


def handle_outliers(data):
    logger.info('Starting handling outliers...')
    start = timer()
    try:
        data = remove_outliers_iqr(data)
        end = timer()
        logger.info('Completed handling of outliers in {} seconds', round(end-start, 6))
    except:
        logger.warning('Handling of outliers failed')
    return data


# def normalize_data(data):
#     # Implement normalization techniques
#     # For example, you can use Min-Max or Z-score normalization
#     # to normalize numerical features
#     return data

def scale_data(data):
    # Perform Min-Max normalization on numerical features
    scaler = MinMaxScaler()

    # Perform Z-score normalization on numerical features
    # scaler = StandardScaler()

    numerical_cols = data.select_dtypes(include=['float', 'int']).columns
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    return data


def normalize_data(data):
    logger.info('Starting normalizing numerical values in data by Min-Max algorithm...')
    start = timer()
    try:
        data = scale_data(data)
        end = timer()
        logger.info('Completed normalizing data in {} seconds', round(end-start, 6))
    except:
        logger.warning('Normalizing data failed')
    return data


 
def clear_data(data):
    data = handle_duplicates(data)
    data = handle_missing_values(data)
    data = handle_outliers(data)
    data = normalize_data(data)
    return data

