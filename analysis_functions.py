import pandas as pd
import numpy as np
import math
import json
from datetime import date, datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

import time
import classifier_functions as cf
import warnings
warnings.filterwarnings('ignore')


def get_avg_spent(df):
    '''
    DESCRIPTION
    Takes transcript dataframe and returns average amount spent by each user
    
    INPUT
    df - transcript dataframe
    
    OUTPUT
    avg_spent - average amount spent dataframe with two columns - person, avg_spent
    
    '''
    # Get transactions
    df = df[df['event'] == 'transaction']
    
    # Get amounts from value column dictionaries
    df['avg_spent'] = df['value'].apply(lambda x: x.get('amount'))
    
    # Get average amount spent per user
    avg_spent = df.groupby('person').mean()['avg_spent'].to_frame()
    
    # Return average amount per user
    return avg_spent


def get_trans_cnt(df):
    '''
    DESCRIPTION
    Takes transcript dataframe and returns transaction count per user
    
    INPUT
    df - transcript dataframe
    
    OUTPUT
    trans_cnt - transaction count dataframe with two columns - person, trans_cnt
    
    '''
    # Get transaction count per user
    trans_cnt = df[df['event'] == 'transaction'].groupby('person').count()['event'].to_frame().rename({'event':'trans_cnt'}, axis = 1)
    
    # Return transaction count per user
    return trans_cnt


def process_transcript(df):
    '''
    DESCRIPTION
    From transcript dataframe, for each 'offer received' event, returns whether a user responded to an event
    Adds one column:
    responded - 1 if a user responded to an offer, 0 if didn't
    
    INPUT
    df - dataframe of transactions
    
    OUTPUT
    df_cleaned = cleaned dataframe of transaction
    '''
    # Drop transacation events
    df_cleaned = df[df['event'] != 'transaction']
    
    # Get offer_ids from 'value' column dictionaries
    df_cleaned['offer_id'] = df_cleaned['value'].apply(lambda x: x['offer id'] if 'offer id' in x else x['offer_id'])
    
    # Drop 'Value' column
    df_cleaned = df_cleaned.drop(['value'], axis = 1)
    
    # Sort data for Window Functions
    df_cleaned = df_cleaned.sort_values(['person', 'offer_id', 'time'])
    
    # For each user & offer & time, get the next two actions as a separate column
    df_cleaned['step_1'] = df_cleaned['event'].shift(-1)
    df_cleaned['step_2'] = df_cleaned['event'].shift(-2)
    
    # Leave rows with 'offer receieved' as 'event' and remove 'event' field
    df_cleaned = df_cleaned[df_cleaned['event'] == 'offer received'].drop('event', axis = 1)
    
    # Add column 'responded', logic - if a user viewed and then completed, then 1, else 0
    df_cleaned['responded'] = (df_cleaned['step_1'] == 'offer viewed') & (df_cleaned['step_2'] == 'offer completed')
    df_cleaned['responded'] = df_cleaned['responded'].map({True: 1, False: 0})
    
    # Drop step columns
    df_cleaned = df_cleaned.drop(['step_1', 'step_2'], axis = 1)
    
    # return cleaned df
    return df_cleaned


def engineer_prev_resp_for_fit(df_cleaned):
    '''
    DESCRIPTION
    From cleaned transcript dataframe, for each 'offer received' returns whether a user has previously responded to
    the same offer. Adds a column:
    prev_resp - 1 if a user previously responded to the same offer, 0 if didn't
    
    INPUT
    df_cleaned - cleaned transcript dataframe
    
    OUTPUT
    df_cleaned - cleaned tranascript dataframe with added prev_resp column
    '''
    # Has a user previously responded to the same offer
    df_cleaned['prev_resp'] = df_cleaned.groupby(['person', 'offer_id']).cumsum()['responded']
    df_cleaned['prev_resp'] = df_cleaned['prev_resp'].apply(lambda x: 1 if x > 1 else 0)
    
    return df_cleaned


def split_train_test(df_cleaned):
    '''
    DESCRIPTION
    Splits cleaned transcript dataframe into train and test sets. Offers received on the last day (day 576)
    are classified as test set. Creates two files:
    xtest.csv - csv with 'person' & 'offer_id'
    ytest.csv - csv with 'responded'
    
    INPUT
    df - cleaned transcript dataframe
    
    OUTPUT
    train_df - dataframe to use for training
    '''
    test_df = df_cleaned[df_cleaned['time'] == 576]
    test_df[['person', 'offer_id']].to_csv('data/xtest.csv', index = False)
    test_df[['responded']].to_csv('data/ytest.csv', index = False)
    
    train_df = df_cleaned[df_cleaned['time'] < 576]
    
    return train_df

def engineer_event_aggs(df, transcript):
    '''
    DESCRIPTION
    Adds transaction count and average amount spent columns 
    
    INPUT
    df - transcript dataframe
    
    OUTPUT
    df_new - transcipt dataframe with added columns
    
    '''
    # Get transaction count per user
    cnt_dict = get_trans_cnt(transcript)
    df_new = df.merge(cnt_dict, how = 'left', on = 'person')
    
    # Get average spent per user
    avg_spent_dict = get_avg_spent(transcript)
    df_new = df_new.merge(avg_spent_dict, how = 'left', on = 'person')
    
    # Fillna with 0-s
    df_new.fillna(0, inplace = True)
    
    # Return
    return df_new


def clean_portfolio(df):
    '''
    DESCRIPTION
    Prepare porttfolio dataframe
    
    INPUT
    df - portfolio dataframe
    
    OUTPUT
    df_cleaned - cleaned portfolio dataaframe
    '''
    # Remove offer_type'informational' because users canno't respond to such type
    df_cleaned = df[df['offer_type'] != 'informational']
    
    # Rename id to offer_id to later merge to transactions df
    df_cleaned.rename(columns = {"id": 'offer_id'}, inplace = True)
    
    # Create dummy variables for channels
    df_cleaned['social'] = df_cleaned['channels'].apply(lambda x: 1 if 'social' in x else 0)
    df_cleaned['mobile'] = df_cleaned['channels'].apply(lambda x: 1 if 'mobile' in x else 0)
    
    # Create dummy variable for offer_type
    df_cleaned['bogo'] = df_cleaned['offer_type'].apply(lambda x: 1 if x == 'bogo' else 0)
    
    # Drop channels and offer_type columns
    df_cleaned.drop(['channels', 'offer_type'], axis = 1, inplace = True)
    
    # Return cleaned df
    return df_cleaned

def clean_profile(df):
    '''
    DESCRIPTION
    Prepares profile dataframe
    
    INPUT
    df - profile data
    
    OUTPUT
    df_cleaned - data with no null values
    df_nulls - data with nulls or default values
    '''
    # Rename 'id' to 'person' for later join
    df_new = df.rename(columns = {"id": 'person'})
    
    # Change registration date to days_registered
    df_new['days_registered'] = df_new['became_member_on'].apply(lambda x: (pd.to_datetime('today').normalize() -
                                                     datetime(int(str(x)[0:4]), int(str(x)[4:6]), int(str(x)[6:]))).days)
    
    # Drop beame_member_on
    df_new.drop('became_member_on', axis = 1, inplace = True)
    
    
    # Return
    return df_new
    

def transform_data(portfolio, profile, transcript):
    '''
    DESCRIPTION
    Takes portfolio, profile, tranascript dataframes, merges and creates analytic two datasets, one with demographic data
    and the other without demographiic data.
    
    INPUT
    portfilio - dataframe containing campaign data
    profile - dataframe containing user data
    transcript - dataframe containing transaction data
    
    OUTPUT
    df - cleaned and merged dataset with demographic data present
    df_nulls - cleaned and merged dataset with no demographic data
    '''
    # Clean Dataframes
    portfolio_cleaned = clean_portfolio(portfolio)
    profile_cleaned = clean_profile(profile)
    transcript_cleaned = process_transcript(transcript)
    transcript_cleaned = engineer_prev_resp_for_fit(engineer_event_aggs(transcript_cleaned, transcript))
    
    # Merge Cleaned Dataframes
    df = transcript_cleaned.merge(portfolio_cleaned, how = 'inner', on = 'offer_id')
    df = df.merge(profile_cleaned, how = 'inner', on = 'person')
    
    df = split_train_test(df)
    # Drop ID columns
    df.drop(['person', 'offer_id', 'time'], axis = 1, inplace = True)
    
    # Split into dataframes, one with demographic data and the other without
    df_nulls = df[df['age'] == 118].drop(['gender', 'income'], axis = 1)
    df = df[df['age'] != 118]
    
    # Get dummies for Gender 
    df = pd.get_dummies(df, drop_first = True)
    
    return df, df_nulls


def split_and_scale(df,scale_cols):
    '''
    DESCRIPTION
    Takes dataframe and column names to scale, splits the dataframe into X and y, scales scale_cols for X
    and splits to train and test sets
    
    INPUT
    df - dataframe to split
    scale-cols - columns to scale
    
    OUTPUT
    X_train - featurs for training
    y_train - target column for training
    X_test - features for testing
    y_test - target column for testing
    scaler - column scaler
    
    
    '''
    # Split to X and y
    X = df.drop('responded', axis = 1)
    y = df['responded']
    
    # Scale
    scaler = StandardScaler()
    X[scale_cols] = scaler.fit_transform(X[scale_cols])
    
    # Split to train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    # Return
    return X_train, X_test, y_train, y_test, scaler

def train_and_evaluate_models(X_train, y_train, X_test, y_test):
    '''
    DESCRIPTION
    Takes train and test datasests, trains three models with different parameter sets, prints
    F1 score and accuracy comparison and returns the best model.
    
    INPUT
    X_train - featurs for training
    y_train - target column for training
    X_test - features for testing
    y_test - target column for testing
    
    OUTPUT
    best_model - model with the highest F1 score
    '''
    # Create dictionary to keep scores
    scores = dict()
    
    # Start Time
    start = time.time()
    
    # Print to notify that training has started
    print('Started training...')
    print()
    
    # Create parameter grid for Random Forest Classifier
    param_grid_rfc = { 
    'n_estimators': [20, 40],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [6,8,10],
    'criterion' :['gini', 'entropy']
    }
    
    # Create, fit and train RFC
    rfc_grid = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid_rfc)
    rfc_grid.fit(X_train, y_train)
    y_pred_rfc = rfc_grid.predict(X_test)
    
    # Add results to dictionary
    scores['rfc'] = [f1_score(y_test, y_pred_rfc), accuracy_score(y_test,y_pred_rfc)]
    
    # End time
    end = time.time()
    
    # Print notifications
    print('rfc done.')
    print('time : {}'.format(end - start))
    print('Params : {}'.format(rfc_grid.best_params_))
    print()
    
    # Start time
    start = time.time()
    
    # Create grid for Logistic Regression
    param_grid_log_reg = { 
    'C': [2, 5, 10],
    'solver': ['lbfgs', 'liblinear', 'sag']
    }
    
    # Create, fit and train Logistic Regression model
    log_reg_grid = GridSearchCV(estimator=LogisticRegression(), param_grid=param_grid_log_reg)
    log_reg_grid.fit(X_train, y_train)
    y_pred_log_reg = log_reg_grid.predict(X_test)
    
    # Add results to dictionary
    scores['log_reg'] = [f1_score(y_test, y_pred_log_reg), accuracy_score(y_test,y_pred_log_reg)]
    
    # End time
    end = time.time()
    
    # Print notificatitons
    print('log_reg done.')
    print('time : {}'.format(end - start))
    print('Params : {}'.format(log_reg_grid.best_params_))
    print()
    
    # Start time
    start = time.time()
    
    # Create grid for KNeighborsClasifier
    param_grid_knn = { 
    'n_neighbors': [5, 10],
    'algorithm': ['ball_tree', 'kd_tree'],
    'leaf_size': [30, 50]
    }
    
    # Create, fit and train KNeighbors classifier
    knn_grid = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=param_grid_knn)
    knn_grid.fit(X_train, y_train)
    y_pred_knn = knn_grid.predict(X_test)
    
    # Add results to dictionary
    scores['knn'] = [f1_score(y_test, y_pred_knn), accuracy_score(y_test,y_pred_knn)]
    
    # End time
    end = time.time()
    
    # Print notifications
    print('knn done.')
    print('time : {}'.format(end - start))
    print('Params : {}'.format(knn_grid.best_params_))
    print()
    
    # Create dataframe from scores and display
    df = pd.DataFrame.from_dict(scores, orient='index', columns = ['F1', 'Accuracy'])
    display(df)
    
    # Get best model name
    best_model = df.sort_values('F1', ascending = False).index[0]
    
    # Return best model
    if best_model == 'rfc':
        return rfc_grid
    elif best_model == 'log_reg':
        return log_reg_grid
    else:
        return knn_grid

