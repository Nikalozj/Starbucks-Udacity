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



def transform_data_to_fit(portfolio, profile, transcript):
    '''
    DESCRIPTION
    Takes portfolio, profiile and transcript, cleanes and merges them
    
    INPUT
    portfilio - dataframe containing campaign data
    profile - dataframe containing user data
    transcript - dataframe containing transaction data
    
    OUTPUT
    df - cleaned and merged dataset with demographic data present
    profile_cleaned - cleaned profile dataframe
    portfolio_cleaned - cleaned portfolio dataframe
    transcript_cleaned - cleaned transcript dataframe
    '''
    # Clean Dataframes
    portfolio_cleaned = clean_portfolio(portfolio)
    profile_cleaned = clean_profile(profile)
    transcript_cleaned = engineer_prev_resp_for_fit(engineer_event_aggs(process_transcript(transcript), transcript))
    
    # Merge Cleaned Dataframes
    df = transcript_cleaned.merge(portfolio_cleaned, how = 'inner', on = 'offer_id')
    df = df.merge(profile_cleaned, how = 'inner', on = 'person')
    
    # Drop person & offer_id & time
    df.drop(['person', 'offer_id', 'time'], axis = 1, inplace = True)
    
    # Remove users with age == 118
    df = df[df['age'] != 118]
    
    # Get dummies for Gender 
    df = pd.get_dummies(df, drop_first = True)
    
    # Return
    return df, profile_cleaned, portfolio_cleaned, transcript_cleaned


def get_and_transform_data_to_predict(pairs_path, profile_cleaned, portfolio_cleaned, transcript, transcript_cleaned):
    '''
    DESCRIPTION
    Receives person & offer pairs and creates dataframes for training
    
    INPUT
    portfilio - dataframe containing campaign data
    profile - dataframe containing user data
    transcript - dataframe containing transaction data
    transcript_cleaned - cleaned transactions data
    
    OUTPUT
    df - cleaned and merged dataset with demographic data present
    df_ids - dataframe with id, person and offer_id columns
    df_nulls - cleaned and merged dataset with no demographic data
    '''
    # Clean Dataframes
    pairs = pd.read_csv(pairs_path)
    pairs_new = engineer_event_aggs(pairs, transcript)
    pairs_new = engineer_prev_resp(pairs_new, transcript_cleaned)
    
    # Merge Cleaned Dataframes
    df = pairs_new.merge(portfolio_cleaned, how = 'left', on = 'offer_id')
    df = df.merge(profile_cleaned, how = 'left', on = 'person')
    
    # Split into dataframes, one with demographic data and the other without
    df_nulls = df[df['age'] == 118][['person', 'offer_id']]
    df = df[df['age'] != 118]
    
    # Manually assign 0-s to users with no demographic data
    df_nulls['responded'] = 0
    
    # Drop person & offer_id
    df_ids = df[['person', 'offer_id']]
    df.drop(['person', 'offer_id'], axis = 1, inplace = True)
    
    # Get dummies for Gender 
    df = pd.get_dummies(df, drop_first = True)
    
    # Return
    return df, df_ids, df_nulls



def train_model(X_train, y_train):
    '''
    DESCRIPTION
    Train RandomForestClassifier
    
    INPUT
    X_train - featurs for training
    y_train - target column for training
    
    OUTPUT
    rfc_grid - rfc model with best parameters
    '''
    # Start time
    start = time.time()
    
    # Create parameter grid
    param_grid_rfc = { 
    'n_estimators': [20, 40],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [6,8,10],
    'criterion' :['gini', 'entropy']
    }
    
    # Create and train classifier
    rfc_grid = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid_rfc)
    rfc_grid.fit(X_train, y_train)
    
    # End time
    end = time.time()
    
    # Print notifications
    print('Model is ready')
    print('Training time : {}'.format(end - start))
    print('Params : {}'.format(rfc_grid.best_params_))
    print()
    
    # Return model
    return rfc_grid


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
    X = df.drop('responded', axis = 1)
    y = df['responded']
    
    scaler = StandardScaler()
    X[scale_cols] = scaler.fit_transform(X[scale_cols])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    return X_train, X_test, y_train, y_test, scaler


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


def engineer_prev_resp(pairs, transcript_cleaned):
    '''
    DESCRIPTION
    For each person-offer pair returns whether the person previously responded to the same offer as column 'prev_resp'
    prev_resp - 1 if a user previously responded, 0 if didn't
    
    INPUT
    transcript_new - prediction transcript
    transcript_cleand - training transcript, which contains responses for each person-offer pair
    
    OUTPUT
    transcript_new - prediction transcript with 'prev_resp' column added
    '''
    # For each person-offer pair return responded
    transcript_cleaned = transcript_cleaned[transcript_cleaned['responded'] == 1][['person', 'offer_id', 'responded']].drop_duplicates()
    
    # Add responded column
    pairs = pairs.merge(transcript_cleaned, how = 'left', on = ['person', 'offer_id'])
    pairs.rename({'responded' : 'prev_resp'}, axis = 1, inplace = True)
    pairs['prev_resp'].fillna(0, inplace = True)
    
    # Return
    return pairs


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