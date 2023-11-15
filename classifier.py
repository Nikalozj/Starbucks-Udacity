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



class Classifier():
    '''
    A classifier that uses portfolio, profile and transcript to train RandomForestClassifier and predict whether a user
    will respond to an offer for person & offer_id pairs.
    '''
    
    def __init__(self):
        '''
        DESCRIPTION
        Nothing to see here
        '''
        
    def fit(self, portfolio_path, profile_path, transcript_path):
        '''
        DESCRIPTION
        Reads portfolio, profile and transcript databases, transforms and merges them and trains RandomForestTClassifier
        
        INPUT
        portfolio_path - path to portfolio dataset
        profile_path - path to profile dataset
        transcript_path - path to transcript dataset
        
        OUTPUT
        self.df - cleaned and merged dataset
        self.profile_cleaned - cleaned profile dataset
        self.portfolio_cleaned - cleaned portfolio dataset
        self.transcript_cleaned - cleaned transcript dataset
        self.transcript - transcript dataset
        self.scalee_cols - columns to scale
        self.model - trained rfc modeel
        '''
        # Notification
        print('Importing Data')
        
        # Import and clean datasets
        portfolio = pd.read_json(portfolio_path, orient='records', lines=True)
        profile = pd.read_json(profile_path, orient='records', lines=True)
        transcript = pd.read_json(transcript_path, orient='records', lines=True)
        
        # Uncomment if testing in jupyter notebooks
        transcript = transcript[(transcript['time'] != 576) | (transcript['event'] != 'offer received')]
        
        # Notification
        print('Transforming Data')
        
        # Prepare datasets for training
        self.df, self.profile_cleaned, self.portfolio_cleaned, self.transcript_cleaned = cf.transform_data_to_fit(portfolio, profile, transcript)
        
        # Save transcript
        self.transcript = transcript
        
        # Create scale_cols
        self.scale_cols = ['reward', 'age', 'income', 'days_registered', 'trans_cnt', 'avg_spent']
        
        # Notification
        print('Training RFC Model')
        
        # Split datasets into train and test, scale train data
        X_train, X_test, y_train, y_test, self.scaler = cf.split_and_scale(self.df, self.scale_cols)
        
        # Train model
        self.model = cf.train_model(X_train, y_train)
        
    def predict(self, pairs_path, result_path):
        '''
        DESCRIPTION
        Takes person & offer_id pairs and predicts whether a user responds, 1 if responds, 0 if won't
        
        INPUT
        pairs_path - csv with person & offer_id pairs
        
        OUTPUT
        a csv with person & offer_id & response columns
        '''
        # Get and transfrom data
        df_pred, df_ids, df_pred_nulls = cf.get_and_transform_data_to_predict(pairs_path, self.profile_cleaned, self.portfolio_cleaned, self.transcript, self.transcript_cleaned)
        
        # Scale columns
        df_pred[self.scale_cols] = self.scaler.transform(df_pred[self.scale_cols])
        
        # Predict
        y_pred = self.model.predict(df_pred)
        df_ids['responded'] = y_pred
        
        # Create final dataframe
        predictions = pd.concat([df_ids, df_pred_nulls])
        
        # Export to csv
        predictions.to_csv(result_path)