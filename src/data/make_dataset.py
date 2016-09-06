# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import LabelEncoder

DATA_DIR = '../data/raw'

def load_datasets():
    train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    words = pd.read_csv(os.path.join(DATA_DIR, 'words.csv'), encoding='ISO-8859-1')
    users = pd.read_csv(os.path.join(DATA_DIR, 'users.csv'))

    return (train, test, words, users)

def get_missing_value_features(df):
    'Retuns list of the features with missing values in a dataset'

    missing_val_df = df.isnull().any()
    return list(missing_val_df[missing_val_df == True].index)

def mark_missing_values(df):
    df = df.fillna(-999)
    return df

def prepare_count_features(df):
    features = ['HEARD_OF', 'OWN_ARTIST_MUSIC']
    
    for feature in features:
        df[feature] = df[feature].fillna('')
        feature_dict = df[feature].value_counts().to_dict()
        df[feature] = df[feature].map(lambda x: feature_dict[x])
    
    return df

def fill_missing_values(df, missing_features):
    'Impute missing values for features for a dataframe'

    for feature in missing_features:
        if feature in ['AGE', 'Q16', 'Q18', 'Q19']:
            df[feature] = df[feature].fillna(-999) # to denote that this is a missing value
        else:
            df[feature] = df[feature].fillna('') # empty string to denote missing value for categorical feature.
    
    return df

def parse_music_pref(df, feature_name):
    return df[feature_name].str.findall(r'\d+').map(lambda x: 0 if len(x) == 0 else x[0])


def encode_features(df, feature_names):
    for feature in feature_names:
        lbl = LabelEncoder()
        lbl.fit(df[feature])
        
        df[feature] = lbl.transform(df[feature])
    
    return df

def main():
    print('Prepare Dataset')
    train, test, words, users = load_datasets()
    words = mark_missing_values(words) 
    words = prepare_count_features(words)

    features_with_missing_values = get_missing_value_features(users)
    users = fill_missing_values(users, features_with_missing_values)

    users['LIST_OWN'] = parse_music_pref(users, 'LIST_OWN')
    users['LIST_BACK'] = parse_music_pref(users, 'LIST_BACK')

    users = encode_features(users, ['GENDER', 'WORKING', 'REGION', 'MUSIC'])

    return (train, test, words, users)

