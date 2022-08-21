#!/usr/bin/env python
# coding: utf-8

# In[64]:


import pandas as pd
import argparse
from argparse import ArgumentParser
import numpy as np
import math
import re
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise import Reader

def recommender(data, user_id, output_file):
    df = pd.read_csv(data) 
    f = ['count','mean']
    df_model_summary = df.groupby('MODEL/DESC')['Rating'].agg(f)
    model_benchmark = round(df_model_summary['count'].quantile(0.8),0)
    drop_model_list = df_model_summary[df_model_summary['count'] < model_benchmark].index
    df_user_summary = df.groupby('UserID')['Rating'].agg(f)
    user_benchmark = round(df_user_summary['count'].quantile(0.8),0)
    drop_user_list = df_user_summary[df_user_summary['count'] < user_benchmark].index
    reader = Reader()
    data = Dataset.load_from_df(df[['UserID', 'MODEL/DESC', 'Rating']], reader)
    model = SVD(random_state = 42)
    trainset = data.build_full_trainset()
    model.fit(trainset)
    df_title = pd.Series(df["MODEL/DESC"]).unique()
    df_title = pd.DataFrame(df_title, columns=['Items']) 
    df_title.head()
    user = df_title.copy()
    user = user.reset_index()
    user = user[~user['Items'].isin(drop_model_list)]
    user['Estimate_Score'] = user['Items'].apply(lambda x: model.predict(user_id, x).est)
    user_recommendation = user.sort_values('Estimate_Score', ascending=False)
    user_recommendation['user_id'] = user_id
    user_recommendation = user_recommendation.reset_index()
    user_recommendation = user_recommendation[['user_id', 'Items', 'Estimate_Score']]
    user_recommendation.to_csv(swag_output_file, index = False)
#     return user_recommendation


if __name__ == '__main__': 
    parser = argparse.ArgumentParser() 
    parser.add_argument('f1')
    parser.add_argument('f2')
    args = parser.parse_args()
    recommender(args.f1, args.f2)


# In[ ]:




