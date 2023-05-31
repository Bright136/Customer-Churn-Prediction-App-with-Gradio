import pandas as pd
import numpy as np
import pickle
import os 



DIRPATH = os.path.dirname(os.path.realpath(__file__))

pipeline_pkl = os.path.join(DIRPATH, "assets",  "full_pipeline.pkl")

def load_pickle(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
        return data

preprocessor = load_pickle(pipeline_pkl)


def create_new_columns(train_data):
    train_data['Monthly Variations'] = (train_data.loc[:, 'TotalCharges']) -((train_data.loc[:, 'tenure'] * train_data.loc[:, 'MonthlyCharges']))
    labels =['{0}-{1}'.format(i, i+2) for i in range(0, 73, 3)]
    train_data['tenure_group'] = pd.cut(train_data['tenure'], bins=(range(0, 78, 3)), right=False, labels=labels)
    train_data.drop(columns=['tenure'], inplace=True)
    return train_data




def create_processed_dataframe(processed_data, train_data):
    train_num_cols=train_data.select_dtypes(exclude=['object', 'category']).columns
    cat_features = preprocessor.named_transformers_['categorical']['cat_encoder'].get_feature_names()
    labels = np.concatenate([train_num_cols, cat_features])
    processed_dataframe = pd.DataFrame(processed_data.toarray(), columns=labels)
    return processed_dataframe