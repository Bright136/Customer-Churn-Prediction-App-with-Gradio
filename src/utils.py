import pandas as pd
import numpy as np


# Function to create new columns in the train_data dataframe
def create_new_columns(train_data):
    # Calculate monthly variations by subtracting the product of tenure and monthly charges from total charges
    train_data['Monthly Variations'] = (train_data.loc[:, 'TotalCharges']) -((train_data.loc[:, 'tenure'] * train_data.loc[:, 'MonthlyCharges']))
    
    # Create labels for tenure groups
    labels =['{0}-{1}'.format(i, i+2) for i in range(0, 73, 3)]
    
    # Categorize tenure into groups using pd.cut and assign the labels to the tenure_group column
    train_data['tenure_group'] = pd.cut(train_data['tenure'], bins=(range(0, 78, 3)), right=False, labels=labels)
    
    # Drop the original tenure column
    train_data.drop(columns=['tenure'], inplace=True)
    
    # Return the modified train_data dataframe
    return train_data


# Function to create a processed dataframe from processed_data, train_data, and preprocessor
def create_processed_dataframe(processed_data, train_data, preprocessor):
    # Get the numerical columns from the train_data dataframe
    train_num_cols = train_data.select_dtypes(exclude=['object', 'category']).columns
    
    # Get the names of categorical features from the preprocessor
    cat_features = preprocessor.named_transformers_['categorical']['cat_encoder'].get_feature_names()
    
    # Concatenate the numerical and categorical feature names to create labels
    labels = np.concatenate([train_num_cols, cat_features])
    
    # Create a new dataframe from the processed_data array and assign the labels as column names
    processed_dataframe = pd.DataFrame(processed_data.toarray(), columns=labels)
    
    # Return the processed_dataframe
    return processed_dataframe



def return_features():
    # Define the list of column features
    features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents','tenure', 'PhoneService', 'MultipleLines', 'InternetService', 
                    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection','TechSupport','StreamingTV', 'StreamingMovies', 
                    'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges']
    return features

