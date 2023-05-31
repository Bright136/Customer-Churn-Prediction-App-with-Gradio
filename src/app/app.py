import os
import sys

# Add the parent directory of the current file to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


import gradio as gr
import pickle 
from gradio.themes.base import Base
import pandas as pd
import numpy as np
from src.utils import create_new_columns, create_processed_dataframe

# Get the directory path of the current file
DIRPATH = os.path.dirname(os.path.realpath(__file__))

# Define the file paths for the pickle files and the CSV file
pipeline_pkl = os.path.join(DIRPATH, "..", "assets", "ml_components", "full_pipeline.pkl")
log_reg = os.path.join(DIRPATH, "..","assets", "ml_components", "logistic_reg_class_model.pkl")
hist_df = os.path.join(DIRPATH, "..", "assets",  "history.csv")

# Function to check if a CSV file exists and append data to it
def check_csv(csv_file, data):
    if os.path.isfile(csv_file):
        # If the file exists, append the data to it without writing the header
        data.to_csv(csv_file, mode='a', header=False, index=False, encoding='utf-8')
    else:
        # If the file doesn't exist, create a new file and write the data to it
        history = data.copy()
        history.to_csv(csv_file, index=False)

# Function to load data from a pickle file
def load_pickle(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
        return data

# Load the pipeline and model from the pickle files
pipeline = load_pickle(pipeline_pkl)
model = load_pickle(log_reg)


def predict_churn(gender, SeniorCitizen, Partner, Dependents, Tenure, PhoneService, MultipleLines, InternetService, 
                  OnlineSecurity, OnlineBackup, DeviceProtection,TechSupport,StreamingTV, StreamingMovies, 
                  Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges):
    # Create a list of input data
    data = [gender, SeniorCitizen, Partner, Dependents, Tenure, PhoneService, MultipleLines, InternetService, 
                   OnlineSecurity, OnlineBackup, DeviceProtection,TechSupport,StreamingTV, StreamingMovies, 
                   Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges]
    
    # Convert the list to a NumPy array
    x = np.array([data])
    
    # Create a DataFrame from the array with column names
    dataframe = pd.DataFrame(x, columns=train_features)
    
    # Convert specific columns to float data type
    dataframe = dataframe.astype({'MonthlyCharges': 'float', 'TotalCharges': 'float', 'tenure': 'float'})
    
    # Create new columns in the DataFrame
    dataframe_ = create_new_columns(dataframe)
    
    try:
        # Apply the pipeline transformation to the processed DataFrame
        processed_data = pipeline.transform(dataframe_)
    except Exception as e:
        raise gr.gradio('Kindly make sure to check/select all')
    else:
        # Check if the history.csv file exists and append the input data to it
        check_csv(hist_df, dataframe)
        
        # Read the history.csv file into a DataFrame
        history = pd.read_csv(hist_df)
        
        # Create a processed DataFrame from the processed_data
        processed_dataframe = create_processed_dataframe(processed_data, dataframe, pipeline)
        
        # Make predictions using the model
        predictions = model.predict_proba(processed_dataframe)
    
    # Return the churn predictions, history DataFrame, and input history
    return round(predictions[0][0], 3), round(predictions[0][1], 3), history.sort_index(ascending=False).head()


# Set the theme for the Gradio interface
theme = gr.themes.Default().set(body_background_fill="#0E1117",
                                 background_fill_secondary="#FFFFFF",
                                 background_fill_primary="#262730",
                                 body_text_color="#FF4B4B",
                                 checkbox_background_color='#FFFFFF', 
                                 button_secondary_background_fill="#FF4B4B")

# Define the list of train features
train_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents','tenure', 'PhoneService', 'MultipleLines', 'InternetService', 
                   'OnlineSecurity', 'OnlineBackup', 'DeviceProtection','TechSupport','StreamingTV', 'StreamingMovies', 
                   'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges']

css = """
.svelte-s1r2yt {font-size: 30px;
                color: white;
                font-weight: 300}
"""


# Create the Gradio interface
with  gr.Blocks(theme=theme, css=css) as demo:
    gr.HTML("""
    <h1 style="color:white; text-align:center">Customer Churn Classification App</h1>
    <h2 style="color:white;">Welcome Cherished User 👋 </h2>
    <h4 style="color:white;">Start predicting customer churn.</h4>
    """)
    
    with gr.Row():
        gender = gr.Dropdown(label='Gender', choices=['Female', 'Male'])
        Contract  = gr.Dropdown(label='Contract', choices=['Month-to-month', 'One year', 'Two year'])
        InternetService = gr.Dropdown(label='Internet Service', choices=['DSL', 'Fiber optic', 'No'])

    with gr.Accordion('Yes or no'):
        with gr.Row():
            OnlineSecurity = gr.Radio(label="Online Security", choices=["Yes", "No", "No internet service"])
            OnlineBackup = gr.Radio(label="Online Backup", choices=["Yes", "No", "No internet service"])
            DeviceProtection = gr.Radio(label="Device Protection", choices=["Yes", "No", "No internet service"])
            TechSupport = gr.Radio(label="Tech Support", choices=["Yes", "No", "No internet service"])
            StreamingTV = gr.Radio(label="TV Streaming", choices=["Yes", "No", "No internet service"])
            StreamingMovies = gr.Radio(label="Movie Streaming", choices=["Yes", "No", "No internet service"]) 
        with gr.Row():
            SeniorCitizen = gr.Radio(label="Senior Citizen", choices=["Yes", "No"])
            Partner = gr.Radio(label="Partner", choices=["Yes", "No"])
            Dependents = gr.Radio(label="Dependents", choices=["Yes", "No"])
            PaperlessBilling = gr.Radio(label="Paperless Billing", choices=["Yes", "No"])
            PhoneService = gr.Radio(label="Phone Service", choices=["Yes", "No"])
            MultipleLines = gr.Radio(label="Multiple Lines", choices=["No phone service", "Yes", "No"]) 
    
    with gr.Row():
        MonthlyCharges = gr.Number(label="Monthly Charges")
        TotalCharges = gr.Number(label="Total Charges")
        Tenure = gr.Number(label='Months of Tenure')
        PaymentMethod = gr.Dropdown(label="Payment Method", choices=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

    submit_button = gr.Button('Prediction')
    
    with gr.Row():
        with gr.Accordion('Churn Prediction', ):
            output1 = gr.Slider(maximum=1,
                                minimum=0,
                                value=0.0,
                                label='Yes')
            output2 = gr.Slider(maximum=1,
                                minimum=0,
                                value=0.0,
                                label='No')
        with gr.Accordion('Input History'):
            output3 = gr.Dataframe()

    submit_button.click(fn=predict_churn, inputs=[gender, SeniorCitizen, Partner, Dependents, Tenure, PhoneService, MultipleLines,     
                                                  InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,TechSupport,StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges], outputs=[output1, output2, output3])




