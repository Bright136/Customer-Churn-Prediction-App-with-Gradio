import gradio as gr
import pickle 
from gradio.themes.base import Base
# import time
import pandas as pd
import numpy as np
from utils import create_new_columns, create_processed_dataframe
def tenure_values():
    cols = ['0-2', '3-5', '6-8', '9-11', '12-14', '15-17', '18-20', '21-23', '24-26', '27-29', '30-32', '33-35', '36-38', '39-41', '42-44', '45-47', '48-50', '51-53', '54-56', '57-59', '60-62', '63-65', '66-68', '69-71', '72-74']
    return cols

def predict_churn(gender, SeniorCitizen, Partner, Dependents, Tenure, PhoneService, MultipleLines, InternetService, 
                  OnlineSecurity, OnlineBackup, DeviceProtection,TechSupport,StreamingTV, StreamingMovies, 
                  Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges):
    
    data = [gender, SeniorCitizen, Partner, Dependents, Tenure, PhoneService, MultipleLines, InternetService, 
                   OnlineSecurity, OnlineBackup, DeviceProtection,TechSupport,StreamingTV, StreamingMovies, 
                   Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges]
    
    x = np.array([data])
    dataframe = pd.DataFrame(x, columns=train_features)
    dataframe = dataframe.astype({'MonthlyCharges': 'float', 'TotalCharges': 'float', 'tenure': 'float'})
    create_new_columns(dataframe)
    processed_data = pipeline.transform(dataframe)
    processed_dataframe = create_processed_dataframe(processed_data, dataframe)
    predictions = model.predict_proba(processed_dataframe)
    return round(predictions[0][0], 3), round(predictions[0][1], 3)

theme = gr.themes.Soft(
    primary_hue=gr.themes.Color(c100="#fef3c7", c200="#fde68a", c300="#fcd34d", c400="#fbbf24", c50="#dac058", c500="#f59e0b", c600="#d97706", c700="#b45309", c800="#92400e", c900="#78350f", c950="#6c370f"),
    secondary_hue="stone",
    neutral_hue=gr.themes.Color(c100="#d1fae5", c200="#a7f3d0", c300="#6ee7b7", c400="#34d399", c50="#95dfbd", c500="#10b981", c600="#059669", c700="#047857", c800="#065f46", c900="#064e3b", c950="#054436"),
).set(
    input_shadow_dark='*shadow_spread'

)
theme2 = gr.themes.Base(
    primary_hue="orange",
    neutral_hue=gr.themes.Color(c100="#dbeafe", c200="#bfdbfe", c300="#93c5fd", c400="#60a5fa", c50="#296dc7", c500="#3b82f6", c600="#2563eb", c700="#1d4ed8", c800="#1e40af", c900="#1e3a8a", c950="#1d3660"),
)

theme3 = gr.themes.Soft(
    primary_hue="orange") 

theme4 = gr.themes.Default().set(body_background_fill="#0E1117",
                                 background_fill_secondary="#FF4B4B",
                                 background_fill_primary="#262730",
                                 body_text_color="#FF4B4B")


def load_pickle(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
        return data

pipeline = load_pickle('full_pipeline.pkl')
model = load_pickle('logistic_reg_class_model.pkl')

train_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents','tenure', 'PhoneService', 'MultipleLines', 'InternetService', 
                   'OnlineSecurity', 'OnlineBackup', 'DeviceProtection','TechSupport','StreamingTV', 'StreamingMovies', 
                   'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges']


# theme = gr.themes.Base()
with  gr.Blocks(theme=theme4) as demo:
    gr.Markdown(
    """
    # Customer Churn Classification App
    ## Welcome Cherished User 👋 !
    
    Start predicting customer churn.
    """, css= "h1 {color: red}")
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
    print(type([[122, 456]]))
    
    with gr.Row():
        with gr.Accordion('Churn Prediction'):
            output1 = gr.Slider(maximum=1,
                                minimum=0,
                                value=0.0,
                                label='Yes')
            output2 = gr.Slider(maximum=1,
                                minimum=0,
                                value=0.0,
                                label='No')

    submit_button.click(fn=predict_churn, inputs=[gender, SeniorCitizen, Partner, Dependents, Tenure, PhoneService, MultipleLines,     
                                                  InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,TechSupport,StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges], outputs=[output1, output2])

    # if submit_button:
    #     print(predict_churn(gender, SeniorCitizen, Partner, Dependents, Tenure, PhoneService, MultipleLines, InternetService, 
    #               OnlineSecurity, OnlineBackup, DeviceProtection,TechSupport,StreamingTV, StreamingMovies, 
    #               Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges))

# demo = gr.Interface(fn=predict_churn, inputs=[gender, SeniorCitizen, Partner, Dependents, Tenure, PhoneService, MultipleLines,
#                                                    InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,TechSupport,StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges], outputs=['slider', 'slider'], theme=theme)

