import gradio as gr
import pickle 
from gradio.themes.base import Base
import time

# # Define your Gradio block with added rows


# gender:['Female', 'Male']
# SeniorCitizen:[0, 1]
# Partner:['Yes', 'No']
# Dependents:['No', 'Yes']
# PhoneService:['No', 'Yes']
# MultipleLines:['No phone service', 'No', 'Yes']
# InternetService:['DSL', 'Fiber optic', 'No']
# OnlineSecurity:['No', 'Yes', 'No internet service']
# OnlineBackup:['Yes', 'No', 'No internet service']
# DeviceProtection:['No', 'Yes', 'No internet service']
# TechSupport:['No', 'Yes', 'No internet service']
# StreamingTV:['No', 'Yes', 'No internet service']
# StreamingMovies:['No', 'Yes', 'No internet service']
# Contract:['Month-to-month', 'One year', 'Two year']
# PaperlessBilling:['Yes', 'No']
# PaymentMethod:['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
# Churn:['No', 'Yes']
# with gr.Blocks(theme=gr.themes.Default(primary_hue="red", secondary_hue="pink")

def load_pickle(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
        return data
    
pipeline = load_pickle('full_pipeline.pkl')
model = load_pickle('logistic_reg_class_model.pkl')

class Seafoam(Base):
    pass

seafoam = Seafoam()

with gr.Blocks() as demo:
    gr.Markdown(
    """
    # Welcome Cherished User 👋 !
    Start predicting customer churn.
    """)
    with gr.Row():
        gender = gr.Dropdown(label='Gender', choices=['Female', 'Male'], )
        contract  = gr.Dropdown(label='Contract', choices=['Month-to-month', 'One year', 'Two year'])
        internet = gr.Dropdown(label='Gender', choices=['DSL', 'Fiber optic', 'No'])
    with gr.Accordion('Yes or no'):

        with gr.Row():
            OnlineSecurity = gr.Radio(label="Online Security", choices=["Yes", "No", "No internet service"])
            OnlineBackup = gr.Radio(label="Online Backup", choices=["Yes", "No"])
            DeviceProtection = gr.Radio(label="Device Protection", choices=["Yes", "No"])
            TechSupport = gr.Radio(label="Tech Support", choices=["Yes", "No"])
            StreamingTV = gr.Radio(label="TV Streaming", choices=["Yes", "No"])
            StreamingMovies = gr.Radio(label="Movie Streaming", choices=["Yes", "No"]) 
        with gr.Row():
            SeniorCitizen = gr.Radio(label="Senior Citizen", choices=["Yes", "No"])
            Partner = gr.Radio(label="Partner", choices=["Yes", "No"])
            Dependents = gr.Radio(label="Dependents", choices=["Yes", "No"])
            PaperlessBilling = gr.Radio(label="Paperless Billing", choices=["Yes", "No"])
            PhoneService = gr.Radio(label="Phone Service", choices=["Yes", "No"])
            MultipleLines = gr.Radio(label="Multiple Lines", choices=["Yes", "No"]) 
    with gr.Row():
        MonthlyCharges = gr.Number(label="Monthly Charges")
        TotalCharges = gr.Number(label="Total Charges")
        PaymentMethod = gr.Dropdown(label="Payment Method", choices=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    submit_button = gr.Button('Prediction')





