import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
import streamlit as st
import pandas as pd
import joblib
import boto3
from pipeline import train_model, test_model, TextPreprocessor
from PIL import Image

# AWS Configuration
AWS_BUCKET_NAME = 'resume-enhancer-49b6ce94'
s3 = boto3.client('s3')

# Disable Streamlit's file-watcher to prevent PyTorch conflict
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

# Helper functions for loading artifacts from S3
def download_from_s3(s3_file, local_file):
    try:
        s3.download_file(AWS_BUCKET_NAME, s3_file, local_file)
        st.write(f"✅ Successfully downloaded {s3_file}")
    except Exception as e:
        st.write(f"❌ Error downloading {s3_file}: {str(e)}")

# Load artifacts if they are not already present
def load_artifacts():
    if not os.path.exists('feature_union.pkl'):
        download_from_s3('feature_union.pkl', 'feature_union.pkl')
    if not os.path.exists('label_encoder.pkl'):
        download_from_s3('label_encoder.pkl', 'label_encoder.pkl')

# Preprocess random CSV files
def preprocess_random_csv(file_path):
    st.write("🔍 Preprocessing the uploaded file...")

    # Load the file
    data = pd.read_csv(file_path)
    preprocessor = TextPreprocessor()
    
    # Automatically detect the text column if not found
    if 'Resume_str' not in data.columns:
        st.warning(f"Column 'Resume_str' not found. Please select the text column.")
        text_column = st.selectbox("Select the text column", data.columns)
    else:
        text_column = 'Resume_str'
    
    # Apply preprocessing
    data['cleaned_text'] = data[text_column].apply(preprocessor.preprocess)
    
    # Save the preprocessed data for further testing
    preprocessed_file = "preprocessed_test_data.csv"
    data.to_csv(preprocessed_file, index=False)
    
    st.write("✅ Preprocessing completed successfully.")
    return preprocessed_file

st.title("Resume Classifier")

# Uploading the file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
operation = st.selectbox("Select Operation", ["Train", "Test"])

if uploaded_file:
    # Ensure the 'uploads' directory exists
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
        
    file_path = os.path.join("uploads", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.write("✅ File uploaded successfully!")

    if operation == "Train":
        if st.button("Start Training"):
            train_model(file_path)
            st.success("✅ Training completed and model uploaded to S3!")

    elif operation == "Test":
        load_artifacts()  # Ensure required artifacts are present

        if st.button("Start Testing"):
            # Preprocess the random CSV file first
            preprocessed_file = preprocess_random_csv(file_path)
            
            # Read the preprocessed file to confirm it contains 'cleaned_text'
            preprocessed_data = pd.read_csv(preprocessed_file)
            
            if 'cleaned_text' not in preprocessed_data.columns:
                st.error("Preprocessing failed. Please ensure the correct text column is selected.")
            else:
                # Now call test_model() with the preprocessed file
                test_model(preprocessed_file)
                
                if os.path.exists("metrics.csv"):
                    metrics = pd.read_csv("metrics.csv")
                    st.write("### 📊 Model Performance Metrics")
                    st.write(metrics)
                    
                if os.path.exists("confusion_matrix.png"):
                    st.image("confusion_matrix.png", caption="Confusion Matrix")
                    
                if os.path.exists("class_distribution.png"):
                    st.image("class_distribution.png", caption="Class Distribution")
                
                if os.path.exists("metrics_plot.png"):
                    st.image("metrics_plot.png", caption="Metrics Plot")

                st.success("✅ Testing completed successfully!")

    if os.path.exists("Resume_test_predictions.csv"):
        st.write("### 🔍 Test Predictions")
        predictions = pd.read_csv("Resume_test_predictions.csv")
        st.dataframe(predictions)
        st.download_button("Download Predictions", predictions.to_csv(index=False), file_name="Predictions.csv")
