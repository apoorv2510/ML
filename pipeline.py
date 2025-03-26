import os
import sys
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import gc
import boto3
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import sparse
import warnings
warnings.filterwarnings('ignore')

# AWS S3 Configuration
AWS_BUCKET_NAME = 'resume-enhancer-49b6ce94'
MODEL_VERSION = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

# Initialize S3 Client
s3 = boto3.client('s3')

# Enhanced Text Preprocessing
class TextPreprocessor:
    def __init__(self):
        nltk.download('stopwords')
        nltk.download('wordnet')
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def preprocess(self, text):
        if pd.isna(text):
            return ""
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)  # Keep only letters and whitespace
        text = text.lower().strip()
        text = ' '.join([self.lemmatizer.lemmatize(word) 
                       for word in text.split() 
                       if word not in self.stop_words and len(word) > 2])
        return text

# Enhanced Visualization and Output Formatter
class ResultVisualizer:
    @staticmethod
    def create_metrics_plot(metrics, filename='metrics_plot.png'):
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x=list(metrics.keys()), y=list(metrics.values()))
        plt.title('Model Performance Metrics', pad=20)
        plt.ylim(0, 1)
        plt.ylabel('Score')
        
        # Add value labels on top of bars
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.3f}", 
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='center', 
                       xytext=(0, 10), 
                       textcoords='offset points')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        return filename

    @staticmethod
    def clean_predictions(df, text_column, prediction_column='Predicted_Category'):
        # Create clean output dataframe
        output_df = pd.DataFrame({
            'Original_Text': df[text_column],
            'Cleaned_Text': df['cleaned_text'],
            'Predicted_Category': df[prediction_column]
        })
        
        # Add probabilities if they exist
        prob_cols = [col for col in df.columns if col.startswith('prob_')]
        if prob_cols:
            output_df = pd.concat([output_df, df[prob_cols]], axis=1)
            # Format probability columns
            for col in prob_cols:
                output_df[col] = output_df[col].apply(lambda x: f"{x:.3f}")
        
        # Sort by predicted category and reset index
        output_df = output_df.sort_values('Predicted_Category').reset_index(drop=True)
        
        return output_df

# Embedding Transformer Class
class EmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def fit(self, X, y=None):
        return self  # Nothing to fit
    
    def transform(self, X):
        return self.model.encode(X, show_progress_bar=False)

# Feature Union with Proper Transformer Support
class FeatureUnionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, transformers):
        self.transformers = transformers
        
    def fit(self, X, y=None):
        for name, transformer in self.transformers:
            if hasattr(transformer, 'fit'):
                transformer.fit(X, y)
        return self
    
    def transform(self, X):
        features = []
        for name, transformer in self.transformers:
            if hasattr(transformer, 'transform'):
                feature = transformer.transform(X)
            else:
                feature = transformer(X)  # Handle callable functions
                
            if sparse.issparse(feature):
                feature = feature.toarray()
            features.append(feature)
        return np.hstack(features)

def upload_to_s3(local_file, s3_file):
    try:
        s3.upload_file(local_file, AWS_BUCKET_NAME, f"{MODEL_VERSION}/{s3_file}")
        print(f"✅ {local_file} uploaded to S3 as {MODEL_VERSION}/{s3_file}.")
    except Exception as e:
        print(f"❌ Error uploading file to S3: {str(e)}")

def train_model(training_file, text_column='Resume_str', label_column='Category'):
    # Initialize components
    preprocessor = TextPreprocessor()
    embed_transformer = EmbeddingTransformer()
    
    # Load and preprocess data
    print("\n🔍 Loading and preprocessing training data...")
    data = pd.read_csv(training_file)
    data = data[[text_column, label_column]].dropna()
    data['cleaned_text'] = data[text_column].apply(preprocessor.preprocess)
    
    # Feature Extraction
    print("🔧 Extracting features...")
    tfidf = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), 
                           min_df=5, max_df=0.7, sublinear_tf=True)
    count_vec = CountVectorizer(max_features=5000, ngram_range=(1, 2))
    
    # Feature Union with proper transformers
    feature_union = FeatureUnionTransformer([
        ('tfidf', tfidf),
        ('count', count_vec),
        ('embeddings', embed_transformer)
    ])
    
    X = feature_union.fit_transform(data['cleaned_text'])
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data[label_column])
    
    # Save artifacts
    print("💾 Saving preprocessing artifacts...")
    joblib.dump(feature_union, 'feature_union.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')
    upload_to_s3('feature_union.pkl', 'feature_union.pkl')
    upload_to_s3('label_encoder.pkl', 'label_encoder.pkl')
    
    # Train-test split
    print("✂️ Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, 
                                                     random_state=42, stratify=y)
    
    # Handle class imbalance
    print("⚖️ Balancing classes...")
    smote = SMOTE(random_state=42, sampling_strategy='not majority')
    X_res, y_res = smote.fit_resample(X_train, y_train)
    
    # Model Training
    print("\n🤖 Training models...")
    models = {
        'random_forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'xgboost': XGBClassifier(random_state=42, eval_metric='mlogloss', 
                               use_label_encoder=False)
    }
    
    best_model = None
    best_score = 0
    best_model_name = ''
    
    for name, model in models.items():
        print(f"\n=== Training {name} ===")
        model.fit(X_res, y_res)
        val_score = f1_score(y_val, model.predict(X_val), average='weighted')
        print(f"Validation F1: {val_score:.4f}")
        
        if val_score > best_score:
            best_score = val_score
            best_model = model
            best_model_name = name
    
    # Save best model
    print("\n💾 Saving best model...")
    model_file = f'{best_model_name}_model.pkl'
    joblib.dump(best_model, model_file)
    upload_to_s3(model_file, model_file)
    
    print(f"\n🎉 Best model ({best_model_name}) trained with F1: {best_score:.4f}")

def test_model(testing_file, text_column='Resume_str', label_column='Category'):
    # Initialize visualizer
    visualizer = ResultVisualizer()
    
    # Load artifacts
    print("\n🔍 Loading model artifacts...")
    feature_union = joblib.load('feature_union.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    model_files = [f for f in os.listdir() if f.endswith('_model.pkl')]
    model_file = max(model_files, key=os.path.getctime)
    model = joblib.load(model_file)
    
    # Process test data
    print("🧹 Preprocessing test data...")
    test_data = pd.read_csv(testing_file)
    test_data['cleaned_text'] = test_data[text_column].apply(TextPreprocessor().preprocess)
    X_test = feature_union.transform(test_data['cleaned_text'])
    
    # Generate predictions
    print("🔮 Making predictions...")
    y_pred = model.predict(X_test)
    test_data['Predicted_Category'] = label_encoder.inverse_transform(y_pred)
    
    # Add prediction probabilities if available
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_test)
        for i, class_name in enumerate(label_encoder.classes_):
            test_data[f'prob_{class_name}'] = proba[:, i]
    
    # Clean and format predictions
    print("✨ Formatting output...")
    clean_data = visualizer.clean_predictions(test_data, text_column)
    
    # Calculate metrics if labels available
    if label_column in test_data.columns:
        y_test = label_encoder.transform(test_data[label_column])
        
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred, average='weighted'),
            'Precision': precision_score(y_test, y_pred, average='weighted'),
            'Recall': recall_score(y_test, y_pred, average='weighted')
        }
        
        # Save metrics and visualizations
        print("📊 Generating visualizations...")
        metrics_file = 'metrics.csv'
        pd.DataFrame.from_dict(metrics, orient='index', columns=['Score']).to_csv(metrics_file)
        upload_to_s3(metrics_file, f'results/{MODEL_VERSION}/{metrics_file}')
        
        # Create and save visualizations
        metrics_plot = visualizer.create_metrics_plot(metrics)
        upload_to_s3(metrics_plot, f'results/{MODEL_VERSION}/metrics_plot.png')
        
        # Enhanced confusion matrix
        plt.figure(figsize=(12, 10))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=label_encoder.classes_,
                   yticklabels=label_encoder.classes_)
        plt.title('Confusion Matrix', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        cm_file = 'confusion_matrix.png'
        plt.savefig(cm_file, dpi=300, bbox_inches='tight')
        plt.close()
        upload_to_s3(cm_file, f'results/{MODEL_VERSION}/{cm_file}')
        
        # Class distribution plot
        plt.figure(figsize=(12, 6))
        test_data['Predicted_Category'].value_counts().plot(kind='bar')
        plt.title('Predicted Class Distribution', pad=20)
        plt.xlabel('Category')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        dist_file = 'class_distribution.png'
        plt.savefig(dist_file, dpi=300, bbox_inches='tight')
        plt.close()
        upload_to_s3(dist_file, f'results/{MODEL_VERSION}/{dist_file}')
    
    # Save cleaned predictions
    predictions_file = testing_file.replace('.csv', '_predictions.csv')
    clean_data.to_csv(predictions_file, index=False)
    upload_to_s3(predictions_file, f'results/{MODEL_VERSION}/{os.path.basename(predictions_file)}')
    
    # Print summary
    print("\n📋 TEST RESULTS SUMMARY")
    print("=====================")
    if label_column in test_data.columns:
        print(pd.DataFrame.from_dict(metrics, orient='index', columns=['Score']))
    print(f"\n📁 Output files saved to S3 bucket under: {MODEL_VERSION}/results/")
    print(f"✅ Predictions saved to: {predictions_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python pipeline.py <train.csv> <test.csv>")
        sys.exit(1)
    
    print("\n" + "="*50)
    print("🚀 Resume Classification Pipeline")
    print(f"📅 Version: {MODEL_VERSION}")
    print("="*50 + "\n")
    
    train_model(sys.argv[1])
    test_model(sys.argv[2])
    gc.collect()
    
    print("\n" + "="*50)
    print("🏁 Pipeline completed successfully!")
    print("="*50)