# Resume Classification Pipeline

## Overview
This project implements a resume classification pipeline using advanced machine learning and natural language processing (NLP) techniques. The goal is to classify resumes into different job-related categories based on their textual content. This approach leverages algorithms such as Random Forest and XGBoost, combined with NLP features like TF-IDF, Count Vectorization, and BERT embeddings.

## Features
- **Data Preprocessing**: Text cleaning, normalization, and lemmatization using NLTK.
- **Feature Extraction**: Combination of TF-IDF, Count Vectorization, and BERT embeddings.
- **Class Imbalance Handling**: Using SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset.
- **Model Training**: Two classifiers - Random Forest and XGBoost are trained to optimize classification performance.
- **Evaluation Metrics**: Accuracy, F1 Score, Precision, Recall, and Confusion Matrix.
- **AWS S3 Integration**: Automated saving of trained models, metrics, and performance plots to AWS S3 bucket.

## Requirements
- Python 3.8+
- pandas
- numpy
- nltk
- scikit-learn
- imbalanced-learn
- seaborn
- matplotlib
- joblib
- boto3
- sentence-transformers
- xgboost

## Installation
Install the required packages via pip:
```bash
pip install pandas numpy nltk scikit-learn imbalanced-learn seaborn matplotlib joblib boto3 sentence-transformers xgboost
```

## Usage
The pipeline script is designed to be executed from the command line. The training and testing datasets should be provided as CSV files.

### Training & Testing
```bash
python pipeline.py <train.csv> <test.csv>
```
Example:
```bash
python pipeline.py train_data.csv test_data.csv
```

## Output
The script generates the following outputs:
- Trained models saved as `.pkl` files.
- Metrics file `metrics.csv` containing model evaluation scores.
- Confusion matrix plot `confusion_matrix.png`.
- Class distribution plot `class_distribution.png`.
- Metrics comparison plot `metrics_plot.png`.
- Cleaned predictions saved as `<test_file>_predictions.csv`.

## AWS S3 Integration
The trained models and output files are automatically uploaded to an S3 bucket specified by the `AWS_BUCKET_NAME` variable. Make sure your AWS credentials are properly configured.

## Model Selection
The pipeline compares the performance of Random Forest and XGBoost using F1 Score as the primary evaluation metric.

## References
1. ResearchGate - Resume Classification using various Machine Learning Algorithms
2. ScienceDirect - Resume Screening using Machine Learning
3. ResearchGate - Resume Classification System using NLP and ML Techniques
4. ScienceDirect - A Machine Learning approach for automation of Resume Recommendation System
5. arXiv - ResumeAtlas: Revisiting Resume Classification with Large-Scale Datasets and Large Language Models



