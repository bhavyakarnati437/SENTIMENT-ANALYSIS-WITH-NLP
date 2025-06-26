# SENTIMENT-ANALYSIS-WITH-NLP
COMPANY : CODTECH IT SOLUTIONS

NAME : KARNATI BHAVYA

INTERN ID : CT2MTDM438

DOMAIN : Machine Learning

DURATION : 8 WEEKS

MENTOR : NEELA SANTOSH

# DESCRIPTION:
This project focuses on performing Sentiment Analysis on a dataset of customer reviews to classify them as positive or negative using Natural Language Processing (NLP) techniques. Sentiment analysis is a powerful tool used in business, marketing, and social media monitoring to understand how customers feel about products or services based on their written feedback.

The implementation makes use of the TF-IDF vectorization technique to convert text data into numerical features and a Logistic Regression model to classify the sentiments. The project involves several key steps including data preprocessing, vectorization, model training, and evaluation. It demonstrates a complete workflow for solving a real-world NLP problem using Python and Scikit-learn.

# What is Sentiment Analysis?
Sentiment Analysis, also known as opinion mining, is the process of determining whether a piece of text expresses a positive, negative, or neutral opinion. In this project, we simplify it to a binary classification problem (positive or negative), based on labeled customer review data.

This type of task is commonly applied in:

E-commerce product feedback
Social media monitoring
Brand reputation management
Customer satisfaction analysis

# Dataset Description
The dataset used contains text reviews written by customers, with each review labeled as either:
1 — Positive sentiment
0 — Negative sentiment

Typical data samples include review phrases like:
“This product is amazing!” → Positive

“Worst experience ever” → Negative

The dataset may be custom-created, downloaded from open sources (e.g., Kaggle), or synthetic for demo purposes. The reviews are usually stored in a CSV format with two columns: Review (text) and Sentiment (label).

# Project Workflow
Importing Required Libraries:
Libraries such as pandas, numpy, scikit-learn, and matplotlib are used for data handling, model training, and evaluation.

Loading the Dataset:
The dataset is read using pandas.read_csv(), and a quick exploratory data analysis is performed to inspect the number of positive and negative reviews.

Text Preprocessing:
Each review is cleaned by:

Lowercasing
Removing punctuation and special characters
Removing stopwords
Tokenization (splitting into words)

# Vectorization using TF-IDF:
The cleaned text is converted into numerical vectors using TF-IDF (Term Frequency-Inverse Document Frequency), which captures the importance of each word in a document relative to all documents in the dataset.

Splitting the Data:
The data is split into training and testing sets using an 80-20 ratio.

Training Logistic Regression Model:
A LogisticRegression classifier is trained on the TF-IDF vectors from the training data.

Model Evaluation:
Predictions are made on the test data, and performance is evaluated using:

Accuracy
Confusion matrix
Classification report (precision, recall, F1-score)

# Results

The model performs well on unseen reviews, accurately predicting their sentiment. This proves the effectiveness of combining TF-IDF with Logistic Regression for text classification tasks.
Visualization of the confusion matrix and classification report helps in understanding how well the model is distinguishing between positive and negative sentiments.

# OUTPUT
![Image](https://github.com/user-attachments/assets/7d17305c-44a5-4bf6-b732-34a24657f50a)

![Image](https://github.com/user-attachments/assets/7c847f0d-6de0-45d6-9d27-a48ea049346d)
