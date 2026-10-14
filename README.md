# SpamShield
AI-Powered Spam Message Detection System Using Machine Learning and NLP.

SpamShield is a machine learning project designed to detect spam messages and distinguish them from legitimate messages (ham). It uses Natural Language Processing (NLP) and machine learning techniques to analyze message content and classify suspicious messages.

The project explores multiple classification models and combines their predictions to improve spam detection.

# Project Objectives
Develop an automated system to classify messages as spam or legitimate.
Apply Natural Language Processing (NLP) techniques to process text data.
Train and evaluate multiple machine learning models.
Explore ensemble learning by combining model predictions.
Build an interactive interface that allows users to analyze messages.

# Features
Spam Classification: Classifies messages as SPAM or HAM.
Text Preprocessing: Cleans and prepares text for machine learning.
Multiple Models: Uses Logistic Regression, XGBoost, and DistilBERT.
Ensemble Prediction: Combines model outputs into a final classification.
Interactive Interface: Allows users to enter messages and view predictions.
Model Evaluation: Supports evaluation using accuracy, precision, recall, and F1-score.

# Technologies Used
(Technology)	            (Purpose)

Python	                  Core programming language
Pandas	                  Data processing and manipulation
Scikit-learn	            Machine learning and model evaluation
XGBoost	                  Gradient-boosting classification
DistilBERT	              Transformer-based text classification
NLTK	                    Natural Language Processing and text preprocessing
TF-IDF	                  Text feature extraction
FastAPI                   Backend API
HTML, CSS, JavaScript	    User interface

# Machine Learning Models
SpamShield explores three classification models:

1. Logistic Regression
A supervised machine learning algorithm used to classify messages based on their extracted text features.

2. XGBoost
A gradient-boosting algorithm that learns patterns in the training data to distinguish spam from legitimate messages.

3. DistilBERT
A transformer-based language model designed to understand contextual information in text and support spam classification.

4. Ensemble Learning
The project also explores combining predictions from multiple models to produce a final classification.

# Dataset
The project uses the SMS Spam Collection dataset, containing SMS messages labeled as spam or ham.
The data is processed before training to prepare the text for classification.
Some external Data is being added to fill the bias gap.

- The preprocessing pipeline include:
Text normalization and cleaning.
URL handling.
Tokenization.
Stop-word removal.
Lemmatization.
TF-IDF feature extraction for compatible models.

# How It Works
Input: The user enters a message.
Preprocessing: The text is prepared for the selected model.
Feature Extraction: TF-IDF converts text into numerical features for compatible models.
Classification: The trained models analyze the message.
Ensemble Prediction: Model outputs are combined when the ensemble is enabled.
Output: The application displays the final classification as SPAM or HAM.

# Installation and Usage
Prerequisites
-Python 3.10 or another version supported by the project's dependencies.
-Git.
-pip.

1. Clone the Repository
git clone https://github.com/Haneen-Gouda/SpamShield.git
cd SpamShield

2. Create a Virtual Environment
python -m venv .venv
Activate it on Windows:
.venv\Scripts\activate

3. Install Dependencies
pip install -r requirements.txt

4. Train the Models
If the trained model files are not included, run the training script:
python train_models.py
If the project supports training without DistilBERT, you can use:
python train_models.py --skip-bert

5. Start the Application
uvicorn app:app --host 127.0.0.1 --port 8000
Open the following address in your browser:
http://127.0.0.1:8000

# Model Evaluation
The models can be evaluated using the following metrics:
Accuracy: The proportion of correctly classified messages.
Precision: The proportion of messages predicted as spam that are actually spam.
Recall: The proportion of actual spam messages correctly detected.
F1-score: The harmonic mean of precision and recall.

# Model                   Accuracy     Precision     Recall     F1-score
Logistic Regression        98.28%	      95.43%	     94.47%	     94.95%
XGBoost	                   97.42%	      91.63%	     93.47%	     92.54%
DistilBERT	               99.74%	      100.00%	     98.49%  	   99.24%
Ensemble	                 99.31%    	  100.00%	     95.98%	     97.95%

# Future Improvements
Deploy SpamShield as a publicly accessible web application.
Extend classification to additional message types and languages.
Improve generalization to previously unseen spam messages.
Add user feedback to support future model improvements.
Investigate explainable AI techniques to help users understand predictions.

# Limitations
SpamShield is an educational machine learning project. Its predictions are not guaranteed to be correct, and legitimate messages may occasionally be classified as spam or vice versa.
Model performance depends on the quality and diversity of the training data.

# Author
Haneen Gouda
AI Engineering Student
GitHub: Haneen-Gouda
