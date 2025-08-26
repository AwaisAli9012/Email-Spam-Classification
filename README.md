# Spam Message Classifier

A machine learning model to classify SMS messages as spam or not spam (ham) using Logistic Regression and TF-IDF vectorization.

## 📊 Dataset
- **Source**: `spam.csv`
- **Rows**: 5,572 messages
- **Columns**: 
  - `v1`: Label (spam/ham)
  - `v2`: Message text
- **Class Distribution**: Balanced with stratified sampling

## 🔍 Methodology

### 1. Text Preprocessing
- Convert to lowercase
- Remove numbers and punctuation
- Strip extra whitespace
- Clean message content

### 2. Feature Engineering
- **TF-IDF Vectorization**: Convert text to numerical features
- **Label Encoding**: ham=0, spam=1

### 3. Model Training
- **Algorithm**: Logistic Regression
- **Train-Test Split**: 80-20 split with stratification
- **Feature Vector**: TF-IDF transformed text

## 📈 Results

| Metric | Score |
|--------|-------|
| Accuracy | 98.25% |
| Precision | 96.83% |
| Recall | 93.62% |
| F1-Score | 95.19% |

The model shows excellent performance in distinguishing spam from legitimate messages.

## 📊 Model Evaluation
- **Classification Report**: Detailed precision, recall, and F1-scores
- **Confusion Matrix**: Visual representation of true/false positives and negatives
- **High Precision**: Low false positive rate (few legitimate messages marked as spam)
- **High Recall**: Effectively catches most spam messages

## 🚀 Usage
```bash
# Clone the repository
git clone https://github.com/yourusername/spam-classifier.git

# Install requirements
pip install pandas scikit-learn matplotlib seaborn joblib

# Run the model
python spam_classifier.py

Message: Congratulations! You've won $1000! Click now to claim!
Prediction: Spam (Confidence: 0.98)

Message: Hi, are we still meeting for lunch tomorrow?
Prediction: Not Spam (Confidence: 0.99)

Message: Free entry in 2 a weekly comp to win FA Cup final tickets.
Prediction: Spam (Confidence: 0.97)

💾 Model Persistence
Trained model saved as spam_classifier_model.pkl
TF-IDF vectorizer saved as tfidf_vectorizer.pkl
Models can be loaded for future predictions
📦 Requirements
Python 3.x
pandas
scikit-learn
matplotlib
seaborn
joblib
🛠️ Key Features
Robust text preprocessing pipeline
TF-IDF for effective text representation
Logistic Regression for binary classification
Comprehensive evaluation metrics
Ready-to-use saved models
📄 License
This project is licensed under the MIT License.
