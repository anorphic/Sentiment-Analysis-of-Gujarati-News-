import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

# Load the hydrated dataset
df = pd.read_csv(r'D:/SEM-1 2024/Sentiment Analysis Project/gujaratinews.csv',encoding='ISO-8859-1')

# Adjust sentiment mapping to match the expected labels for XGBoost (0, 1, 2)
sentiment_mapping = {'positive': 2, 'negative': 0, 'neutral': 1}
df['sentiment'] = df['sentiment'].map(sentiment_mapping)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df['tranlate_text'], df['sentiment'], test_size=0.2, random_state=42
)

# Create pipeline for Naive Bayes
nb_model = make_pipeline(TfidfVectorizer(), MultinomialNB())
# Train Naive Bayes model
nb_model.fit(X_train, y_train)

# Predict with Naive Bayes model
nb_y_pred = nb_model.predict(X_test)

# Evaluate Naive Bayes model
print("Naive Bayes Model Evaluation:")
print(classification_report(y_test, nb_y_pred, target_names=['negative', 'neutral', 'positive']))

# Create pipeline for XGBoost
xgb_model = make_pipeline(TfidfVectorizer(), XGBClassifier(eval_metric='mlogloss'))
# Train XGBoost model
xgb_model.fit(X_train, y_train)

# Predict with XGBoost model
xgb_y_pred = xgb_model.predict(X_test)

# Evaluate XGBoost model
print("XGBoost Model Evaluation:")
print(classification_report(y_test, xgb_y_pred, target_names=['negative', 'neutral', 'positive']))

# Function to predict sentiment for new Gujarati news
def predict_sentiment(new_text, model_type='xgb'):
    if model_type == 'nb':
        model = nb_model
    else:
        model = xgb_model
    
    prediction = model.predict([new_text])[0]
    reverse_sentiment_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
    return reverse_sentiment_mapping[prediction]

# Example usage: Enter new Gujarati news
new_gujarati_news = input("Enter Gujarati news: ")
predicted_sentiment = predict_sentiment(new_gujarati_news, model_type='xgb')
print(f"The predicted sentiment for the entered news is: {predicted_sentiment}")

