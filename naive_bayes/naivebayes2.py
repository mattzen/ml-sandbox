import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from wordcloud import WordCloud
import requests
import io
import zipfile

# Download and extract the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
response = requests.get(url)
zip_file = zipfile.ZipFile(io.BytesIO(response.content))
file_content = zip_file.read('SMSSpamCollection')

# Load the SMS Spam Collection dataset
df = pd.read_csv(io.StringIO(file_content.decode('utf-8')), sep='\t', names=['label', 'message'])

# Preprocessing
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
X = df['message']
y = df['label']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a CountVectorizer
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train the Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_vectorized, y_train)

# Make predictions
y_pred = clf.predict(X_test_vectorized)

# Print the accuracy and classification report
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Feature Importance
feature_importance = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
feature_names = vectorizer.get_feature_names_out()

# Top 20 spam indicators
top_spam_features = feature_importance.argsort()[-20:][::-1]
plt.figure(figsize=(10, 6))
plt.bar(range(20), feature_importance[top_spam_features])
plt.xticks(range(20), [feature_names[i] for i in top_spam_features], rotation=90)
plt.title('Top 20 Spam Indicators')
plt.tight_layout()
plt.show()

# Word Cloud for Spam Messages
spam_text = ' '.join(df[df['label'] == 1]['message'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(spam_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Spam Messages')
plt.show()

# Function to classify new messages
def classify_message(message):
    message_vectorized = vectorizer.transform([message])
    prediction = clf.predict(message_vectorized)
    return "spam" if prediction[0] == 1 else "ham"

# Test the classifier with new messages
new_messages = [
    "Hello, can we reschedule our meeting?",
    "URGENT: Your account has been locked. Click here to unlock.",
    "Don't forget to pick up the kids from school"
]

for message in new_messages:
    print(f"Message: {message}")
    print(f"Classification: {classify_message(message)}\n")