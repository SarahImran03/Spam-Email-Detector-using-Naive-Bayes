import pandas as pd
import matplotlib as mlib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB

# Preprocessing the Data
data_set = pd.read_csv('spam_ham_dataset.csv')
data_set = data_set.where((pd.notnull(data_set)),'')

x = data_set['text']
y = data_set['label']

le = LabelEncoder()
y = le.fit_transform(y)

# Split data into train and test data
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=3)

# Using TfidfVectorizer to convert text to numerical data
tfidf_vector = TfidfVectorizer(ngram_range=(1, 2))
train_x_tfidf = tfidf_vector.fit_transform(train_x)
test_x_tfidf = tfidf_vector.transform(test_x)

# Training the Model
model = MultinomialNB()
model.fit(train_x_tfidf, train_y)

# Performing a small pre-test on the Model
ham_email = ["Hi, how are you feeling?"]
ham_email_tfidf = tfidf_vector.transform(ham_email)
prediction = model.predict(ham_email_tfidf)
print("Prediction for the given email:", prediction)

spam_email = ["stock money buy"]
spam_email_tfidf = tfidf_vector.transform(spam_email)
prediction = model.predict(spam_email_tfidf)
print("Prediction for the given email:", prediction)

# Testing the Model
model_score = model.score(test_x_tfidf, test_y)
model_score *= 100
print(f"The model is {model_score} percent accurate")
