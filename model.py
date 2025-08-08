import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'text']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

x_train, x_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2)

cv = CountVectorizer()
x_train_vec = cv.fit_transform(x_train)

model = MultinomialNB()
model.fit(x_train_vec, y_train)

with open('spam_model.pkl', 'wb') as f:
    pickle.dump((cv, model), f)

