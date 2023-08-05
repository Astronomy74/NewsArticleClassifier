# -*- coding: utf-8 -*-

from itertools import product
import pandas as pd
import matplotlib.pyplot as plt
import re
import num2words
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

df = pd.read_csv("C:/Users/satur/Desktop/ANN project/12/articles.csv")


df['category'] = df['category'].astype('category')
df['title'] = df['title'].astype('string')
df['body'] = df['body'].astype('string')

df.dropna(inplace=True)

def clean_text(web_text):
    # lowercasing
    text_clean = web_text.lower()
    # converting numbers to words
    text_words = text_clean.split()
    converted_words = []
    for word in text_words:
        try:
            converted_word = num2words.num2words(int(word))
            converted_words.append(converted_word)
        except ValueError:
            converted_words.append(word)
    text_clean = ' '.join(converted_words)
    # removing special characters and numbers
    text_clean = re.sub(r'[^a-z]', ' ', text_clean)
    # removing stop words
    stop_words = set(nltk.corpus.stopwords.words("english"))
    text_clean = ' '.join([word for word in text_clean.split() if word not in stop_words])
    # lemmatization
    wordnet_lemmatizer = WordNetLemmatizer()
    lemmatized_list = []
    text_words = text_clean.split(" ")
    for word in text_words:
        lemmatized_list.append(wordnet_lemmatizer.lemmatize(word, pos="v"))
    text_clean = " ".join(lemmatized_list)
    # stemming
    stemmer = PorterStemmer()
    text_clean = stemmer.stem(text_clean)
    return text_clean


# df.category.value_counts()[:15].plot(kind='bar')
# plt.show
# df.category.value_counts()[:15].plot(kind='pie')
# plt.show

# df=df[df['category'] != 'ARTS & CULTURE'.head(500)]


# apply the text cleaning for headlines and articles
for index in df.index:
    text_headline = df.loc[index,'title']
    df.loc[index,'title'] = clean_text(text_headline)
    text_article = df.loc[index,'body']
    df.loc[index,'body'] = clean_text(text_article)

# use the model with the main body of the article for now, try with headline (title) only later
X = df['body']
y = df['category']

# hyperparameters
param_grid = {
    'n_estimators': [50, 100, 200], # number of trees in the forest to consider
    'max_depth': [25, 50, 100], # maximum depth of the tree
    'min_samples_split': [2, 5, 10], # minimum number of samples before splitting
    'min_samples_leaf': [1, 2, 3], # minimum number of samples required to be at a leaf node
    'max_features': [25, 50, 100] # number of features to consider when looking for the best split
}

# splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=74)

# vectorizing the text data
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# init the RandomForestClassifier with default parameters for now
model = RandomForestClassifier(n_jobs=5)

# list of all possible combinations of hyperparameters
param_combinations = list(product(*param_grid.values()))
# total number of parameter combinations
total_combinations = len(param_combinations)

# iterate over each parameter combination
for params in param_combinations:
    # set the hyperparameters for the model
    model.set_params(**dict(zip(param_grid.keys(), params)))
    # fit the model
    model.fit(X_train, y_train)

# predicting on the test data
y_pred_test = model.predict(X_test)
precision = precision_score(y_test, y_pred_test, average='weighted')
recall = recall_score(y_test, y_pred_test, average='weighted')
f1 = f1_score(y_test, y_pred_test, average='weighted')

print("Best Parameters: ", model.get_params())
print("Best Score: ", model.score(X_test, y_test))
print("--------")
print("Precision: ", precision)
print("Recall: ", recall)
print("F1-score: ", f1)

model_file = "C:/Users/satur/Desktop/ANN project/saved model/nac_model.pkl"
with open(model_file, 'wb') as file:
    pickle.dump(model, file)
vectorizer_file = "C:/Users/satur/Desktop/ANN project/saved model/tfidf_vectorizer.pkl"
with open(vectorizer_file, 'wb') as file:
    pickle.dump(vectorizer, file)


