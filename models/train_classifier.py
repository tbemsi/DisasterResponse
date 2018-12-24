# -*- coding: utf-8 -*-
import sys
import nltk
nltk.download(['punkt', 'wordnet'])
nltk.download('stopwords')
from nltk.corpus import stopwords

import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.externals import joblib
from sklearn.metrics import classification_report


def load_data(database_filepath):
    '''
        INPUT: database_filepath: File path where sql database is saved
        OUTPUT:
        X: Training messages
        Y: Training targets
        category_names (list): the names of the categories of the messages
        '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('FigureEight', engine)
    X = df["message"]
    Y = df.iloc[:,4:]
    category_names = list(df.columns[4:])
    return X, Y, category_names



def tokenize(text):
    '''
        INPUT: text (str): text to be tokenized
        OUTPUT: tokens (list): list of important words in text
        '''
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    
    text = re.sub("[^a-zA-Z0-9]", " ", text.lower())
    
    tokens = word_tokenize(text)
    
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return tokens

def build_model():
    '''
        INPUT: none
        OUTPUT: cv: Grid Search model
        '''
    pipeline = Pipeline([
                         ('vect', CountVectorizer(tokenizer=tokenize)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=0)))
                         ])
                         parameters = {'tfidf__smooth_idf': [True, False],
                             'clf__estimator__n_estimators':[2,5,10]
                             }
                         cv = GridSearchCV(pipeline, param_grid=parameters, scoring='accuracy', cv=2)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
        evaluates a model, providing metrics
        INPUT:
        model: python machine learning model
        X_test (DataFrame): testing dataset
        Y_test (DataFrame or Series): testing labels
        category_names (list): list of category names
        OUTPUT:
        classification_report: model evaluation metrics for each category in dataset
        '''
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names = category_names))


def save_model(model, model_filepath):
    '''
        saves a given model as a pickle file
        INPUT:
        model: python machine learning model
        model_filepath (str): the filepath to which the model should be saved
        OUTPUT:
        pickle file containing the model
        '''
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)
        
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)
        
        print('Trained model saved!')
    
    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
